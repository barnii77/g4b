import math
import triton
from triton import language as tl
from g4b.tensor import Tensor
from g4b.kernels.utils import launch
from g4b.utils import to_int_exact


def _cfg(
    b0: int,
    b1: int,
    b2: int,
    *,
    warps: int,
    stages: int = 3,
):
    return triton.Config(
        {
            "BLOCKSIZE0": b0,
            "BLOCKSIZE1": b1,
            "BLOCKSIZE2": b2,
        },
        num_warps=warps,
        num_stages=stages,
    )


@triton.jit
def _bitonic_reduce_jfn(accum, accum_idx, tile, tile_offs):
    # TODO technically I could reverse bitonic sort `tile` only, and then do a single bitonic iter on the joined tile.
    tl.static_assert(tile_offs.shape[0] == 1)
    tl.static_assert(tile_offs.shape[1] == accum_idx.shape[1])
    tl.static_assert(tile_offs.shape[2] == accum_idx.shape[2])

    x = tl.cat(accum, tile, dim=-1)
    x_idx = tl.cat(accum_idx, tile_offs.broadcast_to(accum_idx.shape), dim=-1)
    TILESIZE: tl.constexpr = accum.shape[-1]

    n_bitonic_iters: tl.constexpr = int(math.log2(x.shape[-1]))
    for it in tl.static_range(0, n_bitonic_iters):
        cmp_idx_pattern = tl.arange(0, TILESIZE)[None, None, :] & ((1 << it + 1) - 1)
        is_reversed_desc = cmp_idx_pattern >= (1 << it)  # `<` -> ascending sort
        for inner_it in tl.static_range(0, it + 1):
            a_idx = (tl.arange(0, TILESIZE)[None, None, :] * 2).broadcast_to((accum.shape[0], accum.shape[1], TILESIZE))
            b_idx = a_idx + (1 << inner_it)
            a = x.gather(a_idx, axis=-1)
            b = x.gather(b_idx, axis=-1)
            cmp_mask = (a < b) ^ is_reversed_desc
            d_idx = (b_idx - a_idx) * cmp_mask
            a_idx += d_idx
            b_idx -= d_idx
            reorder_idx = tl.cat(a_idx, b_idx, dim=-1)
            x = x.gather(reorder_idx, axis=-1)
            x_idx = x_idx.gather(reorder_idx, axis=-1)

    split_shape: tl.constexpr = x.shape[0], x.shape[1], accum.shape[2], 2
    accum, _ = tl.split(x.reshape(split_shape))
    accum_idx, _ = tl.split(x_idx.reshape(split_shape))
    return accum, accum_idx


@triton.jit
def _bitonic_scan_find_top_k_logits_jfn(
    # fmt: off
    logits_ptr,
    B: tl.constexpr, T: tl.constexpr, D: tl.constexpr,
    stride_b: tl.constexpr, stride_t: tl.constexpr, stride_d: tl.constexpr,
    BLOCKSIZE_B: tl.constexpr, BLOCKSIZE_T: tl.constexpr, BLOCKSIZE_D: tl.constexpr,
    # fmt: on
):
    pid_t = tl.program_id(0)
    pid_b = tl.program_id(1)
    # processing across D dimension is sequential within each program

    off_b = pid_b * BLOCKSIZE_B + tl.arange(0, BLOCKSIZE_B)[:, None, None]
    off_t = pid_t * BLOCKSIZE_T + tl.arange(0, BLOCKSIZE_T)[None, :, None]

    accum = tl.full((BLOCKSIZE_B, BLOCKSIZE_T, BLOCKSIZE_D), float("-inf"), dtype=logits_ptr.dtype.element_ty)
    accum_idx = tl.full((BLOCKSIZE_B, BLOCKSIZE_T, BLOCKSIZE_D), -1, dtype=tl.int32)

    for d in tl.range(0, D, BLOCKSIZE_D):
        off_d = d + tl.arange(0, BLOCKSIZE_D)[None, None, :]
        logits_offs = off_b * stride_b + off_t * stride_t + off_d * stride_d
        logits = tl.load(logits_ptr + logits_offs, mask=(off_b < B) & (off_t < T) & (off_d < D), other=float("-inf"))
        accum, accum_idx = _bitonic_reduce_jfn(accum, accum_idx, logits, off_d)

    return accum, accum_idx


@triton.autotune(
    # fmt: off
    configs=[
        # TODO comment back in
        # # ---- decode / one sample row per program ----
        # _cfg(1, 1, 128, warps=4),
        # _cfg(1, 1, 256, warps=8),
        # _cfg(1, 1, 512, warps=8),
        # # ---- small token batching ----
        # _cfg(1, 2, 128, warps=4),
        # _cfg(1, 2, 256, warps=8),
        # _cfg(1, 4, 128, warps=4),
        # _cfg(1, 4, 256, warps=8),
        # # ---- batch batching ----
        # _cfg(2, 1, 128, warps=4),
        # _cfg(2, 1, 256, warps=8),
        _cfg(4, 1, 128, warps=4),
    ],
    # fmt: on
    key=[
        # fmt: off
        "logits_shape0", "logits_shape1", "logits_shape2",
        "out_token_ids_shape0", "out_token_ids_shape1",
        "logits_stride0", "logits_stride1", "logits_stride2",
        "out_token_ids_stride0", "out_token_ids_stride1",
        "temperature", "top_k", "top_p",
        # fmt: on
    ],
)
@triton.jit
def _sample_logits_kernel(
    # fmt: off
    logits_ptr, out_token_ids_ptr, seed,
    logits_shape0: tl.constexpr, logits_shape1: tl.constexpr, logits_shape2: tl.constexpr,
    out_token_ids_shape0: tl.constexpr, out_token_ids_shape1: tl.constexpr,
    logits_stride0: tl.constexpr, logits_stride1: tl.constexpr, logits_stride2: tl.constexpr,
    out_token_ids_stride0: tl.constexpr, out_token_ids_stride1: tl.constexpr,
    temperature: tl.constexpr, top_k: tl.constexpr, top_p: tl.constexpr,
    BLOCKSIZE0: tl.constexpr, BLOCKSIZE1: tl.constexpr, BLOCKSIZE2: tl.constexpr,
    # fmt: on
):
    # TODO this kernel could (and probably should) allow split-D processing
    tl.static_assert(logits_shape0 == out_token_ids_shape0)
    tl.static_assert(logits_shape1 == out_token_ids_shape1)
    tl.static_assert(top_k < BLOCKSIZE2)  # if I didn't do this, the kernel would be highly non-trivial
    B: tl.constexpr = logits_shape0
    T: tl.constexpr = logits_shape1
    D: tl.constexpr = logits_shape2

    # sorted in descending order
    top_BS2_logits, top_BS2_idx = _bitonic_scan_find_top_k_logits_jfn(
        logits_ptr,
        B,
        T,
        D,
        logits_stride0,
        logits_stride1,
        logits_stride2,
        BLOCKSIZE0,
        BLOCKSIZE1,
        BLOCKSIZE2,
    )
    gather_top_k_idx = tl.arange(0, top_k)[None, None, :].broadcast_to((BLOCKSIZE0, BLOCKSIZE1, top_k))
    top_k_logits = top_BS2_logits.gather(gather_top_k_idx, axis=-1)
    top_k_idx = top_BS2_idx.gather(gather_top_k_idx, axis=-1)

    probs = tl.softmax(top_k_logits / temperature, dim=-1, keep_dims=True)

    # apply top-p sampling
    p_inclusive_cumsum = probs.cumsum(axis=-1)
    within_top_p = (p_inclusive_cumsum - probs) <= top_p  # include the token which crosses top_p as well
    probs = tl.where(within_top_p, probs, 0.0)  # top-p masking
    probs /= probs.sum(axis=-1, keep_dims=True)  # renormalize
    p_inclusive_cumsum = probs.cumsum(axis=-1)

    # sample the probability distribution
    pid_t = tl.program_id(0)
    pid_b = tl.program_id(1)
    off_b = pid_b * BLOCKSIZE0 + tl.arange(0, BLOCKSIZE0)[:, None]
    off_t = pid_t * BLOCKSIZE1 + tl.arange(0, BLOCKSIZE1)[None, :]
    offs = off_b * T + off_t  # sampling grid with fake-contiguous striding (-> samples independent of mem layout)
    rands = tl.rand(seed, offs).reshape((BLOCKSIZE0, BLOCKSIZE1, 1))

    accept_mask = rands <= p_inclusive_cumsum  # transition from ...,False,False -> True,True,... at the sampled token
    token_ids_tile_idx = tl.argmin(p_inclusive_cumsum + (1 - accept_mask) * float("inf"), axis=-1, keep_dims=True)
    token_ids = top_k_idx.gather(token_ids_tile_idx.broadcast_to((BLOCKSIZE0, BLOCKSIZE1, 1)), axis=-1).reshape(
        (BLOCKSIZE0, BLOCKSIZE1)
    )

    out_token_ids_offs = off_b * out_token_ids_stride0 + off_t * out_token_ids_stride1
    tl.store(out_token_ids_ptr + out_token_ids_offs, token_ids, mask=(off_b < B) & (off_t < T))


def sample_logits(logits: Tensor, out_token_ids: Tensor, temperature: float, top_k: int, top_p: float, seed: int):
    grid_fn = lambda META: (
        triton.cdiv(logits.shape[1], META["BLOCKSIZE1"]),
        triton.cdiv(logits.shape[0], META["BLOCKSIZE0"]),
    )
    return launch[_sample_logits_kernel, grid_fn](
        logits=logits,
        out_token_ids=out_token_ids,
        temperature=float(temperature),
        top_k=to_int_exact(top_k),
        top_p=float(top_p),
        seed=to_int_exact(seed),
    )
