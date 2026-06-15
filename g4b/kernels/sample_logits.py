import math
import triton
from triton import language as tl
from g4b.tensor import Tensor
from g4b.kernels.utils import launch, default_bencher, tanh_jfn
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
    tl.static_assert(tile_offs.shape[1] == 1)
    tl.static_assert(tile_offs.shape[2] == accum_idx.shape[2])

    x = tl.cat(accum, tile, dim=-1)
    x_idx = tl.cat(accum_idx, tile_offs.broadcast_to(accum_idx.shape), dim=-1)
    BLOCKSIZE: tl.constexpr = x.shape[-1]

    n_bitonic_iters: tl.constexpr = int(math.log2(x.shape[-1]) + 0.5)  # round(log2(shape[-1]))
    for it in tl.static_range(0, n_bitonic_iters):
        for inner_it in tl.static_range(0, it + 1):
            phase = it - inner_it
            idx = tl.arange(0, BLOCKSIZE)[None, None, :]
            other_offs = (idx ^ (1 << phase)).broadcast_to(x.shape)
            other = x.gather(other_offs, axis=-1)
            other_idx = x_idx.gather(other_offs, axis=-1)
            is_reversed = ((idx >> it + 1) ^ (idx >> phase)) & 1  # 0 -> asc, 1 -> desc cas sort for pair
            should_swap = ((x < other) ^ is_reversed) != 0
            should_swap &= x != other  # without this, x_idx would not be preserved correctly
            x = tl.where(should_swap, other, x)
            x_idx = tl.where(should_swap, other_idx, x_idx)

    split_shape: tl.constexpr = x.shape[0], x.shape[1], 2, accum.shape[2]
    accum, _ = tl.split(x.reshape(split_shape).trans(0, 1, 3, 2))
    accum_idx, _ = tl.split(x_idx.reshape(split_shape).trans(0, 1, 3, 2))
    return accum, accum_idx


@triton.jit
def _bitonic_scan_find_top_k_logits_jfn(
    # fmt: off
    logits_ptr,
    off_b, off_t, split_off_v,
    B: tl.constexpr, T: tl.constexpr, V: tl.constexpr,
    stride_b: tl.constexpr, stride_t: tl.constexpr, stride_v: tl.constexpr,
    BLOCKSIZE_B: tl.constexpr, BLOCKSIZE_T: tl.constexpr, BLOCKSIZE_V: tl.constexpr,
    ELEMS_PER_SPLIT: tl.constexpr,
    # fmt: on
):
    accum = tl.full((BLOCKSIZE_B, BLOCKSIZE_T, BLOCKSIZE_V), float("-inf"), dtype=logits_ptr.dtype.element_ty)
    accum_idx = tl.full((BLOCKSIZE_B, BLOCKSIZE_T, BLOCKSIZE_V), -1, dtype=tl.int32)

    for v in tl.range(0, ELEMS_PER_SPLIT, BLOCKSIZE_V):
        off_v = split_off_v + v
        logits_offs = off_b * stride_b + off_t * stride_t + off_v * stride_v
        logits = tl.load(logits_ptr + logits_offs, mask=(off_b < B) & (off_t < T) & (off_v < V), other=float("-inf"))
        accum, accum_idx = _bitonic_reduce_jfn(accum, accum_idx, logits, off_v)

    return accum, accum_idx


# TODO these configs assume top_k <= 64, but really I should filter them based on whether this is true
@triton.autotune(
    # fmt: off
    configs=[
        # ---- decode / one sample row per program ----
        # TODO one of these configs seems to be triggering a triton bug?
        _cfg(1, 1, 64, warps=1),
        _cfg(1, 1, 64, warps=2),
        _cfg(1, 1, 64, warps=4),
        # _cfg(1, 1, 128, warps=4),
        # _cfg(1, 1, 256, warps=8),
        # _cfg(1, 1, 512, warps=8),
        # ---- small token batching ----
        # _cfg(1, 2, 128, warps=4),
        # _cfg(1, 2, 256, warps=8),
        # _cfg(1, 4, 128, warps=4),
        # _cfg(1, 4, 256, warps=8),
        # ---- batch batching ----
        # _cfg(2, 1, 128, warps=4),
        # _cfg(2, 1, 256, warps=8),
        # _cfg(4, 1, 128, warps=4),
    ],
    # fmt: on
    key=[
        # fmt: off
        "logits_shape0", "logits_shape1", "logits_shape2",
        "out_top_k_logits_shape0", "out_top_k_logits_shape1", "out_top_k_logits_shape2",
        "out_top_k_idx_shape0", "out_top_k_idx_shape1", "out_top_k_idx_shape2",
        "logits_stride0", "logits_stride1", "logits_stride2",
        "out_top_k_logits_stride0", "out_top_k_logits_stride1", "out_top_k_logits_stride2",
        "out_top_k_idx_stride0", "out_top_k_idx_stride1", "out_top_k_idx_stride2",
        "top_k", "NUM_V_SPLITS",
        # fmt: on
    ],
    do_bench=default_bencher,
    cache_results=True,
)
@triton.jit
def _sample_logits_parallel_reduce_kernel(
    # fmt: off
    logits_ptr, out_top_k_logits_ptr, out_top_k_idx_ptr,
    logits_shape0: tl.constexpr, logits_shape1: tl.constexpr, logits_shape2: tl.constexpr,
    out_top_k_logits_shape0: tl.constexpr, out_top_k_logits_shape1: tl.constexpr, out_top_k_logits_shape2: tl.constexpr, out_top_k_logits_shape3: tl.constexpr,
    out_top_k_idx_shape0: tl.constexpr, out_top_k_idx_shape1: tl.constexpr, out_top_k_idx_shape2: tl.constexpr, out_top_k_idx_shape3: tl.constexpr,
    logits_stride0: tl.constexpr, logits_stride1: tl.constexpr, logits_stride2: tl.constexpr,
    out_top_k_logits_stride0: tl.constexpr, out_top_k_logits_stride1: tl.constexpr, out_top_k_logits_stride2: tl.constexpr, out_top_k_logits_stride3: tl.constexpr,
    out_top_k_idx_stride0: tl.constexpr, out_top_k_idx_stride1: tl.constexpr, out_top_k_idx_stride2: tl.constexpr, out_top_k_idx_stride3: tl.constexpr,
    top_k: tl.constexpr, NUM_V_SPLITS: tl.constexpr,
    BLOCKSIZE0: tl.constexpr, BLOCKSIZE1: tl.constexpr, BLOCKSIZE2: tl.constexpr,
    # fmt: on
):
    tl.static_assert(logits_shape0 == out_top_k_logits_shape0)
    tl.static_assert(logits_shape0 == out_top_k_idx_shape0)
    tl.static_assert(logits_shape1 == out_top_k_logits_shape1)
    tl.static_assert(logits_shape1 == out_top_k_idx_shape1)
    tl.static_assert(out_top_k_logits_shape2 == NUM_V_SPLITS)
    tl.static_assert(out_top_k_idx_shape2 == NUM_V_SPLITS)
    tl.static_assert(out_top_k_logits_shape3 == top_k)
    tl.static_assert(out_top_k_idx_shape3 == top_k)
    tl.static_assert(top_k <= BLOCKSIZE2)  # if I didn't do this, the kernel would be highly non-trivial
    tl.static_assert(logits_shape2 % NUM_V_SPLITS == 0)

    B: tl.constexpr = logits_shape0
    T: tl.constexpr = logits_shape1
    V: tl.constexpr = logits_shape2
    ELEMS_PER_SPLIT: tl.constexpr = V // NUM_V_SPLITS
    tl.static_assert(ELEMS_PER_SPLIT % BLOCKSIZE2 == 0)

    pid_split_v = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_b = tl.program_id(2)

    off_b = pid_b * BLOCKSIZE0 + tl.arange(0, BLOCKSIZE0)[:, None, None]
    off_t = pid_t * BLOCKSIZE1 + tl.arange(0, BLOCKSIZE1)[None, :, None]
    off_v = pid_split_v * ELEMS_PER_SPLIT + tl.arange(0, BLOCKSIZE2)[None, None, :]

    # sorted in descending order
    top_BS2_logits, top_BS2_idx = _bitonic_scan_find_top_k_logits_jfn(
        logits_ptr,
        off_b,
        off_t,
        off_v,
        B,
        T,
        V,
        logits_stride0,
        logits_stride1,
        logits_stride2,
        BLOCKSIZE0,
        BLOCKSIZE1,
        BLOCKSIZE2,
        ELEMS_PER_SPLIT,
    )
    gather_top_k_idx = tl.arange(0, top_k)[None, None, :].broadcast_to((BLOCKSIZE0, BLOCKSIZE1, top_k))
    top_k_logits = top_BS2_logits.gather(gather_top_k_idx, axis=-1)
    top_k_idx = top_BS2_idx.gather(gather_top_k_idx, axis=-1)

    # write partial top-k reduction result out to intermediate buffer
    off_v_top_k = pid_split_v * top_k + tl.arange(0, top_k)[None, None, :]
    logits_offs = (
        off_b * out_top_k_logits_stride0
        + off_t * out_top_k_logits_stride1
        + pid_split_v * out_top_k_logits_stride2
        + tl.arange(0, top_k)[None, None, :] * out_top_k_logits_stride3
    )
    tl.store(
        out_top_k_logits_ptr + logits_offs,
        top_k_logits,
        mask=(off_b < B) & (off_t < T) & (off_v_top_k < top_k * NUM_V_SPLITS),
    )
    idx_offs = (
        off_b * out_top_k_idx_stride0
        + off_t * out_top_k_idx_stride1
        + pid_split_v * out_top_k_idx_stride2
        + tl.arange(0, top_k)[None, None, :] * out_top_k_idx_stride3
    )
    tl.store(
        out_top_k_idx_ptr + idx_offs,
        top_k_idx,
        mask=(off_b < B) & (off_t < T) & (off_v_top_k < top_k * NUM_V_SPLITS),
    )


# TODO these configs assume top_k <= 64, but really I should filter them based on whether this is true
@triton.autotune(
    # fmt: off
    configs=[
        # ---- decode / one sample row per program ----
        # TODO one of these configs seems to be triggering a triton bug?
        _cfg(1, 1, 64, warps=1),
        _cfg(1, 1, 64, warps=2),
        _cfg(1, 1, 64, warps=4),
        # _cfg(1, 1, 128, warps=4),
        # _cfg(1, 1, 256, warps=8),
        # _cfg(1, 1, 512, warps=8),
        # ---- small token batching ----
        # _cfg(1, 2, 128, warps=4),
        # _cfg(1, 2, 256, warps=8),
        # _cfg(1, 4, 128, warps=4),
        # _cfg(1, 4, 256, warps=8),
        # ---- batch batching ----
        # _cfg(2, 1, 128, warps=4),
        # _cfg(2, 1, 256, warps=8),
        # _cfg(4, 1, 128, warps=4),
    ],
    # fmt: on
    key=[
        # fmt: off
        "top_k_logits_shape0", "top_k_logits_shape1", "top_k_logits_shape2", "top_k_logits_shape3",
        "top_k_idx_shape0", "top_k_idx_shape1", "top_k_idx_shape2", "top_k_idx_shape3",
        "out_token_ids_shape0", "out_token_ids_shape1",
        "seed_shape0",
        "top_k_logits_stride0", "top_k_logits_stride1", "top_k_logits_stride2", "top_k_logits_stride3",
        "top_k_idx_stride0", "top_k_idx_stride1", "top_k_idx_stride2", "top_k_idx_stride3",
        "out_token_ids_stride0", "out_token_ids_stride1",
        "seed_stride0",
        "temperature", "top_k", "top_p",
        "NUM_V_SPLITS",
        # fmt: on
    ],
    cache_results=True,
)
@triton.jit
def _sample_logits_finalize_kernel(
    # fmt: off
    top_k_logits_ptr, top_k_idx_ptr,
    out_token_ids_ptr, seed_ptr,
    top_k_logits_shape0: tl.constexpr, top_k_logits_shape1: tl.constexpr, top_k_logits_shape2: tl.constexpr, top_k_logits_shape3: tl.constexpr,
    top_k_idx_shape0: tl.constexpr, top_k_idx_shape1: tl.constexpr, top_k_idx_shape2: tl.constexpr, top_k_idx_shape3: tl.constexpr,
    out_token_ids_shape0: tl.constexpr, out_token_ids_shape1: tl.constexpr,
    seed_shape0: tl.constexpr,
    top_k_logits_stride0: tl.constexpr, top_k_logits_stride1: tl.constexpr, top_k_logits_stride2: tl.constexpr, top_k_logits_stride3: tl.constexpr,
    top_k_idx_stride0: tl.constexpr, top_k_idx_stride1: tl.constexpr, top_k_idx_stride2: tl.constexpr, top_k_idx_stride3: tl.constexpr,
    out_token_ids_stride0: tl.constexpr, out_token_ids_stride1: tl.constexpr,
    seed_stride0: tl.constexpr,
    temperature: tl.constexpr, top_k: tl.constexpr, top_p: tl.constexpr,
    NUM_V_SPLITS: tl.constexpr,
    BLOCKSIZE0: tl.constexpr, BLOCKSIZE1: tl.constexpr, BLOCKSIZE2: tl.constexpr,
    logit_softcap: tl.constexpr = None,
    # fmt: on
):
    tl.static_assert(top_k_logits_shape0 == top_k_idx_shape0)
    tl.static_assert(top_k_logits_shape1 == top_k_idx_shape1)
    tl.static_assert(top_k_logits_shape2 == top_k_idx_shape2)
    tl.static_assert(top_k_logits_shape3 == top_k_idx_shape3)
    tl.static_assert(top_k_logits_shape0 == out_token_ids_shape0)
    tl.static_assert(top_k_logits_shape1 == out_token_ids_shape1)
    tl.static_assert(top_k_logits_shape2 == NUM_V_SPLITS)
    tl.static_assert(top_k_logits_shape3 == top_k)
    tl.static_assert(top_k_idx_shape3 == top_k)
    tl.static_assert(top_k <= BLOCKSIZE2)  # if I didn't do this, the kernel would be highly non-trivial
    # contiguity requirements
    tl.static_assert(top_k_logits_stride3 == 1)
    tl.static_assert(top_k_logits_stride2 == top_k_logits_shape3)
    tl.static_assert(top_k_idx_stride3 == 1)
    tl.static_assert(top_k_idx_stride2 == top_k_idx_shape3)

    B: tl.constexpr = top_k_logits_shape0
    T: tl.constexpr = top_k_logits_shape1

    ### do the final reduction across partial top-k's -> final top-k logits/idx

    pid_t = tl.program_id(0)
    pid_b = tl.program_id(1)

    reduce_off_b = pid_b * BLOCKSIZE0 + tl.arange(0, BLOCKSIZE0)[:, None, None]
    reduce_off_t = pid_t * BLOCKSIZE1 + tl.arange(0, BLOCKSIZE1)[None, :, None]
    reduce_off_v = tl.arange(0, BLOCKSIZE2)[None, None, :]

    # sorted in descending order
    top_BS2_logits, top_BS2_idx_idx = _bitonic_scan_find_top_k_logits_jfn(
        top_k_logits_ptr,
        reduce_off_b,
        reduce_off_t,
        reduce_off_v,
        B,
        T,
        top_k_logits_shape2 * top_k_logits_shape3,
        top_k_logits_stride0,
        top_k_logits_stride1,
        1,
        BLOCKSIZE0,
        BLOCKSIZE1,
        BLOCKSIZE2,
        top_k_logits_shape2 * top_k_logits_shape3,
    )

    # since we reduced over partial reduction results, the indices reference the partial reduction result and must be
    #  remapped to token ids.
    top_k_idx_offs = reduce_off_b * top_k_idx_stride0 + reduce_off_t * top_k_idx_stride1 + top_BS2_idx_idx
    top_BS2_idx = tl.load(
        top_k_idx_ptr + top_k_idx_offs,
        mask=(reduce_off_b < B) & (reduce_off_t < T) & (top_BS2_idx_idx != -1),
        other=-1,
    )

    gather_top_k_idx = tl.arange(0, top_k)[None, None, :].broadcast_to((BLOCKSIZE0, BLOCKSIZE1, top_k))
    top_k_logits = top_BS2_logits.gather(gather_top_k_idx, axis=-1)
    top_k_idx = top_BS2_idx.gather(gather_top_k_idx, axis=-1)

    ### final reduction done - sample now

    # logit softcap
    if logit_softcap is not None:
        top_k_logits = logit_softcap * tanh_jfn(top_k_logits / logit_softcap)

    probs = tl.softmax(top_k_logits / temperature, dim=-1, keep_dims=True)

    # apply top-p sampling
    p_inclusive_cumsum = probs.cumsum(axis=-1)
    within_top_p = (p_inclusive_cumsum - probs) <= top_p  # include the token which crosses top_p as well
    probs = tl.where(within_top_p, probs, 0.0)  # top-p masking
    probs /= probs.sum(axis=-1, keep_dims=True)  # renormalize
    p_inclusive_cumsum = probs.cumsum(axis=-1)

    # sample the probability distribution
    off_b = pid_b * BLOCKSIZE0 + tl.arange(0, BLOCKSIZE0)[:, None]
    off_t = pid_t * BLOCKSIZE1 + tl.arange(0, BLOCKSIZE1)[None, :]
    offs = off_b * T + off_t  # sampling grid with fake-contiguous striding (-> samples independent of mem layout)

    # use 2-part seed (rng seed, offsets base)
    seed_tile = tl.load(seed_ptr + tl.arange(0, seed_shape0) * seed_stride0)
    seed, offs_base = tl.split(seed_tile)
    rands = tl.rand(seed, offs_base + offs).reshape((BLOCKSIZE0, BLOCKSIZE1, 1))

    accept_mask = rands <= p_inclusive_cumsum  # transition from ...,False,False -> True,True,... at the sampled token
    token_ids_tile_idx = tl.argmax(accept_mask, axis=-1, tie_break_left=True, keep_dims=True)
    token_ids = top_k_idx.gather(token_ids_tile_idx, axis=-1).reshape((BLOCKSIZE0, BLOCKSIZE1))

    out_token_ids_offs = off_b * out_token_ids_stride0 + off_t * out_token_ids_stride1
    tl.store(out_token_ids_ptr + out_token_ids_offs, token_ids, mask=(off_b < B) & (off_t < T))


@triton.jit
def _sample_logits_update_seed_kernel(
    # fmt: off
    seed_ptr,
    logits_shape1: tl.constexpr,
    seed_shape0: tl.constexpr,
    seed_stride0: tl.constexpr,
    SEED_UPDATE: tl.constexpr = 7, OFFS_BASE_UPDATE: tl.constexpr = 11,
    # fmt: on
):
    # update 2-part seed
    T: tl.constexpr = logits_shape1
    seed_ptrs = seed_ptr + tl.arange(0, seed_shape0) * seed_stride0
    seed_tile = tl.load(seed_ptrs)
    seed, offs_base = tl.split(seed_tile)
    seed_tile = tl.join(seed + SEED_UPDATE * T, offs_base + OFFS_BASE_UPDATE * T).reshape((seed_shape0,))
    tl.store(seed_ptrs, seed_tile)


def sample_logits(
    logits: Tensor,
    out_token_ids: Tensor,
    seed: Tensor,
    tmp_top_k_logits_scratchpad: Tensor,
    tmp_top_k_idx_scratchpad: Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
    NUM_V_SPLITS: int,
    logit_softcap: float | None = None,
):
    assert list(seed.shape) == [2]
    grid_fn_parallel_reduce = lambda META: (
        NUM_V_SPLITS,
        triton.cdiv(logits.shape[1], META["BLOCKSIZE1"]),
        triton.cdiv(logits.shape[0], META["BLOCKSIZE0"]),
    )
    grid_fn_finalize = lambda META: (
        triton.cdiv(logits.shape[1], META["BLOCKSIZE1"]),
        triton.cdiv(logits.shape[0], META["BLOCKSIZE0"]),
    )
    k1 = launch[_sample_logits_parallel_reduce_kernel, grid_fn_parallel_reduce](
        logits=logits,
        out_top_k_logits=tmp_top_k_logits_scratchpad,
        out_top_k_idx=tmp_top_k_idx_scratchpad,
        top_k=to_int_exact(top_k),
        NUM_V_SPLITS=NUM_V_SPLITS,
    )
    k2 = launch[_sample_logits_finalize_kernel, grid_fn_finalize](
        top_k_logits=tmp_top_k_logits_scratchpad,
        top_k_idx=tmp_top_k_idx_scratchpad,
        out_token_ids=out_token_ids,
        seed=seed,
        temperature=float(temperature),
        top_k=to_int_exact(top_k),
        top_p=float(top_p),
        NUM_V_SPLITS=NUM_V_SPLITS,
        logit_softcap=(float(logit_softcap) if logit_softcap else None),
    )
    k3 = launch[_sample_logits_update_seed_kernel, (1,)](seed=seed, logits=logits)
    return k1, k2, k3


def get_recommended_num_v_splits(V: int) -> int:
    # TODO autotune this
    return (
        # fmt: off
        32 if V >= 65536 else
        8 if V >= 16384 else
        2 if V >= 4096 else
        1
        # fmt: on
    )
