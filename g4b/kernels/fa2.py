# TODO skip decode-batch-dim-indices for prefill and vice versa + more advanced job assignment based on per-B seq lens
# TODO flash decode, i.e. tile across not only Q time dim but also with different tile size across KV time dim and have
#  it write into temp buffers and have a second kernel reduce the partial results.
# TODO I will also likely need to come up with fancy work partitioning strategies involving persistent kernels and
#  blackwell-style work-stealing-ish dynamically-scheduled work assignment to deal with the highly heterogeneous nature
#  of the decode phase especially but also prefill, since the context length between users differs greatly, likely
#  following a long-tailed distribution.
#  >> Skip decode-batch-dim-indices for prefill and vice versa + more advanced job assignment based on per-B seq lens
# TODO since this kernel is probably the one which deals with the largest indices, I'll have to consider doing some
#  indexing computations in int64 explicitly instead of the implicit int32 default that you get with naive triton.
#  This does however come with a performance penalty, so maybe gate it conditionally based on input sizes. A O(GB) KV
#  cache is pretty common after all.
# TODO can I, for chunked prefill, within a qk tile, split them into subtiles and skip some fully masked subtiles?
#  tradeoff: smaller tile sizes for less wasted compute
# TODO one could potentially try introducing an extra reduction loop across the head dim (innermost dim), though that
#  would increase flash attn memory traffic by ~50% so probably bad unless it massively boosts MMA throughput because
#  of better tile shapes. May be interesting though for gemma 4 specifically because of the huge 512 head dim, which
#  implies small M_tile and N_tile due to SMEM constraints.

import triton
from typing import Literal
from triton import language as tl
from g4b.tensor import Tensor
from g4b.kernels.utils import launch, default_bencher

STAGE_FULL = 1
STAGE_ON_BAND = 2
STAGE_CAUSAL = STAGE_FULL | STAGE_ON_BAND
INNER_STAGE_OFF_BAND = 1
INNER_STAGE_ON_BAND = 2
INNER_STAGE_FULL = 3
STAGE_FULL_CONSTEXPR = tl.constexpr(1)
STAGE_ON_BAND_CONSTEXPR = tl.constexpr(2)
STAGE_CAUSAL_CONSTEXPR = tl.constexpr(STAGE_FULL | STAGE_ON_BAND)
INNER_STAGE_OFF_BAND_CONSTEXPR = tl.constexpr(1)
INNER_STAGE_ON_BAND_CONSTEXPR = tl.constexpr(2)
INNER_STAGE_FULL_CONSTEXPR = tl.constexpr(3)
PHASE_PREFILL = 0
PHASE_DECODE = 1
PHASE_PREFILL_CONSTEXPR = tl.constexpr(0)
PHASE_DECODE_CONSTEXPR = tl.constexpr(1)


@triton.jit
def _attn_inner(
    # fmt: off
    acc, l_i, m_i, q, k_cache_ptr, v_cache_ptr, time_dim_sizes_ptr,
    off_b, off_kv_h, start_m,
    k_cache_stride0: tl.constexpr, k_cache_stride1: tl.constexpr, k_cache_stride2: tl.constexpr, k_cache_stride3: tl.constexpr,
    v_cache_stride0: tl.constexpr, v_cache_stride1: tl.constexpr, v_cache_stride2: tl.constexpr, v_cache_stride3: tl.constexpr,
    time_dim_sizes_stride0: tl.constexpr,
    Q_BLOCKSIZE_H: tl.constexpr, Q_BLOCKSIZE_T: tl.constexpr, HEAD_DIM: tl.constexpr, KV_BLOCKSIZE_T: tl.constexpr,
    STAGE: tl.constexpr, offs_q_t: tl.constexpr, offs_n: tl.constexpr,
    Q_CTX: tl.constexpr, KV_CTX: tl.constexpr, ctx_window_size: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr, IS_HOPPER: tl.constexpr,
    # fmt: on
):
    time_dim_size = tl.load(time_dim_sizes_ptr + off_b * time_dim_sizes_stride0)
    window_size = tl.minimum(time_dim_size, ctx_window_size)
    ring_start = tl.maximum(time_dim_size - ctx_window_size, 0) % KV_CTX
    q_t_base = window_size - Q_CTX
    q_tile_start = q_t_base + start_m * Q_BLOCKSIZE_T
    # range of values handled by this stage
    if STAGE == INNER_STAGE_OFF_BAND_CONSTEXPR:
        lo, hi = 0, q_tile_start
    elif STAGE == INNER_STAGE_ON_BAND_CONSTEXPR:
        lo, hi = q_tile_start, q_tile_start + Q_BLOCKSIZE_T
    else:
        # causal = False
        lo, hi = 0, window_size

    offs_d = tl.arange(0, HEAD_DIM)
    # loop over k, v and update accumulator
    for start_n in tl.range(lo, hi, KV_BLOCKSIZE_T, warp_specialize=WARP_SPECIALIZE):
        start_n = tl.multiple_of(start_n, KV_BLOCKSIZE_T)

        # compute qk
        kv_logical_t = start_n + offs_n
        kv_mask = kv_logical_t < hi
        offs_kv_t = (ring_start + kv_logical_t) % KV_CTX
        k = tl.load(
            k_cache_ptr
            + off_b * k_cache_stride0
            + off_kv_h * k_cache_stride1
            + offs_kv_t[:, None] * k_cache_stride2
            + offs_d[None, :] * k_cache_stride3,
            mask=kv_mask[:, None],
            other=0.0,
        ).T
        qk = tl.dot(q, k)
        qk *= 1.44269504  # 1 / log(2), because this kernel uses exp2.
        qk = qk + tl.where(kv_mask[None, :], 0, -1.0e6)

        # causal mask if required
        if STAGE == INNER_STAGE_ON_BAND_CONSTEXPR:
            q_logical_t = q_t_base + offs_q_t
            mask = q_logical_t[:, None] >= kv_logical_t[None, :]
            qk = qk + tl.where(mask, 0, -1.0e6)

        # softmax
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk)

        # compute correction factor
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, 1)

        # update output accumulator
        if not IS_HOPPER and WARP_SPECIALIZE and Q_BLOCKSIZE_T == 128 and HEAD_DIM == 128:
            BM: tl.constexpr = acc.shape[0]
            BN: tl.constexpr = acc.shape[1]
            acc0, acc1 = acc.reshape((BM, 2, BN // 2)).permute(0, 2, 1).split()
            acc0 = acc0 * alpha[:, None]
            acc1 = acc1 * alpha[:, None]
            acc = tl.join(acc0, acc1).permute(0, 2, 1).reshape((BM, BN))
        else:
            acc = acc * alpha[:, None]
        v = tl.load(
            v_cache_ptr
            + off_b * v_cache_stride0
            + off_kv_h * v_cache_stride1
            + offs_kv_t[:, None] * v_cache_stride2
            + offs_d[None, :] * v_cache_stride3,
            mask=kv_mask[:, None],
            other=0.0,
        )
        p = p.to(tl.float16)
        acc = tl.dot(p, v, acc)

        # update m_i and l_i
        # place this at the end of the loop to reduce register pressure
        l_i = l_i * alpha + l_ij
        m_i = m_ij

    return acc, l_i, m_i


def _cfg(m: int, n: int, *, warps: int, stages: int):
    return triton.Config({"Q_BLOCKSIZE_T": m, "KV_BLOCKSIZE_T": n}, num_warps=warps, num_stages=stages)


def _configs():
    return [
        # fmt: off
        _cfg(1, 32, warps=4, stages=2),
        _cfg(1, 64, warps=4, stages=3),
        _cfg(1, 128, warps=4, stages=3),
        _cfg(16, 32, warps=4, stages=2),
        _cfg(16, 64, warps=4, stages=3),
        _cfg(16, 128, warps=4, stages=3),
        _cfg(32, 32, warps=4, stages=2),
        _cfg(32, 64, warps=4, stages=3),
        _cfg(32, 128, warps=4, stages=3),
        _cfg(64, 32, warps=4, stages=2),
        _cfg(64, 64, warps=4, stages=3),
        _cfg(64, 128, warps=4, stages=3),
        _cfg(128, 32, warps=4, stages=2),
        _cfg(128, 64, warps=4, stages=3),
        _cfg(128, 128, warps=8, stages=3),
        # fmt: on
    ]


def _prune_invalid_configs(configs, named_args, **kwargs):
    Q_CTX = kwargs["q_shape2"]

    # Filter out configs where Q_BLOCKSIZE_T > Q_CTX
    return [
        conf
        for conf in configs
        if conf.kwargs.get("Q_BLOCKSIZE_T", 0) <= Q_CTX
    ]


@triton.autotune(
    configs=_configs(),
    key=[
        # fmt: off
        "q_shape0", "q_shape1", "q_shape2", "q_shape3",
        "k_cache_shape0", "k_cache_shape1", "k_cache_shape2", "k_cache_shape3",
        "v_cache_shape0", "v_cache_shape1", "v_cache_shape2", "v_cache_shape3",
        "o_shape0", "o_shape1", "o_shape2", "o_shape3",
        "time_dim_sizes_shape0", "user_in_prefill_or_decode_shape0",
        "q_stride0", "q_stride1", "q_stride2", "q_stride3",
        "k_cache_stride0", "k_cache_stride1", "k_cache_stride2", "k_cache_stride3",
        "v_cache_stride0", "v_cache_stride1", "v_cache_stride2", "v_cache_stride3",
        "o_stride0", "o_stride1", "o_stride2", "o_stride3",
        "time_dim_sizes_stride0", "user_in_prefill_or_decode_stride0",
        "ctx_window_size", "Q_BLOCKSIZE_H", "PHASE", "STAGE", "WARP_SPECIALIZE", "IS_HOPPER",
        # fmt: on
    ],
    prune_configs_by={"early_config_prune": _prune_invalid_configs},
    do_bench=default_bencher,
    cache_results=True,
)
@triton.jit
def _attn_kernel(
    # fmt: off
    q_ptr, k_cache_ptr, v_cache_ptr, o_ptr, time_dim_sizes_ptr, user_in_prefill_or_decode_ptr,
    q_shape0: tl.constexpr, q_shape1: tl.constexpr, q_shape2: tl.constexpr, q_shape3: tl.constexpr,
    k_cache_shape0: tl.constexpr, k_cache_shape1: tl.constexpr, k_cache_shape2: tl.constexpr, k_cache_shape3: tl.constexpr,
    v_cache_shape0: tl.constexpr, v_cache_shape1: tl.constexpr, v_cache_shape2: tl.constexpr, v_cache_shape3: tl.constexpr,
    o_shape0: tl.constexpr, o_shape1: tl.constexpr, o_shape2: tl.constexpr, o_shape3: tl.constexpr,
    time_dim_sizes_shape0: tl.constexpr, user_in_prefill_or_decode_shape0: tl.constexpr,
    q_stride0: tl.constexpr, q_stride1: tl.constexpr, q_stride2: tl.constexpr, q_stride3: tl.constexpr,
    k_cache_stride0: tl.constexpr, k_cache_stride1: tl.constexpr, k_cache_stride2: tl.constexpr, k_cache_stride3: tl.constexpr,
    v_cache_stride0: tl.constexpr, v_cache_stride1: tl.constexpr, v_cache_stride2: tl.constexpr, v_cache_stride3: tl.constexpr,
    o_stride0: tl.constexpr, o_stride1: tl.constexpr, o_stride2: tl.constexpr, o_stride3: tl.constexpr,
    time_dim_sizes_stride0: tl.constexpr, user_in_prefill_or_decode_stride0: tl.constexpr,
    ctx_window_size: tl.constexpr, Q_BLOCKSIZE_H: tl.constexpr,
    PHASE: tl.constexpr, STAGE: tl.constexpr, WARP_SPECIALIZE: tl.constexpr, IS_HOPPER: tl.constexpr,
    Q_BLOCKSIZE_T: tl.constexpr, KV_BLOCKSIZE_T: tl.constexpr,
    # fmt: on
):
    HEAD_DIM: tl.constexpr = q_shape3
    Q_CTX: tl.constexpr = q_shape2
    KV_CTX: tl.constexpr = k_cache_shape2
    H: tl.constexpr = q_shape1
    G: tl.constexpr = k_cache_shape1
    tl.static_assert(KV_BLOCKSIZE_T <= HEAD_DIM)
    tl.static_assert(q_shape0 == k_cache_shape0 and q_shape0 == v_cache_shape0 and q_shape0 == o_shape0)
    tl.static_assert(q_shape0 == time_dim_sizes_shape0)
    tl.static_assert(q_shape0 == user_in_prefill_or_decode_shape0)
    tl.static_assert(q_shape1 == o_shape1 and q_shape2 == o_shape2 and q_shape3 == o_shape3)
    tl.static_assert(k_cache_shape1 == v_cache_shape1 and k_cache_shape2 == v_cache_shape2)
    tl.static_assert(q_shape3 == k_cache_shape3 and q_shape3 == v_cache_shape3)
    tl.static_assert(q_shape1 % k_cache_shape1 == 0)
    tl.static_assert((q_shape1 // k_cache_shape1) % Q_BLOCKSIZE_H == 0)
    tl.static_assert(ctx_window_size <= k_cache_shape2)
    tl.static_assert(q_stride1 == q_shape2 * q_stride2 and q_stride0 == q_shape1 * q_stride1)
    tl.static_assert(k_cache_stride1 == k_cache_shape2 * k_cache_stride2 and k_cache_stride0 == k_cache_shape1 * k_cache_stride1)
    tl.static_assert(v_cache_stride1 == v_cache_shape2 * v_cache_stride2 and v_cache_stride0 == v_cache_shape1 * v_cache_stride1)
    tl.static_assert(o_stride1 == o_shape2 * o_stride2 and o_stride0 == o_shape1 * o_stride1)

    start_m = tl.program_id(0)
    off_bgh = tl.program_id(1)
    Q_HEADS_PER_KV: tl.constexpr = H // G
    Q_HEAD_TILES_PER_KV: tl.constexpr = Q_HEADS_PER_KV // Q_BLOCKSIZE_H
    off_b = off_bgh // (G * Q_HEAD_TILES_PER_KV)
    off_gqh = off_bgh % (G * Q_HEAD_TILES_PER_KV)
    off_kv_h = off_gqh // Q_HEAD_TILES_PER_KV
    off_q_h_tile = off_gqh % Q_HEAD_TILES_PER_KV

    # if this kernel is doing decode, skip users in prefill, and vice versa
    user_phase = tl.load(user_in_prefill_or_decode_ptr + off_b * user_in_prefill_or_decode_stride0)
    if user_phase != PHASE:
        return

    offs_q_h = off_kv_h * Q_HEADS_PER_KV + off_q_h_tile * Q_BLOCKSIZE_H + tl.arange(0, Q_BLOCKSIZE_H)[:, None, None]
    offs_q_t = start_m * Q_BLOCKSIZE_T + tl.arange(0, Q_BLOCKSIZE_T)[None, :, None]
    offs_d = tl.arange(0, HEAD_DIM)[None, None, :]
    offs_n = tl.arange(0, KV_BLOCKSIZE_T)

    # initialize pointer to m and l
    q_mask = offs_q_t < Q_CTX
    q = tl.load(
        q_ptr
        + off_b * q_stride0
        + offs_q_h * q_stride1
        + offs_q_t * q_stride2
        + offs_d * q_stride3,
        mask=q_mask,
        other=0.0,
    ).reshape((Q_BLOCKSIZE_H * Q_BLOCKSIZE_T, HEAD_DIM))
    offs_q_t_flat = offs_q_t.broadcast_to((Q_BLOCKSIZE_H, Q_BLOCKSIZE_T, 1)).reshape(
        (Q_BLOCKSIZE_H * Q_BLOCKSIZE_T,)
    )
    m_i = tl.zeros([Q_BLOCKSIZE_H * Q_BLOCKSIZE_T], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([Q_BLOCKSIZE_H * Q_BLOCKSIZE_T], dtype=tl.float32) + 1.0
    acc = tl.zeros([Q_BLOCKSIZE_H * Q_BLOCKSIZE_T, HEAD_DIM], dtype=tl.float32)

    # stage 1: off-band
    # For causal attention this handles all complete K/V blocks before the diagonal block.
    # For full attention this handles the complete K/V sequence.
    if STAGE & STAGE_FULL_CONSTEXPR:
        INNER_STAGE: tl.constexpr = INNER_STAGE_FULL_CONSTEXPR if STAGE == STAGE_FULL_CONSTEXPR else INNER_STAGE_OFF_BAND_CONSTEXPR
        acc, l_i, m_i = _attn_inner(
            acc,
            l_i,
            m_i,
            q,
            k_cache_ptr,
            v_cache_ptr,
            time_dim_sizes_ptr,
            off_b,
            off_kv_h,
            start_m,
            k_cache_stride0, k_cache_stride1, k_cache_stride2, k_cache_stride3,
            v_cache_stride0, v_cache_stride1, v_cache_stride2, v_cache_stride3,
            time_dim_sizes_stride0,
            Q_BLOCKSIZE_H,
            Q_BLOCKSIZE_T,
            HEAD_DIM,
            KV_BLOCKSIZE_T,
            INNER_STAGE,
            offs_q_t_flat,
            offs_n,
            Q_CTX,
            KV_CTX,
            ctx_window_size,
            WARP_SPECIALIZE,
            IS_HOPPER,
        )

    # stage 2: on-band
    if STAGE & STAGE_ON_BAND_CONSTEXPR:
        acc, l_i, m_i = _attn_inner(
            acc,
            l_i,
            m_i,
            q,
            k_cache_ptr,
            v_cache_ptr,
            time_dim_sizes_ptr,
            off_b,
            off_kv_h,
            start_m,
            k_cache_stride0, k_cache_stride1, k_cache_stride2, k_cache_stride3,
            v_cache_stride0, v_cache_stride1, v_cache_stride2, v_cache_stride3,
            time_dim_sizes_stride0,
            Q_BLOCKSIZE_H,
            Q_BLOCKSIZE_T,
            HEAD_DIM,
            KV_BLOCKSIZE_T,
            INNER_STAGE_ON_BAND_CONSTEXPR,
            offs_q_t_flat,
            offs_n,
            Q_CTX,
            KV_CTX,
            ctx_window_size,
            WARP_SPECIALIZE,
            IS_HOPPER,
        )

    # epilogue
    acc = acc / l_i[:, None]
    acc = acc.reshape((Q_BLOCKSIZE_H, Q_BLOCKSIZE_T, HEAD_DIM))
    tl.store(
        o_ptr
        + off_b * o_stride0
        + offs_q_h * o_stride1
        + offs_q_t * o_stride2
        + offs_d * o_stride3,
        acc.to(tl.float16),
        mask=q_mask,
    )


def flash_attention(
    q: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    o: Tensor,
    time_dim_sizes: Tensor,
    user_in_prefill_or_decode: Tensor,
    ctx_window_size: int,
    phase: Literal["prefill", "decode"],
    *,
    use_grouped_query_tile: bool = True,
    warp_specialize: bool = False,
    is_hopper: bool = False,
    stage: int = STAGE_CAUSAL,
):
    q_heads_per_kv = q.shape[1] // k_cache.shape[1]
    q_blocksize_h = q_heads_per_kv if use_grouped_query_tile else 1
    if phase == "prefill":
        phase_id = PHASE_PREFILL
    elif phase == "decode":
        phase_id = PHASE_DECODE
    else:
        raise ValueError(f"phase must be 'prefill' or 'decode', got {phase!r}")
    grid_fn = lambda META: (
        triton.cdiv(q.shape[2], META["Q_BLOCKSIZE_T"]),
        q.shape[0] * k_cache.shape[1] * triton.cdiv(q_heads_per_kv, q_blocksize_h),
    )
    return launch[_attn_kernel, grid_fn](
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        o=o,
        time_dim_sizes=time_dim_sizes,
        user_in_prefill_or_decode=user_in_prefill_or_decode,
        ctx_window_size=int(ctx_window_size),
        Q_BLOCKSIZE_H=q_blocksize_h,
        PHASE=phase_id,
        WARP_SPECIALIZE=warp_specialize,
        IS_HOPPER=is_hopper,
        STAGE=stage,
    )
