# TODO since this kernel is probably the one which deals with the largest indices, I'll have to consider doing some
#  indexing computations in int64 explicitly instead of the implicit int32 default that you get with naive triton.
#  This does however come with a performance penalty, so maybe gate it conditionally based on input sizes. A O(GB) KV
#  cache is pretty common after all.
# TODO can I, for chunked prefill, within a qk tile, split them into sub-tiles and skip some fully masked sub-tiles?
#  tradeoff: smaller tile sizes for less wasted compute

import triton
import os
from triton import language as tl
from g4b.tensor import Tensor
from g4b.kernels.utils import launch, default_bencher, gated_configs

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
# Per-slot phase encoding: 0 -> slot unallocated (skipped by every phase), 1 -> prefill, 2 -> decode.
# A kernel launched for a given phase compares each slot's value against its PHASE constant and skips
# mismatches; reserving 0 for "unallocated" means an empty slot is skipped in both prefill and decode.
PHASE_UNALLOCATED = 0
PHASE_PREFILL = 1
PHASE_DECODE = 2
PHASE_UNALLOCATED_CONSTEXPR = tl.constexpr(0)
PHASE_PREFILL_CONSTEXPR = tl.constexpr(1)
PHASE_DECODE_CONSTEXPR = tl.constexpr(2)


@triton.jit
def _attn_inner(
    # fmt: off
    acc, l_i, m_i, q, k_cache_ptr, v_cache_ptr, time_dim_sizes_ptr,
    off_b, off_kv_h, off_kv_split, start_m,
    k_cache_stride0: tl.constexpr, k_cache_stride1: tl.constexpr, k_cache_stride2: tl.constexpr, k_cache_stride3: tl.constexpr,
    v_cache_stride0: tl.constexpr, v_cache_stride1: tl.constexpr, v_cache_stride2: tl.constexpr, v_cache_stride3: tl.constexpr,
    time_dim_sizes_stride0: tl.constexpr,
    Q_BLOCKSIZE_H: tl.constexpr, Q_BLOCKSIZE_T: tl.constexpr, HEAD_DIM: tl.constexpr, KV_BLOCKSIZE_T: tl.constexpr,
    STAGE: tl.constexpr, offs_q_t: tl.constexpr, offs_n: tl.constexpr,
    Q_CTX: tl.constexpr, KV_CTX: tl.constexpr, ctx_window_size: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr, IS_HOPPER: tl.constexpr, USE_FP32_DOT: tl.constexpr,
    # fmt: on
):
    time_dim_size = tl.load(time_dim_sizes_ptr + off_b * time_dim_sizes_stride0)
    window_size = tl.minimum(time_dim_size, ctx_window_size)
    ring_start = tl.maximum(time_dim_size - ctx_window_size, 0) % KV_CTX
    q_t_base = window_size - Q_CTX
    q_tile_start = q_t_base + start_m * Q_BLOCKSIZE_T
    # range of values handled by this stage before split-KV partitioning
    if STAGE == INNER_STAGE_OFF_BAND_CONSTEXPR:
        lo, hi = 0, q_tile_start
    elif STAGE == INNER_STAGE_ON_BAND_CONSTEXPR:
        lo, hi = q_tile_start, q_tile_start + Q_BLOCKSIZE_T
    else:
        # causal = False
        lo, hi = 0, window_size
    split_size = tl.cdiv(hi - lo, NUM_KV_SPLITS)
    lo = lo + off_kv_split * split_size
    hi = tl.minimum(hi, lo + split_size)

    offs_d = tl.arange(0, HEAD_DIM)
    # loop over k, v and update accumulator
    for start_n in tl.range(lo, hi, KV_BLOCKSIZE_T, warp_specialize=WARP_SPECIALIZE):
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
        )
        if USE_FP32_DOT:
            qk = tl.dot(q.to(tl.float32), k.to(tl.float32).T, input_precision="tf32x3")
        else:
            qk = tl.dot(q.to(tl.float16), k.to(tl.float16).T)
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
        if USE_FP32_DOT:
            acc = tl.dot(p.to(tl.float32), v.to(tl.float32), acc, input_precision="tf32x3")
        else:
            acc = tl.dot(p.to(tl.float16), v.to(tl.float16), acc)

        # update m_i and l_i
        # place this at the end of the loop to reduce register pressure
        l_i = l_i * alpha + l_ij
        m_i = m_ij

    return acc, l_i, m_i


def _cfg(m: int, n: int, splits: int, *, warps: int, stages: int):
    return triton.Config(
        {"Q_BLOCKSIZE_T": m, "KV_BLOCKSIZE_T": n, "NUM_KV_SPLITS": splits},
        num_warps=warps,
        num_stages=stages,
    )


def _configs():
    if forced := os.environ.get("G4B_FA_FORCE_CONFIG"):
        m, n, splits, warps, stages = map(int, forced.split(","))
        return [_cfg(m, n, splits, warps=warps, stages=stages)]
    return gated_configs(
        default=[
            _cfg(1, 32, 1, warps=4, stages=2),
            _cfg(1, 32, 2, warps=4, stages=2),
            _cfg(1, 32, 4, warps=4, stages=2),
            _cfg(1, 32, 8, warps=4, stages=2),
            _cfg(1, 32, 16, warps=4, stages=2),
        ],
        tuned=[
            _cfg(1, 64, 1, warps=4, stages=3),
            _cfg(1, 128, 1, warps=4, stages=3),
            _cfg(1, 64, 2, warps=4, stages=3),
            _cfg(1, 128, 2, warps=4, stages=3),
            _cfg(1, 64, 4, warps=4, stages=3),
            _cfg(1, 128, 4, warps=4, stages=3),
            _cfg(16, 32, 1, warps=4, stages=2),
            _cfg(16, 64, 1, warps=4, stages=3),
            _cfg(16, 128, 1, warps=4, stages=3),
            _cfg(16, 256, 1, warps=4, stages=3),
            _cfg(32, 32, 1, warps=4, stages=2),
            _cfg(32, 64, 1, warps=4, stages=3),
            _cfg(32, 128, 1, warps=4, stages=3),
            _cfg(32, 256, 1, warps=4, stages=3),
            _cfg(64, 32, 1, warps=4, stages=2),
            _cfg(64, 64, 1, warps=4, stages=3),
            _cfg(64, 128, 1, warps=4, stages=3),
            _cfg(64, 256, 1, warps=4, stages=3),
            _cfg(128, 32, 1, warps=4, stages=2),
            _cfg(128, 64, 1, warps=4, stages=3),
            _cfg(128, 128, 1, warps=8, stages=3),
            _cfg(128, 256, 1, warps=8, stages=3),
        ],
    )


def _prune_invalid_configs(configs, named_args, **kwargs):
    Q_CTX = kwargs["q_shape2"]
    max_kv_splits = kwargs["MAX_KV_SPLITS"]
    phase = kwargs["PHASE"]
    ctx_window_size = kwargs["ctx_window_size"]

    # Filter out configs where Q_BLOCKSIZE_T > Q_CTX
    valid = [
        conf
        for conf in configs
        if conf.kwargs.get("Q_BLOCKSIZE_T", 0) <= Q_CTX and conf.kwargs.get("NUM_KV_SPLITS", 1) <= max_kv_splits
    ]
    if os.environ.get("G4B_FA_FORCE_CONFIG"):
        return valid

    if phase != PHASE_DECODE or Q_CTX != 1:
        return [conf for conf in valid if conf.kwargs.get("NUM_KV_SPLITS", 1) == 1]

    if max_kv_splits <= 1 or ctx_window_size <= 4096:
        target_splits = 1
    elif ctx_window_size < 8192:
        target_splits = 4
    elif ctx_window_size < 32768:
        target_splits = 8
    else:
        target_splits = 16
    target_splits = min(target_splits, max_kv_splits)

    while target_splits > 1 and not any(
        conf.kwargs.get("KV_BLOCKSIZE_T") == 32 and conf.kwargs.get("NUM_KV_SPLITS") == target_splits
        for conf in valid
    ):
        target_splits //= 2

    target = [
        conf
        for conf in valid
        if conf.kwargs.get("KV_BLOCKSIZE_T") == 32 and conf.kwargs.get("NUM_KV_SPLITS") == target_splits
    ]
    return target or valid[:1]


@triton.autotune(
    configs=_configs(),
    key=[
        # fmt: off
        "q_shape0", "q_shape1", "q_shape2", "q_shape3",
        "k_cache_shape0", "k_cache_shape1", "k_cache_shape2", "k_cache_shape3",
        "v_cache_shape0", "v_cache_shape1", "v_cache_shape2", "v_cache_shape3",
        "o_shape0", "o_shape1", "o_shape2", "o_shape3",
        "time_dim_sizes_shape0", "user_in_prefill_or_decode_shape0",
        "partial_o_shape0", "partial_o_shape1", "partial_o_shape2", "partial_o_shape3", "partial_o_shape4",
        "partial_l_shape0", "partial_l_shape1", "partial_l_shape2", "partial_l_shape3",
        "partial_m_shape0", "partial_m_shape1", "partial_m_shape2", "partial_m_shape3",
        "q_stride0", "q_stride1", "q_stride2", "q_stride3",
        "k_cache_stride0", "k_cache_stride1", "k_cache_stride2", "k_cache_stride3",
        "v_cache_stride0", "v_cache_stride1", "v_cache_stride2", "v_cache_stride3",
        "o_stride0", "o_stride1", "o_stride2", "o_stride3",
        "time_dim_sizes_stride0", "user_in_prefill_or_decode_stride0",
        "partial_o_stride0", "partial_o_stride1", "partial_o_stride2", "partial_o_stride3", "partial_o_stride4",
        "partial_l_stride0", "partial_l_stride1", "partial_l_stride2", "partial_l_stride3",
        "partial_m_stride0", "partial_m_stride1", "partial_m_stride2", "partial_m_stride3",
        "ctx_window_size", "Q_BLOCKSIZE_H", "MAX_KV_SPLITS",
        "PHASE", "STAGE", "WARP_SPECIALIZE", "IS_HOPPER", "USE_FP32_DOT",
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
    ctx_window_size: tl.constexpr, Q_BLOCKSIZE_H: tl.constexpr, MAX_KV_SPLITS: tl.constexpr,
    PHASE: tl.constexpr, STAGE: tl.constexpr, WARP_SPECIALIZE: tl.constexpr, IS_HOPPER: tl.constexpr,
    USE_FP32_DOT: tl.constexpr,
    Q_BLOCKSIZE_T: tl.constexpr, KV_BLOCKSIZE_T: tl.constexpr, NUM_KV_SPLITS: tl.constexpr,
    partial_o_ptr=None, partial_l_ptr=None, partial_m_ptr=None,
    partial_o_shape0: tl.constexpr = 0, partial_o_shape1: tl.constexpr = 0, partial_o_shape2: tl.constexpr = 0, partial_o_shape3: tl.constexpr = 0, partial_o_shape4: tl.constexpr = 0,
    partial_l_shape0: tl.constexpr = 0, partial_l_shape1: tl.constexpr = 0, partial_l_shape2: tl.constexpr = 0, partial_l_shape3: tl.constexpr = 0,
    partial_m_shape0: tl.constexpr = 0, partial_m_shape1: tl.constexpr = 0, partial_m_shape2: tl.constexpr = 0, partial_m_shape3: tl.constexpr = 0,
    partial_o_stride0: tl.constexpr = 0, partial_o_stride1: tl.constexpr = 0, partial_o_stride2: tl.constexpr = 0, partial_o_stride3: tl.constexpr = 0, partial_o_stride4: tl.constexpr = 0,
    partial_l_stride0: tl.constexpr = 0, partial_l_stride1: tl.constexpr = 0, partial_l_stride2: tl.constexpr = 0, partial_l_stride3: tl.constexpr = 0,
    partial_m_stride0: tl.constexpr = 0, partial_m_stride1: tl.constexpr = 0, partial_m_stride2: tl.constexpr = 0, partial_m_stride3: tl.constexpr = 0,
    partial_o: None = None, partial_l: None = None, partial_m: None = None,
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
    tl.static_assert(NUM_KV_SPLITS <= MAX_KV_SPLITS)
    if MAX_KV_SPLITS > 1:
        tl.static_assert(partial_o_shape0 >= NUM_KV_SPLITS)
        tl.static_assert(
            partial_o_shape1 == o_shape0
            and partial_o_shape2 == o_shape1
            and partial_o_shape3 == o_shape2
            and partial_o_shape4 == o_shape3
        )
        tl.static_assert(partial_l_shape0 >= NUM_KV_SPLITS)
        tl.static_assert(partial_l_shape1 == o_shape0 and partial_l_shape2 == o_shape1 and partial_l_shape3 == o_shape2)
        tl.static_assert(partial_m_shape0 >= NUM_KV_SPLITS)
        tl.static_assert(partial_m_shape1 == o_shape0 and partial_m_shape2 == o_shape1 and partial_m_shape3 == o_shape2)
    # NOTE: q and o are permuted views of physical [B, t, H, D] scratchpads, so they are NOT contiguous in
    # [B, H, t, D] layout. The kernel only ever accesses them via explicit strides (loads/stores below), so
    # contiguity is not required for correctness; do not assert it for q/o.
    tl.static_assert(
        k_cache_stride1 == k_cache_shape2 * k_cache_stride2 and k_cache_stride0 == k_cache_shape1 * k_cache_stride1
    )
    tl.static_assert(
        v_cache_stride1 == v_cache_shape2 * v_cache_stride2 and v_cache_stride0 == v_cache_shape1 * v_cache_stride1
    )

    start_m = tl.program_id(0)
    off_kv_split = tl.program_id(1)
    off_bgh = tl.program_id(2)
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
        q_ptr + off_b * q_stride0 + offs_q_h * q_stride1 + offs_q_t * q_stride2 + offs_d * q_stride3,
        mask=q_mask,
        other=0.0,
    ).reshape((Q_BLOCKSIZE_H * Q_BLOCKSIZE_T, HEAD_DIM))
    offs_q_t_flat = offs_q_t.broadcast_to((Q_BLOCKSIZE_H, Q_BLOCKSIZE_T, 1)).reshape((Q_BLOCKSIZE_H * Q_BLOCKSIZE_T,))
    m_i = tl.zeros([Q_BLOCKSIZE_H * Q_BLOCKSIZE_T], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([Q_BLOCKSIZE_H * Q_BLOCKSIZE_T], dtype=tl.float32)
    acc = tl.zeros([Q_BLOCKSIZE_H * Q_BLOCKSIZE_T, HEAD_DIM], dtype=tl.float32)

    # stage 1: off-band
    # For causal attention this handles all complete K/V blocks before the diagonal block.
    # For full attention this handles the complete K/V sequence.
    if STAGE & STAGE_FULL_CONSTEXPR:
        INNER_STAGE: tl.constexpr = (
            INNER_STAGE_FULL_CONSTEXPR if STAGE == STAGE_FULL_CONSTEXPR else INNER_STAGE_OFF_BAND_CONSTEXPR
        )
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
            off_kv_split,
            start_m,
            k_cache_stride0,
            k_cache_stride1,
            k_cache_stride2,
            k_cache_stride3,
            v_cache_stride0,
            v_cache_stride1,
            v_cache_stride2,
            v_cache_stride3,
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
            NUM_KV_SPLITS,
            WARP_SPECIALIZE,
            IS_HOPPER,
            USE_FP32_DOT,
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
            off_kv_split,
            start_m,
            k_cache_stride0,
            k_cache_stride1,
            k_cache_stride2,
            k_cache_stride3,
            v_cache_stride0,
            v_cache_stride1,
            v_cache_stride2,
            v_cache_stride3,
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
            NUM_KV_SPLITS,
            WARP_SPECIALIZE,
            IS_HOPPER,
            USE_FP32_DOT,
        )

    # epilogue
    if NUM_KV_SPLITS > 1:
        acc = acc.reshape((Q_BLOCKSIZE_H, Q_BLOCKSIZE_T, HEAD_DIM))
        l_i = l_i.reshape((Q_BLOCKSIZE_H, Q_BLOCKSIZE_T))
        m_i = m_i.reshape((Q_BLOCKSIZE_H, Q_BLOCKSIZE_T))
        offs_q_h_2d = off_kv_h * Q_HEADS_PER_KV + off_q_h_tile * Q_BLOCKSIZE_H + tl.arange(0, Q_BLOCKSIZE_H)[:, None]
        offs_q_t_2d = start_m * Q_BLOCKSIZE_T + tl.arange(0, Q_BLOCKSIZE_T)[None, :]
        q_mask_2d = offs_q_t_2d < Q_CTX
        tl.store(
            partial_o_ptr
            + off_kv_split * partial_o_stride0
            + off_b * partial_o_stride1
            + offs_q_h * partial_o_stride2
            + offs_q_t * partial_o_stride3
            + offs_d * partial_o_stride4,
            acc,
            mask=q_mask,
        )
        tl.store(
            partial_l_ptr
            + off_kv_split * partial_l_stride0
            + off_b * partial_l_stride1
            + offs_q_h_2d * partial_l_stride2
            + offs_q_t_2d * partial_l_stride3,
            l_i,
            mask=q_mask_2d,
        )
        tl.store(
            partial_m_ptr
            + off_kv_split * partial_m_stride0
            + off_b * partial_m_stride1
            + offs_q_h_2d * partial_m_stride2
            + offs_q_t_2d * partial_m_stride3,
            m_i,
            mask=q_mask_2d,
        )
    else:
        acc = acc / l_i[:, None]
        acc = acc.reshape((Q_BLOCKSIZE_H, Q_BLOCKSIZE_T, HEAD_DIM))
        tl.store(
            o_ptr + off_b * o_stride0 + offs_q_h * o_stride1 + offs_q_t * o_stride2 + offs_d * o_stride3,
            acc.to(tl.float16),
            mask=q_mask,
        )


def _reduce_cfg(b: int, h: int, t: int, *, warps: int, stages: int = 3):
    return triton.Config(
        {
            "BLOCKSIZE_B": b,
            "BLOCKSIZE_H": h,
            "BLOCKSIZE_T": t,
        },
        num_warps=warps,
        num_stages=stages,
    )


def _reduce_configs():
    return gated_configs(
        default=[_reduce_cfg(1, 1, 1, warps=1, stages=2)],
        tuned=[
            _reduce_cfg(1, 2, 1, warps=1, stages=2),
            _reduce_cfg(1, 4, 1, warps=2, stages=2),
            _reduce_cfg(2, 1, 1, warps=1, stages=2),
            _reduce_cfg(4, 1, 1, warps=2, stages=2),
            _reduce_cfg(1, 1, 16, warps=2, stages=2),
            _reduce_cfg(1, 2, 16, warps=4, stages=2),
            _reduce_cfg(1, 4, 16, warps=4, stages=2),
            _reduce_cfg(1, 1, 32, warps=4, stages=2),
            _reduce_cfg(1, 2, 32, warps=4, stages=2),
            _reduce_cfg(1, 1, 64, warps=4, stages=3),
            _reduce_cfg(1, 2, 64, warps=4, stages=3),
        ],
    )


def _prune_invalid_reduce_configs(configs, named_args, **kwargs):
    B = kwargs["o_shape0"]
    H = kwargs["o_shape1"]
    T = kwargs["o_shape2"]
    return [
        conf
        for conf in configs
        if conf.kwargs.get("BLOCKSIZE_B", 0) <= B
        and conf.kwargs.get("BLOCKSIZE_H", 0) <= H
        and conf.kwargs.get("BLOCKSIZE_T", 0) <= T
    ]


@triton.autotune(
    configs=_reduce_configs(),
    key=[
        # fmt: off
        "partial_o_shape0", "partial_o_shape1", "partial_o_shape2", "partial_o_shape3", "partial_o_shape4",
        "partial_l_shape0", "partial_l_shape1", "partial_l_shape2", "partial_l_shape3",
        "partial_m_shape0", "partial_m_shape1", "partial_m_shape2", "partial_m_shape3",
        "o_shape0", "o_shape1", "o_shape2", "o_shape3",
        "partial_o_stride0", "partial_o_stride1", "partial_o_stride2", "partial_o_stride3", "partial_o_stride4",
        "partial_l_stride0", "partial_l_stride1", "partial_l_stride2", "partial_l_stride3",
        "partial_m_stride0", "partial_m_stride1", "partial_m_stride2", "partial_m_stride3",
        "o_stride0", "o_stride1", "o_stride2", "o_stride3",
        "NUM_KV_SPLITS",
        # fmt: on
    ],
    prune_configs_by={"early_config_prune": _prune_invalid_reduce_configs},
    do_bench=default_bencher,
    cache_results=True,
)
@triton.jit
def _reduce_split_kv_kernel(
    # fmt: off
    partial_o_ptr, partial_l_ptr, partial_m_ptr, o_ptr,
    partial_o_shape0: tl.constexpr, partial_o_shape1: tl.constexpr, partial_o_shape2: tl.constexpr, partial_o_shape3: tl.constexpr, partial_o_shape4: tl.constexpr,
    partial_l_shape0: tl.constexpr, partial_l_shape1: tl.constexpr, partial_l_shape2: tl.constexpr, partial_l_shape3: tl.constexpr,
    partial_m_shape0: tl.constexpr, partial_m_shape1: tl.constexpr, partial_m_shape2: tl.constexpr, partial_m_shape3: tl.constexpr,
    o_shape0: tl.constexpr, o_shape1: tl.constexpr, o_shape2: tl.constexpr, o_shape3: tl.constexpr,
    partial_o_stride0: tl.constexpr, partial_o_stride1: tl.constexpr, partial_o_stride2: tl.constexpr, partial_o_stride3: tl.constexpr, partial_o_stride4: tl.constexpr,
    partial_l_stride0: tl.constexpr, partial_l_stride1: tl.constexpr, partial_l_stride2: tl.constexpr, partial_l_stride3: tl.constexpr,
    partial_m_stride0: tl.constexpr, partial_m_stride1: tl.constexpr, partial_m_stride2: tl.constexpr, partial_m_stride3: tl.constexpr,
    o_stride0: tl.constexpr, o_stride1: tl.constexpr, o_stride2: tl.constexpr, o_stride3: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr, BLOCKSIZE_B: tl.constexpr, BLOCKSIZE_H: tl.constexpr, BLOCKSIZE_T: tl.constexpr,
    # fmt: on
):
    tl.static_assert(partial_o_shape1 == o_shape0)
    tl.static_assert(partial_o_shape2 == o_shape1)
    tl.static_assert(partial_o_shape3 == o_shape2)
    tl.static_assert(partial_o_shape4 == o_shape3)
    tl.static_assert(partial_l_shape0 == partial_o_shape0)
    tl.static_assert(partial_l_shape1 == o_shape0)
    tl.static_assert(partial_l_shape2 == o_shape1)
    tl.static_assert(partial_l_shape3 == o_shape2)
    tl.static_assert(partial_m_shape0 == partial_o_shape0)
    tl.static_assert(partial_m_shape1 == o_shape0)
    tl.static_assert(partial_m_shape2 == o_shape1)
    tl.static_assert(partial_m_shape3 == o_shape2)

    HEAD_DIM: tl.constexpr = o_shape3

    pid_q_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_b = pid_b * BLOCKSIZE_B + tl.arange(0, BLOCKSIZE_B)[:, None, None, None]
    offs_h = pid_h * BLOCKSIZE_H + tl.arange(0, BLOCKSIZE_H)[None, :, None, None]
    offs_t = pid_q_t * BLOCKSIZE_T + tl.arange(0, BLOCKSIZE_T)[None, None, :, None]
    offs_d = tl.arange(0, HEAD_DIM)[None, None, None, :]

    m = tl.full((BLOCKSIZE_B, BLOCKSIZE_H, BLOCKSIZE_T, 1), float("-inf"), dtype=tl.float32)
    l = tl.zeros((BLOCKSIZE_B, BLOCKSIZE_H, BLOCKSIZE_T, 1), dtype=tl.float32)
    accum = tl.zeros((BLOCKSIZE_B, BLOCKSIZE_H, BLOCKSIZE_T, HEAD_DIM), dtype=tl.float32)
    for split_i in tl.range(0, NUM_KV_SPLITS):
        accum_i = tl.load(
            partial_o_ptr
            + split_i * partial_o_stride0
            + offs_b * partial_o_stride1
            + offs_h * partial_o_stride2
            + offs_t * partial_o_stride3
            + offs_d * partial_o_stride4,
            mask=(offs_b < partial_o_shape1)
            & (offs_h < partial_o_shape2)
            & (offs_t < partial_o_shape3)
            & (offs_d < partial_o_shape4),
        )
        m_i = tl.load(
            partial_m_ptr
            + split_i * partial_m_stride0
            + offs_b * partial_m_stride1
            + offs_h * partial_m_stride2
            + offs_t * partial_m_stride3,
            mask=(offs_b < partial_m_shape1) & (offs_h < partial_m_shape2) & (offs_t < partial_m_shape3),
        )
        l_i = tl.load(
            partial_l_ptr
            + split_i * partial_l_stride0
            + offs_b * partial_l_stride1
            + offs_h * partial_l_stride2
            + offs_t * partial_l_stride3,
            mask=(offs_b < partial_l_shape1) & (offs_h < partial_l_shape2) & (offs_t < partial_l_shape3),
        )

        m_new = tl.maximum(m_i, m)
        alpha_new = tl.exp2(m_i - m_new)
        alpha_cur = tl.exp2(m - m_new)

        accum = accum * alpha_cur + accum_i.to(tl.float32) * alpha_new
        l = l * alpha_cur + l_i * alpha_new
        m = m_new

    accum /= l
    tl.store(
        o_ptr + offs_b * o_stride0 + offs_h * o_stride1 + offs_t * o_stride2 + offs_d * o_stride3,
        accum.to(o_ptr.dtype.element_ty),
        mask=(offs_b < o_shape0) & (offs_h < o_shape1) & (offs_t < o_shape2) & (offs_d < o_shape3),
    )


def flash_attention(
    q: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    o: Tensor,
    time_dim_sizes: Tensor,
    user_in_prefill_or_decode: Tensor,
    ctx_window_size: int,
    phase: int,
    partial_o: Tensor | None = None,
    partial_l: Tensor | None = None,
    partial_m: Tensor | None = None,
    *,
    use_grouped_query_tile: bool = True,
    warp_specialize: bool = False,
    is_hopper: bool = False,
    use_fp32_dot: bool = False,
    stage: int = STAGE_CAUSAL,
):
    assert isinstance(phase, int)
    q_heads_per_kv = q.shape[1] // k_cache.shape[1]
    q_blocksize_h = q_heads_per_kv if use_grouped_query_tile else 1
    if (partial_o is None) != (partial_l is None) or (partial_o is None) != (partial_m is None):
        raise ValueError("partial_o, partial_l, and partial_m must be provided together")
    max_kv_splits = min(partial_o.shape[0], partial_l.shape[0], partial_m.shape[0]) if partial_o is not None else 1
    selected_num_kv_splits = 1
    selected_meta = {}

    def grid_fn(META):
        nonlocal selected_num_kv_splits, selected_meta
        selected_num_kv_splits = META["NUM_KV_SPLITS"]
        selected_meta = dict(META)
        return (
            triton.cdiv(q.shape[2], META["Q_BLOCKSIZE_T"]),
            selected_num_kv_splits,
            q.shape[0] * k_cache.shape[1] * triton.cdiv(q_heads_per_kv, q_blocksize_h),
        )

    k1 = launch[_attn_kernel, grid_fn](
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        o=o,
        time_dim_sizes=time_dim_sizes,
        user_in_prefill_or_decode=user_in_prefill_or_decode,
        partial_o=partial_o,
        partial_l=partial_l,
        partial_m=partial_m,
        ctx_window_size=int(ctx_window_size),
        Q_BLOCKSIZE_H=q_blocksize_h,
        MAX_KV_SPLITS=max_kv_splits,
        PHASE=phase,
        WARP_SPECIALIZE=warp_specialize,
        IS_HOPPER=is_hopper,
        USE_FP32_DOT=use_fp32_dot,
        STAGE=stage,
    )
    if os.environ.get("G4B_FA_PRINT_CONFIG"):
        print(
            "fa2 selected",
            {
                "Q_BLOCKSIZE_T": selected_meta.get("Q_BLOCKSIZE_T"),
                "KV_BLOCKSIZE_T": selected_meta.get("KV_BLOCKSIZE_T"),
                "NUM_KV_SPLITS": selected_meta.get("NUM_KV_SPLITS"),
                "Q_BLOCKSIZE_H": q_blocksize_h,
            },
        )
    k2 = None
    if selected_num_kv_splits > 1:
        reduce_grid_fn = lambda META: (
            triton.cdiv(o.shape[2], META["BLOCKSIZE_T"]),
            triton.cdiv(o.shape[1], META["BLOCKSIZE_H"]),
            triton.cdiv(o.shape[0], META["BLOCKSIZE_B"]),
        )
        k2 = launch[_reduce_split_kv_kernel, reduce_grid_fn](
            partial_o=partial_o,
            partial_l=partial_l,
            partial_m=partial_m,
            o=o,
            NUM_KV_SPLITS=selected_num_kv_splits,
        )
    return k1, k2
