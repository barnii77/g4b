# TODO separate Q (current from residual) from KV (cache) time dim size
# TODO use context_window_offsets
# TODO skip decode-batch-dim-indices for prefill and vice versa + more advanced job assignment based on per-B seq lens
# TODO GQA support
# TODO foreach KV group (GQA) I should load all corresponding queries in a single thread-block so I can get a
#  ((g*tile_T1) x d_k) @ (d_k x tile_T2) tl.dot operation -> better arithmetic intensity than loading KV tiles for each
#  query separately.
# TODO flash decode, i.e. tile across not only Q time dim but also with different tile size across KV time dim and have
#  it write into temp buffers and have a second kernel reduce the partial results.
# TODO I will also likely need to come up with fancy work partitioning strategies involving persistent kernels and
#  blackwell-style work-stealing-ish dynamically-scheduled work assignment to deal with the highly heterogeneous nature
#  of the decode phase especially but also prefill, since the context length between users differs greatly, likely
#  following a long-tailed distribution.
# TODO since this kernel is probably the one which deals with the largest indices, I'll have to consider doing some
#  indexing computations in int64 explicitly instead of the implicit int32 default that you get with naive triton.
#  This does however come with a performance penalty, so maybe gate it conditionally based on input sizes. A O(GB) KV
#  cache is pretty common after all.
# TODO one could potentially try introducing an extra reduction loop across the head dim (innermost dim), though that
#  would increase flash attn memory traffic by ~50% so probably bad unless it massively boosts MMA throughput because
#  of better tile shapes. May be interesting though for gemma 4 specifically because of the huge 512 head dim, which
#  implies small M_tile and N_tile due to SMEM constraints.

import triton
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


@triton.jit
def _attn_inner(
    # fmt: off
    acc, l_i, m_i, q, desc_k, desc_v,
    offset_y, start_m,
    Q_BLOCKSIZE_T: tl.constexpr, HEAD_DIM: tl.constexpr, KV_BLOCKSIZE_T: tl.constexpr,
    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,
    N_CTX: tl.constexpr, WARP_SPECIALIZE: tl.constexpr, IS_HOPPER: tl.constexpr,
    # fmt: on
):
    # range of values handled by this stage
    if STAGE == INNER_STAGE_OFF_BAND_CONSTEXPR:
        lo, hi = 0, start_m * Q_BLOCKSIZE_T
    elif STAGE == INNER_STAGE_ON_BAND_CONSTEXPR:
        lo, hi = start_m * Q_BLOCKSIZE_T, (start_m + 1) * Q_BLOCKSIZE_T
        lo = tl.multiple_of(lo, Q_BLOCKSIZE_T)
    else:
        # causal = False
        lo, hi = 0, N_CTX

    offsetk_y = offset_y + lo
    offsetv_y = offset_y + lo
    # loop over k, v and update accumulator
    for start_n in tl.range(lo, hi, KV_BLOCKSIZE_T, warp_specialize=WARP_SPECIALIZE):
        start_n = tl.multiple_of(start_n, KV_BLOCKSIZE_T)

        # compute qk
        k = desc_k.load([offsetk_y, 0]).T
        qk = tl.dot(q, k)
        qk *= 1.44269504  # 1 / log(2), because this kernel uses exp2.

        # causal mask if required
        if STAGE == INNER_STAGE_ON_BAND_CONSTEXPR:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
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
        v = desc_v.load([offsetv_y, 0])
        p = p.to(tl.float16)
        acc = tl.dot(p, v, acc)

        # update m_i and l_i
        # place this at the end of the loop to reduce register pressure
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        offsetk_y += KV_BLOCKSIZE_T
        offsetv_y += KV_BLOCKSIZE_T

    return acc, l_i, m_i


def _cfg(m: int, n: int, *, warps: int, stages: int):
    return triton.Config({"Q_BLOCKSIZE_T": m, "KV_BLOCKSIZE_T": n}, num_warps=warps, num_stages=stages)


def _configs():
    return [
        # fmt: off
        _cfg(64, 32, warps=4, stages=2),
        _cfg(64, 64, warps=4, stages=3),
        _cfg(64, 128, warps=4, stages=3),
        _cfg(128, 32, warps=4, stages=2),
        _cfg(128, 64, warps=4, stages=3),
        _cfg(128, 128, warps=8, stages=3),
        # fmt: on
    ]


def _prune_invalid_configs(configs, named_args, **kwargs):
    N_CTX = kwargs["q_shape2"]
    STAGE = kwargs["STAGE"]

    # Filter out configs where Q_BLOCKSIZE_T > N_CTX
    # Filter out configs where Q_BLOCKSIZE_T < KV_BLOCKSIZE_T when causal is True
    return [
        conf
        for conf in configs
        if conf.kwargs.get("Q_BLOCKSIZE_T", 0) <= N_CTX
        and (conf.kwargs.get("Q_BLOCKSIZE_T", 0) >= conf.kwargs.get("KV_BLOCKSIZE_T", 0) or STAGE == STAGE_FULL)
    ]


@triton.autotune(
    configs=_configs(),
    key=[
        # fmt: off
        "q_shape0", "q_shape1", "q_shape2", "q_shape3",
        "k_shape0", "k_shape1", "k_shape2", "k_shape3",
        "v_shape0", "v_shape1", "v_shape2", "v_shape3",
        "o_shape0", "o_shape1", "o_shape2", "o_shape3",
        "q_stride0", "q_stride1", "q_stride2", "q_stride3",
        "k_stride0", "k_stride1", "k_stride2", "k_stride3",
        "v_stride0", "v_stride1", "v_stride2", "v_stride3",
        "o_stride0", "o_stride1", "o_stride2", "o_stride3",
        "STAGE", "WARP_SPECIALIZE", "IS_HOPPER",
        # fmt: on
    ],
    prune_configs_by={"early_config_prune": _prune_invalid_configs},
    do_bench=default_bencher,
    cache_results=True,
)
@triton.jit
def _attn_kernel(
    # fmt: off
    q_ptr, k_ptr, v_ptr, o_ptr,
    q_shape0: tl.constexpr, q_shape1: tl.constexpr, q_shape2: tl.constexpr, q_shape3: tl.constexpr,
    k_shape0: tl.constexpr, k_shape1: tl.constexpr, k_shape2: tl.constexpr, k_shape3: tl.constexpr,
    v_shape0: tl.constexpr, v_shape1: tl.constexpr, v_shape2: tl.constexpr, v_shape3: tl.constexpr,
    o_shape0: tl.constexpr, o_shape1: tl.constexpr, o_shape2: tl.constexpr, o_shape3: tl.constexpr,
    q_stride0: tl.constexpr, q_stride1: tl.constexpr, q_stride2: tl.constexpr, q_stride3: tl.constexpr,
    k_stride0: tl.constexpr, k_stride1: tl.constexpr, k_stride2: tl.constexpr, k_stride3: tl.constexpr,
    v_stride0: tl.constexpr, v_stride1: tl.constexpr, v_stride2: tl.constexpr, v_stride3: tl.constexpr,
    o_stride0: tl.constexpr, o_stride1: tl.constexpr, o_stride2: tl.constexpr, o_stride3: tl.constexpr,
    STAGE: tl.constexpr, WARP_SPECIALIZE: tl.constexpr, IS_HOPPER: tl.constexpr,
    Q_BLOCKSIZE_T: tl.constexpr, KV_BLOCKSIZE_T: tl.constexpr,
    # fmt: on
):
    HEAD_DIM: tl.constexpr = q_shape3
    N_CTX: tl.constexpr = q_shape2
    H: tl.constexpr = q_shape1
    tl.static_assert(KV_BLOCKSIZE_T <= HEAD_DIM)
    tl.static_assert(q_shape0 == k_shape0 and q_shape0 == v_shape0 and q_shape0 == o_shape0)
    tl.static_assert(q_shape1 == k_shape1 and q_shape1 == v_shape1 and q_shape1 == o_shape1)
    tl.static_assert(q_shape2 == k_shape2 and q_shape2 == v_shape2 and q_shape2 == o_shape2)
    tl.static_assert(q_shape3 == k_shape3 and q_shape3 == v_shape3 and q_shape3 == o_shape3)
    tl.static_assert(q_stride1 == q_shape2 * q_stride2 and q_stride0 == q_shape1 * q_stride1)
    tl.static_assert(k_stride1 == k_shape2 * k_stride2 and k_stride0 == k_shape1 * k_stride1)
    tl.static_assert(v_stride1 == v_shape2 * v_stride2 and v_stride0 == v_shape1 * v_stride1)
    tl.static_assert(o_stride1 == o_shape2 * o_stride2 and o_stride0 == o_shape1 * o_stride1)

    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    y_dim: tl.constexpr = q_shape0 * q_shape1 * q_shape2
    # fmt: off
    desc_q = tl.make_tensor_descriptor(
        q_ptr, shape=(y_dim, HEAD_DIM), strides=(q_stride2, q_stride3), block_shape=(Q_BLOCKSIZE_T, HEAD_DIM)
    )
    desc_k = tl.make_tensor_descriptor(
        k_ptr, shape=(y_dim, HEAD_DIM), strides=(k_stride2, k_stride3), block_shape=(KV_BLOCKSIZE_T, HEAD_DIM)
    )
    desc_v = tl.make_tensor_descriptor(
        v_ptr, shape=(y_dim, HEAD_DIM), strides=(v_stride2, v_stride3), block_shape=(KV_BLOCKSIZE_T, HEAD_DIM)
    )
    desc_o = tl.make_tensor_descriptor(
        o_ptr, shape=(y_dim, HEAD_DIM), strides=(o_stride2, o_stride3), block_shape=(Q_BLOCKSIZE_T, HEAD_DIM)
    )
    # fmt: on

    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * Q_BLOCKSIZE_T
    # initialize offsets
    offs_m = start_m * Q_BLOCKSIZE_T + tl.arange(0, Q_BLOCKSIZE_T)
    offs_n = tl.arange(0, KV_BLOCKSIZE_T)

    # initialize pointer to m and l
    m_i = tl.zeros([Q_BLOCKSIZE_T], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([Q_BLOCKSIZE_T], dtype=tl.float32) + 1.0
    acc = tl.zeros([Q_BLOCKSIZE_T, HEAD_DIM], dtype=tl.float32)
    # load q: it will stay in SRAM throughout
    q = desc_q.load([qo_offset_y, 0])

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
            desc_k,
            desc_v,
            offset_y,
            start_m,
            Q_BLOCKSIZE_T,
            HEAD_DIM,
            KV_BLOCKSIZE_T,
            INNER_STAGE,
            offs_m,
            offs_n,
            N_CTX,
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
            desc_k,
            desc_v,
            offset_y,
            start_m,
            Q_BLOCKSIZE_T,
            HEAD_DIM,
            KV_BLOCKSIZE_T,
            INNER_STAGE_ON_BAND_CONSTEXPR,
            offs_m,
            offs_n,
            N_CTX,
            WARP_SPECIALIZE,
            IS_HOPPER,
        )

    # epilogue
    acc = acc / l_i[:, None]
    desc_o.store([qo_offset_y, 0], acc.to(tl.float16))


def flash_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    o: Tensor,
    *,
    warp_specialize: bool = False,
    is_hopper: bool = False,
    stage: int = STAGE_CAUSAL,
):
    grid_fn = lambda META: (
        triton.cdiv(q.shape[2], META["Q_BLOCKSIZE_T"]),
        q.shape[0] * q.shape[1],
    )
    return launch[_attn_kernel, grid_fn](
        q=q,
        k=k,
        v=v,
        o=o,
        WARP_SPECIALIZE=warp_specialize,
        IS_HOPPER=is_hopper,
        STAGE=stage,
    )
