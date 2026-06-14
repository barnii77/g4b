import math
import triton
from triton import language as tl
from g4b.tensor import Tensor
from g4b.kernels.utils import launch, default_bencher


@triton.jit
def _rope_tile_jfn(
    # fmt: off
    x_ptr, x_rsos_ptr, x_rmsnorm_w_ptr, x_mask,
    sin, cos,
    offs_b, offs_h, offs_t, offs_k1, offs_k2,
    x_stride0: tl.constexpr, x_stride1: tl.constexpr, x_stride2: tl.constexpr, x_stride3: tl.constexpr,
    x_rsos_stride0: tl.constexpr, x_rsos_stride1: tl.constexpr, x_rsos_stride2: tl.constexpr,
    x_rmsnorm_w_stride0: tl.constexpr,
    x_shape3: tl.constexpr,
    rmsnorm_eps: tl.constexpr,
    # fmt: on
):
    start_of_head_ptrs = x_ptr + offs_b * x_stride0 + offs_h * x_stride1 + offs_t * x_stride2
    x1_ptrs = start_of_head_ptrs + offs_k1 * x_stride3
    x2_ptrs = start_of_head_ptrs + offs_k2 * x_stride3
    x1 = tl.load(x1_ptrs, mask=x_mask)
    x2 = tl.load(x2_ptrs, mask=x_mask)

    rsos_ptrs = x_rsos_ptr + offs_b * x_rsos_stride0 + offs_h * x_rsos_stride1 + offs_t * x_rsos_stride2
    rsos = tl.load(rsos_ptrs, mask=x_mask)
    inv_rms = tl.rsqrt(rsos / x_shape3 + rmsnorm_eps)
    w1 = tl.load(x_rmsnorm_w_ptr + offs_k1 * x_rmsnorm_w_stride0, mask=x_mask)
    w2 = tl.load(x_rmsnorm_w_ptr + offs_k2 * x_rmsnorm_w_stride0, mask=x_mask)
    x1 *= inv_rms * w1
    x2 *= inv_rms * w2

    y1 = cos * x1 - sin * x2
    y2 = cos * x2 + sin * x1

    tl.store(x1_ptrs, y1, mask=x_mask)
    tl.store(x2_ptrs, y2, mask=x_mask)


def _cfg(
    b0: int,
    b1: int,
    b2: int,
    b3: int,
    *,
    warps: int,
    stages: int = 3,
):
    return triton.Config(
        {
            "BLOCKSIZE0": b0,
            "BLOCKSIZE1": b1,
            "BLOCKSIZE2": b2,
            "BLOCKSIZE3": b3,
        },
        num_warps=warps,
        num_stages=stages,
    )


@triton.autotune(
    # fmt: off
    configs=[
        # ---- decode / tiny token count ----
        _cfg(1, 1, 1, 64, warps=1),
        _cfg(1, 1, 1, 128, warps=2),
        _cfg(1, 1, 1, 256, warps=4),
        _cfg(1, 2, 1, 64, warps=1),
        _cfg(1, 2, 1, 128, warps=2),
        _cfg(1, 2, 1, 256, warps=4),
        _cfg(1, 4, 1, 64, warps=2),
        _cfg(1, 4, 1, 128, warps=4),
        _cfg(1, 4, 1, 256, warps=4),
        # ---- small prefill / a few positions per program ----
        _cfg(1, 1, 2, 64, warps=1),
        _cfg(1, 1, 2, 128, warps=2),
        _cfg(1, 1, 2, 256, warps=4),
        _cfg(1, 2, 2, 64, warps=2),
        _cfg(1, 2, 2, 128, warps=4),
        _cfg(1, 1, 4, 64, warps=2),
        _cfg(1, 1, 4, 128, warps=4),
        _cfg(1, 2, 4, 64, warps=4),
        # ---- more position batching ----
        _cfg(1, 1, 8, 64, warps=4),
        _cfg(1, 1, 8, 128, warps=4),
        # ---- batch batching ----
        _cfg(2, 1, 1, 64, warps=1),
        _cfg(2, 1, 1, 128, warps=2),
        _cfg(2, 1, 1, 256, warps=4),
        _cfg(2, 2, 1, 64, warps=2),
        _cfg(4, 1, 1, 64, warps=2),
    ],
    # fmt: on
    key=[
        # fmt: off
        "q_shape0", "q_shape1", "q_shape2", "q_shape3",
        "k_shape0", "k_shape1", "k_shape2", "k_shape3",
        "q_rsos_shape0", "q_rsos_shape1", "q_rsos_shape2",
        "k_rsos_shape0", "k_rsos_shape1", "k_rsos_shape2",
        "q_rmsnorm_w_shape0",
        "k_rmsnorm_w_shape0",
        "cache_offsets_shape0",
        "q_stride0", "q_stride1", "q_stride2", "q_stride3",
        "k_stride0", "k_stride1", "k_stride2", "k_stride3",
        "q_rsos_stride0", "q_rsos_stride1", "q_rsos_stride2",
        "k_rsos_stride0", "k_rsos_stride1", "k_rsos_stride2",
        "q_rmsnorm_w_stride0",
        "k_rmsnorm_w_stride0",
        "cache_offsets_stride0",
        "rmsnorm_eps",
        # fmt: on
    ],
    do_bench=default_bencher,
    cache_results=True,
)
@triton.jit
def _apply_rope_kernel(
    # fmt: off
    q_ptr, k_ptr, rope_freqs_ptr, time_dim_offsets_ptr,
    q_rsos_ptr, k_rsos_ptr, q_rmsnorm_w_ptr, k_rmsnorm_w_ptr,
    q_shape0: tl.constexpr, q_shape1: tl.constexpr, q_shape2: tl.constexpr, q_shape3: tl.constexpr,
    k_shape0: tl.constexpr, k_shape1: tl.constexpr, k_shape2: tl.constexpr, k_shape3: tl.constexpr,
    rope_freqs_shape0: tl.constexpr,
    q_rsos_shape0: tl.constexpr, q_rsos_shape1: tl.constexpr, q_rsos_shape2: tl.constexpr,
    k_rsos_shape0: tl.constexpr, k_rsos_shape1: tl.constexpr, k_rsos_shape2: tl.constexpr,
    q_rmsnorm_w_shape0: tl.constexpr,
    k_rmsnorm_w_shape0: tl.constexpr,
    time_dim_offsets_shape0: tl.constexpr,
    q_stride0: tl.constexpr, q_stride1: tl.constexpr, q_stride2: tl.constexpr, q_stride3: tl.constexpr,
    k_stride0: tl.constexpr, k_stride1: tl.constexpr, k_stride2: tl.constexpr, k_stride3: tl.constexpr,
    rope_freqs_stride0: tl.constexpr,
    q_rsos_stride0: tl.constexpr, q_rsos_stride1: tl.constexpr, q_rsos_stride2: tl.constexpr,
    k_rsos_stride0: tl.constexpr, k_rsos_stride1: tl.constexpr, k_rsos_stride2: tl.constexpr,
    q_rmsnorm_w_stride0: tl.constexpr,
    k_rmsnorm_w_stride0: tl.constexpr,
    time_dim_offsets_stride0: tl.constexpr,
    BLOCKSIZE0: tl.constexpr, BLOCKSIZE1: tl.constexpr, BLOCKSIZE2: tl.constexpr, BLOCKSIZE3: tl.constexpr,
    rmsnorm_eps: tl.constexpr,
    # fmt: on
):
    tl.static_assert(k_shape3 // 2 == rope_freqs_shape0)
    tl.static_assert(k_shape0 == time_dim_offsets_shape0)
    tl.static_assert(k_shape0 == q_shape0)
    tl.static_assert(k_shape2 == q_shape2)
    tl.static_assert(k_shape3 == q_shape3)
    tl.static_assert(q_rsos_shape0 == q_shape0)
    tl.static_assert(q_rsos_shape1 == q_shape1)
    tl.static_assert(q_rsos_shape2 == q_shape2)
    tl.static_assert(k_rsos_shape0 == k_shape0)
    tl.static_assert(k_rsos_shape1 == k_shape1)
    tl.static_assert(k_rsos_shape2 == k_shape2)
    tl.static_assert(q_rmsnorm_w_shape0 == q_shape3)
    tl.static_assert(k_rmsnorm_w_shape0 == k_shape3)

    tl.static_assert(k_shape3 % 2 == 0)  # so my split logic makes sense
    k_split_size: tl.constexpr = k_shape3 // 2

    pid_b = tl.program_id(2)
    n_pid_h: tl.constexpr = tl.maximum(tl.cdiv(k_shape1, BLOCKSIZE1), tl.cdiv(q_shape1, BLOCKSIZE1))
    pid_h = pid_b % n_pid_h
    pid_b //= n_pid_h
    pid_t = tl.program_id(1)
    pid_k = tl.program_id(0)

    offs_b = pid_b * BLOCKSIZE0 + tl.arange(0, BLOCKSIZE0)[:, None, None, None]
    offs_h = pid_h * BLOCKSIZE1 + tl.arange(0, BLOCKSIZE1)[None, :, None, None]
    offs_t = pid_t * BLOCKSIZE2 + tl.arange(0, BLOCKSIZE2)[None, None, :, None]
    offs_k1 = pid_k * BLOCKSIZE3 + tl.arange(0, BLOCKSIZE3)[None, None, None, :]
    offs_k2 = offs_k1 + k_shape3 // 2

    _time_dim_offsets_base = tl.load(
        time_dim_offsets_ptr + offs_b * time_dim_offsets_stride0, mask=offs_b < time_dim_offsets_shape0
    )
    time_dim_offsets = _time_dim_offsets_base + offs_t

    _rope_freqs = tl.load(rope_freqs_ptr + offs_k1 * rope_freqs_stride0, mask=offs_k1 < rope_freqs_shape0)
    theta = time_dim_offsets * _rope_freqs

    sin = tl.sin(theta)
    cos = tl.cos(theta)

    k_mask = (offs_b < k_shape0) & (offs_h < k_shape1) & (offs_t < k_shape2) & (offs_k1 < k_split_size)
    q_mask = (offs_b < q_shape0) & (offs_h < q_shape1) & (offs_t < q_shape2) & (offs_k1 < k_split_size)

    # fmt: off
    _rope_tile_jfn(
        k_ptr, k_rsos_ptr, k_rmsnorm_w_ptr, k_mask,
        sin, cos,
        offs_b, offs_h, offs_t, offs_k1, offs_k2,
        k_stride0, k_stride1, k_stride2, k_stride3,
        k_rsos_stride0, k_rsos_stride1, k_rsos_stride2,
        k_rmsnorm_w_stride0,
        k_shape3,
        rmsnorm_eps,
    )
    _rope_tile_jfn(
        q_ptr, q_rsos_ptr, q_rmsnorm_w_ptr, q_mask,
        sin, cos,
        offs_b, offs_h, offs_t, offs_k1, offs_k2,
        q_stride0, q_stride1, q_stride2, q_stride3,
        q_rsos_stride0, q_rsos_stride1, q_rsos_stride2,
        q_rmsnorm_w_stride0,
        q_shape3,
        rmsnorm_eps,
    )
    # fmt: on


def apply_rope(
    q: Tensor,
    k: Tensor,
    rope_freqs: Tensor,
    time_dim_offsets: Tensor,
    q_rsos: Tensor,
    k_rsos: Tensor,
    q_rmsnorm_w: Tensor,
    k_rmsnorm_w: Tensor,
    rmsnorm_eps: float,
):
    assert rope_freqs.shape[-1] == k.shape[-1] // 2
    grid_fn = lambda META: (
        triton.cdiv(k.shape[3] // 2, META["BLOCKSIZE3"]),
        triton.cdiv(k.shape[2], META["BLOCKSIZE2"]),
        max(triton.cdiv(q.shape[1], META["BLOCKSIZE1"]), triton.cdiv(k.shape[1], META["BLOCKSIZE1"]))
        * triton.cdiv(k.shape[0], META["BLOCKSIZE0"]),
    )
    return launch[_apply_rope_kernel, grid_fn](
        q=q,
        k=k,
        rope_freqs=rope_freqs,
        time_dim_offsets=time_dim_offsets,
        q_rsos=q_rsos,
        k_rsos=k_rsos,
        q_rmsnorm_w=q_rmsnorm_w,
        k_rmsnorm_w=k_rmsnorm_w,
        rmsnorm_eps=float(rmsnorm_eps),
    )


# Only used once during model loading
@triton.jit
def _populate_rope_frequencies_kernel(
    out_ptr,
    freq_base: tl.constexpr,
    out_shape0: tl.constexpr,
    out_stride0: tl.constexpr,
    freq_scalars_stride0: tl.constexpr = None,
    freq_scalars_ptr=None,
    BLOCKSIZE: tl.constexpr = 128,
    freq_scalars: None = None,  # sink for when freq_scalars=None arg to launch[...](...)
):
    pid = tl.program_id(0)
    offs = pid * BLOCKSIZE + tl.arange(0, BLOCKSIZE)

    powers = offs.to(tl.float32) / out_shape0
    out = 1.0 / tl.exp2(math.log2(freq_base) * powers)  # 1 / freq_base ** powers

    if freq_scalars_ptr is not None:
        freq_scalars = tl.load(freq_scalars_ptr + offs * freq_scalars_stride0, mask=offs < out_shape0)
        out *= freq_scalars

    tl.store(out_ptr + offs * out_stride0, out, mask=offs < out_shape0)


def populate_rope_frequencies(out: Tensor, freq_scalars: Tensor | None, freq_base: float):
    assert freq_scalars is None or out.shape[-1] == freq_scalars.shape[-1]
    grid_fn = lambda META: (triton.cdiv(out.shape[0], META["BLOCKSIZE"]),)
    return launch[_populate_rope_frequencies_kernel, grid_fn](out=out, freq_scalars=freq_scalars, freq_base=freq_base)
