import triton
from triton import language as tl
from g4b.tensor import Tensor
from g4b.kernels.utils import launch, default_bencher


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
        "x_shape0", "x_shape1", "x_shape2", "x_shape3",
        "cache_shape0", "cache_shape1", "cache_shape2", "cache_shape3",
        "x_rsos_shape0", "x_rsos_shape1", "x_rsos_shape2",
        "cache_offsets_shape0",
        "x_stride0", "x_stride1", "x_stride2", "x_stride3",
        "x_rsos_stride0", "x_rsos_stride1", "x_rsos_stride2",
        "cache_stride0", "cache_stride1", "cache_stride2", "cache_stride3",
        "cache_offsets_stride0",
        "rmsnorm_eps",
        # fmt: on
    ],
    do_bench=default_bencher,
    cache_results=True,
)
@triton.jit
def _add_kv_to_cache_kernel(
    # fmt: off
    x_ptr, cache_ptr, cache_offsets_ptr,
    x_shape0: tl.constexpr, x_shape1: tl.constexpr, x_shape2: tl.constexpr, x_shape3: tl.constexpr,
    cache_shape0: tl.constexpr, cache_shape1: tl.constexpr, cache_shape2: tl.constexpr, cache_shape3: tl.constexpr,
    cache_offsets_shape0: tl.constexpr,
    x_stride0: tl.constexpr, x_stride1: tl.constexpr, x_stride2: tl.constexpr, x_stride3: tl.constexpr,
    cache_stride0: tl.constexpr, cache_stride1: tl.constexpr, cache_stride2: tl.constexpr, cache_stride3: tl.constexpr,
    cache_offsets_stride0: tl.constexpr,
    BLOCKSIZE0: tl.constexpr, BLOCKSIZE1: tl.constexpr, BLOCKSIZE2: tl.constexpr, BLOCKSIZE3: tl.constexpr,
    rmsnorm_eps: tl.constexpr,
    x_rsos_ptr=None,
    x_rsos_shape0: tl.constexpr = 0, x_rsos_shape1: tl.constexpr = 0, x_rsos_shape2: tl.constexpr = 0,
    x_rsos_stride0: tl.constexpr = 0, x_rsos_stride1: tl.constexpr = 0, x_rsos_stride2: tl.constexpr = 0,
    x_rsos: None = None,
    # fmt: on
):
    tl.static_assert(x_shape0 == cache_shape0)
    tl.static_assert(x_shape1 == cache_shape1)
    tl.static_assert(x_shape3 == cache_shape3)
    tl.static_assert(x_shape0 == cache_offsets_shape0)
    if x_rsos_ptr is not None:
        tl.static_assert(x_rsos_shape0 == x_shape0)
        tl.static_assert(x_rsos_shape1 == x_shape1)
        tl.static_assert(x_rsos_shape2 == x_shape2)

    B: tl.constexpr = x_shape0
    t: tl.constexpr = x_shape2
    T: tl.constexpr = cache_shape2
    G: tl.constexpr = x_shape1  # grouped-query attention
    d: tl.constexpr = x_shape3
    tl.static_assert(t <= T)  # otherwise we would try to fill more tokens than we have kv cache

    pid_b = tl.program_id(1)
    off_b = pid_b * BLOCKSIZE0 + tl.arange(0, BLOCKSIZE0)
    cache_offsets = tl.load(cache_offsets_ptr + off_b * cache_offsets_stride0, mask=off_b < B)

    pid = tl.program_id(0)
    n_d_tiles = tl.cdiv(d, BLOCKSIZE3)
    n_g_tiles = tl.cdiv(G, BLOCKSIZE1)
    n_t_tiles = tl.cdiv(t, BLOCKSIZE2)
    pid_d = pid % n_d_tiles
    pid //= n_d_tiles
    pid_g = pid % n_g_tiles
    pid //= n_g_tiles
    pid_t = pid
    tl.device_assert(pid_t < n_t_tiles)

    off_d = pid_d * BLOCKSIZE3 + tl.arange(0, BLOCKSIZE3)[None, None, None, :]
    off_t = pid_t * BLOCKSIZE2 + tl.arange(0, BLOCKSIZE2)[None, None, :, None]
    off_g = pid_g * BLOCKSIZE1 + tl.arange(0, BLOCKSIZE1)[None, :, None, None]
    off_b = off_b[:, None, None, None]

    x_offs = off_b * x_stride0 + off_g * x_stride1 + off_t * x_stride2 + off_d * x_stride3
    x = tl.load(x_ptr + x_offs, mask=(off_b < B) & (off_g < G) & (off_t < t) & (off_d < d))
    if x_rsos_ptr is not None:
        x_rsos_offs = off_b * x_rsos_stride0 + off_g * x_rsos_stride1 + off_t * x_rsos_stride2
        x_rsos = tl.load(x_rsos_ptr + x_rsos_offs, mask=(off_b < B) & (off_g < G) & (off_t < t))
        x *= tl.rsqrt(x_rsos / d + rmsnorm_eps)

    cache_offs = (
        off_b * cache_stride0
        + off_g * cache_stride1
        + ((off_t + cache_offsets[:, None, None, None]) % T) * cache_stride2
        + off_d * cache_stride3
    )
    tl.store(cache_ptr + cache_offs, x, mask=(off_b < B) & (off_g < G) & (off_t < t) & (off_d < d))


def add_kv_to_cache(x: Tensor, cache: Tensor, cache_offsets: Tensor, rmsnorm_eps: float, x_rsos: Tensor | None = None):
    grid_fn = lambda META: (
        triton.cdiv(x.shape[1], META["BLOCKSIZE1"])
        * triton.cdiv(x.shape[2], META["BLOCKSIZE2"])
        * triton.cdiv(x.shape[3], META["BLOCKSIZE3"]),
        triton.cdiv(x.shape[0], META["BLOCKSIZE0"]),
    )
    return launch[_add_kv_to_cache_kernel, grid_fn](
        x=x,
        cache=cache,
        cache_offsets=cache_offsets,
        x_rsos=x_rsos,
        rmsnorm_eps=rmsnorm_eps,
    )
