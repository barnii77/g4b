import triton
from triton import language as tl
from g4b.tensor import Tensor
from g4b.kernels.utils import launch


# TODO autotune
@triton.jit
def _add_kv_to_cache_kernel(
    x_ptr,
    cache_ptr,
    cache_offsets_ptr,
    x_shape0: tl.constexpr,
    x_shape1: tl.constexpr,
    x_shape2: tl.constexpr,
    x_shape3: tl.constexpr,
    cache_shape0: tl.constexpr,
    cache_shape1: tl.constexpr,
    cache_shape2: tl.constexpr,
    cache_shape3: tl.constexpr,
    cache_offsets_shape0: tl.constexpr,
    x_stride0: tl.constexpr,
    x_stride1: tl.constexpr,
    x_stride2: tl.constexpr,
    x_stride3: tl.constexpr,
    cache_stride0: tl.constexpr,
    cache_stride1: tl.constexpr,
    cache_stride2: tl.constexpr,
    cache_stride3: tl.constexpr,
    cache_offsets_stride0: tl.constexpr,
    BLOCKSIZE0: tl.constexpr,
    BLOCKSIZE1: tl.constexpr,
    BLOCKSIZE2: tl.constexpr,
    BLOCKSIZE3: tl.constexpr,
):
    tl.static_assert(x_shape0 == cache_shape0)
    tl.static_assert(x_shape2 == cache_shape2)
    tl.static_assert(x_shape3 == cache_shape3)
    tl.static_assert(x_shape0 == cache_offsets_shape0)

    B: tl.constexpr = x_shape0
    t: tl.constexpr = x_shape2
    T: tl.constexpr = cache_shape1
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

    cache_offs = (
        off_b * cache_stride0
        + off_g * cache_stride1
        + ((off_t + cache_offsets[:, None, None, None]) % T) * cache_stride2
        + off_d * cache_stride3
    )
    tl.store(cache_ptr + cache_offs, x, mask=(off_b < B) & (off_g < G) & (off_d < d))


def add_kv_to_cache(x: Tensor, cache: Tensor, cache_offsets: Tensor):
    grid_fn = lambda META: (
        triton.cdiv(x.shape[1], META["BLOCKSIZE1"])
        * triton.cdiv(x.shape[2], META["BLOCKSIZE2"])
        * triton.cdiv(x.shape[3], META["BLOCKSIZE3"]),
        triton.cdiv(x.shape[0], META["BLOCKSIZE0"]),
    )
    return launch[_add_kv_to_cache_kernel, grid_fn](x=x, cache=cache, cache_offsets=cache_offsets)
