import triton
from triton import language as tl
from g4b.tensor import Tensor
from g4b.kernels.memset import memset_contiguous
from g4b.kernels.utils import launch


@triton.jit
def _compute_rsos_3d_kernel(
    # fmt: off
    x_ptr, rsos_ptr,
    x_shape0: tl.constexpr, x_shape1: tl.constexpr, x_shape2: tl.constexpr,
    rsos_shape0: tl.constexpr, rsos_shape1: tl.constexpr,
    x_stride0: tl.constexpr, x_stride1: tl.constexpr, x_stride2: tl.constexpr,
    rsos_stride0: tl.constexpr, rsos_stride1: tl.constexpr,
    scale,
    BLOCKSIZE0: tl.constexpr, BLOCKSIZE1: tl.constexpr, BLOCKSIZE2: tl.constexpr,
    # fmt: on
):
    tl.static_assert(x_shape0 == rsos_shape0)
    tl.static_assert(x_shape1 == rsos_shape1)

    off0 = tl.program_id(2) * BLOCKSIZE0 + tl.arange(0, BLOCKSIZE0)[:, None, None]
    off1 = tl.program_id(1) * BLOCKSIZE1 + tl.arange(0, BLOCKSIZE1)[None, :, None]
    off2 = tl.program_id(0) * BLOCKSIZE2 + tl.arange(0, BLOCKSIZE2)[None, None, :]

    x_offsets = off0 * x_stride0 + off1 * x_stride1 + off2 * x_stride2
    x = tl.load(
        x_ptr + x_offsets,
        mask=(off0 < x_shape0) & (off1 < x_shape1) & (off2 < x_shape2),
        other=0.0,
    ).to(tl.float32)
    x *= scale

    rsos_offsets = off0 * rsos_stride0 + off1 * rsos_stride1
    tl.atomic_add(
        rsos_ptr + rsos_offsets.reshape((BLOCKSIZE0, BLOCKSIZE1)),
        (x * x).sum(-1),
        mask=((off0 < rsos_shape0) & (off1 < rsos_shape1)).reshape((BLOCKSIZE0, BLOCKSIZE1)),
    )


def compute_rsos(x: Tensor, rsos: Tensor, scale: float = 1.0):
    assert len(x.shape) >= 2
    assert rsos.shape == x.shape[:-1]
    assert rsos.is_contiguous(), "sum of squares buffer must be contiguous"

    x = x.merge_leading_dims(2)
    rsos = rsos.merge_leading_dims(1)

    grid_fn = lambda META: (
        triton.cdiv(x.shape[2], META["BLOCKSIZE2"]),
        triton.cdiv(x.shape[1], META["BLOCKSIZE1"]),
        triton.cdiv(x.shape[0], META["BLOCKSIZE0"]),
    )
    k1 = memset_contiguous(rsos, 0)
    k2 = launch[_compute_rsos_3d_kernel, grid_fn](
        x=x,
        rsos=rsos,
        scale=scale,
        BLOCKSIZE0=1,
        BLOCKSIZE1=1,
        BLOCKSIZE2=1024,
        num_warps=4,
    )
    return k1, k2
