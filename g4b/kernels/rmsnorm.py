import triton
from triton import language as tl
from typing import Literal
from g4b.tensor import Tensor, DType
from g4b.kernels.utils import launch, default_bencher


@triton.autotune(
    configs=[
        triton.Config({"BLOCKSIZE0": 1, "BLOCKSIZE1": 128}),
        triton.Config({"BLOCKSIZE0": 16, "BLOCKSIZE1": 128}),
        triton.Config({"BLOCKSIZE0": 1, "BLOCKSIZE1": 256}),
        triton.Config({"BLOCKSIZE0": 8, "BLOCKSIZE1": 256}),
        triton.Config({"BLOCKSIZE0": 2, "BLOCKSIZE1": 256}),
        triton.Config({"BLOCKSIZE0": 4, "BLOCKSIZE1": 512}),
        triton.Config({"BLOCKSIZE0": 1, "BLOCKSIZE1": 512}),
        triton.Config({"BLOCKSIZE0": 1, "BLOCKSIZE1": 1024}),
        triton.Config({"BLOCKSIZE0": 4, "BLOCKSIZE1": 1024}),
        triton.Config({"BLOCKSIZE0": 16, "BLOCKSIZE1": 256}),
        triton.Config({"BLOCKSIZE0": 64, "BLOCKSIZE1": 512}),
        triton.Config({"BLOCKSIZE0": 32, "BLOCKSIZE1": 1024}),
    ],
    key=[
        "x_shape0",
        "x_shape1",
        "x_rsos_shape0",
        "x_rmsnorm_w_shape0",
        "x_stride0",
        "x_stride1",
        "x_rsos_stride0",
        "x_rmsnorm_w_stride0",
    ],
    do_bench=default_bencher,
    cache_results=True,
)
@triton.jit
def _finish_rmsnorm_inplace_kernel(
    x_ptr,
    x_rsos_ptr,
    x_rmsnorm_w_ptr,
    x_shape0: tl.constexpr,
    x_shape1: tl.constexpr,
    x_rsos_shape0: tl.constexpr,
    x_rmsnorm_w_shape0: tl.constexpr,
    x_stride0: tl.constexpr,
    x_stride1: tl.constexpr,
    x_rsos_stride0: tl.constexpr,
    x_rmsnorm_w_stride0: tl.constexpr,
    epsilon: tl.constexpr,
    BLOCKSIZE0: tl.constexpr,
    BLOCKSIZE1: tl.constexpr,
):
    tl.static_assert(x_rsos_shape0 == x_shape0)
    tl.static_assert(x_rmsnorm_w_shape0 == x_shape1)

    pid_b = tl.program_id(1)
    pid_d = tl.program_id(0)

    offs_b = pid_b * BLOCKSIZE0 + tl.arange(0, BLOCKSIZE0)[:, None]
    offs_d = pid_d * BLOCKSIZE1 + tl.arange(0, BLOCKSIZE1)[None, :]

    x_ptrs = x_ptr + offs_b * x_stride0 + offs_d * x_stride1
    x_mask = (offs_b < x_shape0) & (offs_d < x_shape1)
    x = tl.load(x_ptrs, mask=x_mask)
    rsos = tl.load(x_rsos_ptr + offs_b * x_rsos_stride0, mask=offs_b < x_rsos_shape0)
    if x_rmsnorm_w_ptr is not None:
        w = tl.load(x_rmsnorm_w_ptr + offs_d * x_rmsnorm_w_stride0, mask=offs_d < x_rmsnorm_w_shape0)
    else:
        w = tl.full((1, 1), 1, dtype=x.dtype)

    x /= (rsos / x_shape1 + epsilon).sqrt()
    x *= w

    tl.store(x_ptrs, x, mask=x_mask)


def finish_rmsnorm_inplace(x: Tensor, x_rsos: Tensor, x_rmsnorm_w: Tensor, epsilon: float):
    assert x.shape[:-1] == x_rsos.shape
    assert x.shape[-1:] == x_rmsnorm_w.shape

    x = x.reshape((-1, x.shape[-1]))
    x_rsos = x_rsos.reshape((-1,))
    x_rmsnorm_w = x_rmsnorm_w.reshape((-1,))

    grid_fn = lambda META: (
        triton.cdiv(x.shape[1], META["BLOCKSIZE1"]),
        triton.cdiv(x.shape[0], META["BLOCKSIZE0"]),
    )
    return launch[_finish_rmsnorm_inplace_kernel, grid_fn](
        x=x,
        x_rsos=x_rsos,
        x_rmsnorm_w=x_rmsnorm_w,
        epsilon=epsilon,
    )


@triton.jit
def _finish_rmsnorm_out_kernel(
    x_ptr,
    y_ptr,
    x_rsos_ptr,
    x_rmsnorm_w_ptr,
    x_shape0: tl.constexpr,
    x_shape1: tl.constexpr,
    y_shape0: tl.constexpr,
    y_shape1: tl.constexpr,
    x_rsos_shape0: tl.constexpr,
    x_rmsnorm_w_shape0: tl.constexpr,
    x_stride0: tl.constexpr,
    x_stride1: tl.constexpr,
    y_stride0: tl.constexpr,
    y_stride1: tl.constexpr,
    x_rsos_stride0: tl.constexpr,
    x_rmsnorm_w_stride0: tl.constexpr,
    epsilon: tl.constexpr,
    BLOCKSIZE0: tl.constexpr,
    BLOCKSIZE1: tl.constexpr,
):
    tl.static_assert(x_shape0 == y_shape0)
    tl.static_assert(x_shape1 == y_shape1)
    tl.static_assert(x_rsos_shape0 == x_shape0)
    tl.static_assert(x_rmsnorm_w_shape0 == x_shape1)

    pid_b = tl.program_id(1)
    pid_d = tl.program_id(0)
    offs_b = pid_b * BLOCKSIZE0 + tl.arange(0, BLOCKSIZE0)[:, None]
    offs_d = pid_d * BLOCKSIZE1 + tl.arange(0, BLOCKSIZE1)[None, :]
    mask = (offs_b < x_shape0) & (offs_d < x_shape1)

    x = tl.load(x_ptr + offs_b * x_stride0 + offs_d * x_stride1, mask=mask)
    rsos = tl.load(x_rsos_ptr + offs_b * x_rsos_stride0, mask=offs_b < x_rsos_shape0)
    w = tl.load(x_rmsnorm_w_ptr + offs_d * x_rmsnorm_w_stride0, mask=offs_d < x_rmsnorm_w_shape0)
    y = x * tl.rsqrt(rsos / x_shape1 + epsilon) * w
    tl.store(y_ptr + offs_b * y_stride0 + offs_d * y_stride1, y, mask=mask)


def finish_rmsnorm_out(x: Tensor, y: Tensor, x_rsos: Tensor, x_rmsnorm_w: Tensor, epsilon: float):
    assert x.shape == y.shape
    assert x.shape[:-1] == x_rsos.shape
    assert x.shape[-1:] == x_rmsnorm_w.shape
    x = x.reshape((-1, x.shape[-1]))
    y = y.reshape((-1, y.shape[-1]))
    x_rsos = x_rsos.reshape((-1,))
    x_rmsnorm_w = x_rmsnorm_w.reshape((-1,))
    grid_fn = lambda META: (
        triton.cdiv(x.shape[1], META["BLOCKSIZE1"]),
        triton.cdiv(x.shape[0], META["BLOCKSIZE0"]),
    )
    return launch[_finish_rmsnorm_out_kernel, grid_fn](
        x=x,
        y=y,
        x_rsos=x_rsos,
        x_rmsnorm_w=x_rmsnorm_w,
        epsilon=epsilon,
        BLOCKSIZE0=1,
        BLOCKSIZE1=1024,
        num_warps=4,
    )


# to do take epsilon parameter
# to do autotune block sizes
@triton.jit
def _rmsnorm_x_4d_to_y_kernel(
    y_ptr,
    x_ptr,
    x_shape0,
    x_shape1,
    x_shape2,
    x_shape3,
    y_stride0,
    y_stride1,
    y_stride2,
    y_stride3,
    x_stride0,
    x_stride1,
    x_stride2,
    x_stride3,
    BLOCKSIZE2: tl.constexpr,
    BLOCKSIZE3: tl.constexpr,
    ACCUM_DTYPE: tl.constexpr | None = None,
    OUTPUT_ACTION: tl.constexpr = "write",
):
    """
    Computes RMSNorm using tiles of BLOCKSIZE2 x BLOCKSIZE3.
    Dim 0 and 1 are handled with a separate program for each entry.
    It does not launch multiple programs in parallel per row (across dim 3).
    It uses a 3D launch grid where the program_id(0) corresponds to dim2 and program_id(1/2) corresponds to dim 1
    and dim 0 respectively.
    """
    off0 = tl.program_id(2)
    off1 = tl.program_id(1)
    off2 = tl.program_id(0) * BLOCKSIZE2

    x_desc = tl.make_tensor_descriptor(
        x_ptr,
        (x_shape0, x_shape1, x_shape2, x_shape3),
        (x_stride0, x_stride1, x_stride2, x_stride3),
        (1, 1, BLOCKSIZE2, BLOCKSIZE3),
    )
    y_desc = tl.make_tensor_descriptor(
        y_ptr,
        (x_shape0, x_shape1, x_shape2, x_shape3),
        (y_stride0, y_stride1, y_stride2, y_stride3),
        (1, 1, BLOCKSIZE2, BLOCKSIZE3),
    )

    square_accum = tl.zeros((1, 1, BLOCKSIZE2, BLOCKSIZE3), dtype=ACCUM_DTYPE or y_ptr.dtype.element_ty)
    for off3 in tl.range(0, x_shape3, BLOCKSIZE3):
        x = x_desc.load((off0, off1, off2, off3))
        x = x.to(square_accum.dtype)
        square_accum += x * x
    rms = (square_accum.sum() / x_shape3).sqrt().to(y_ptr.dtype.element_ty)
    inv_rms = 1.0 / rms

    for off3 in tl.range(0, x_shape3, BLOCKSIZE3):
        x = x_desc.load((off0, off1, off2, off3))

        tl.static_assert(OUTPUT_ACTION == "write" or OUTPUT_ACTION == "add", "Invalid OUTPUT_ACTION")
        if OUTPUT_ACTION == "write":
            y = x * inv_rms
        else:
            y = y_desc.load((off0, off1, off2, off3))
            y += x * inv_rms

        y_desc.store((off0, off1, off2, off3), y)


def _rmsnorm_x_4d_to_y(
    y: Tensor, x: Tensor, accum_dtype: DType | None = None, output_action: Literal["write", "add"] = "write"
):
    grid_fn = lambda META: (triton.cdiv(x.shape[2], META["BLOCKSIZE2"]), x.shape[1], x.shape[0])
    return launch[_rmsnorm_x_4d_to_y_kernel, grid_fn](
        y=y,
        x=x,
        ACCUM_DTYPE=(accum_dtype or x.dtype).tl_dtype,
        OUTPUT_ACTION=output_action,
        # TODO autotune these
        BLOCKSIZE2=1,
        BLOCKSIZE3=16384,
    )


def rmsnorm_x_4d_add_to_y(y: Tensor, x: Tensor, accum_dtype: DType | None = None):
    return _rmsnorm_x_4d_to_y(y, x, accum_dtype, output_action="add")


def rmsnorm_x_4d_write_to_y(y: Tensor, x: Tensor, accum_dtype: DType | None = None):
    return _rmsnorm_x_4d_to_y(y, x, accum_dtype, output_action="write")


# to do inplace rmsnorm
# to do maybe fuse weight elementwise mul into it optionally?
