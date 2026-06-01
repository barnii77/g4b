import triton
from triton import language as tl
from typing import Literal
from g4b.tensor import Tensor, DType
from g4b.kernels.utils import launch

# TODO take epsilon parameter


# TODO autotune block sizes
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


# TODO inplace rmsnorm
# TODO maybe fuse weight elementwise mul into it optionally?
