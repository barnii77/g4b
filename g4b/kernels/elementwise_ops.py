import triton
from triton import language as tl
from typing import Literal
from g4b.tensor import Tensor
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
        # fmt: off
        "OP",
        "output_scale_factor",
        "rmsnorm_eps",
        "a_shape0", "a_shape1",
        "b_shape0", "b_shape1",
        "b_rsos_shape0",
        "b_rmsnorm_w_shape0",
        "out_shape0",  "out_shape1",
        "a_stride0", "a_stride1",
        "b_stride0", "b_stride1",
        "b_rsos_stride0",
        "b_rmsnorm_w_stride0",
        "out_stride0", "out_stride1",
        # fmt: on
    ],
    do_bench=default_bencher,
    cache_results=True,
)
@triton.jit
def _elementwise_2d_kernel(
    # fmt: off
    a_ptr, b_ptr, out_ptr,
    OP: tl.constexpr,
    output_scale_factor: tl.constexpr,
    rmsnorm_eps: tl.constexpr,
    a_shape0: tl.constexpr, a_shape1: tl.constexpr,
    b_shape0: tl.constexpr, b_shape1: tl.constexpr,
    out_shape0: tl.constexpr, out_shape1: tl.constexpr,
    a_stride0: tl.constexpr, a_stride1: tl.constexpr,
    b_stride0: tl.constexpr, b_stride1: tl.constexpr,
    out_stride0: tl.constexpr, out_stride1: tl.constexpr,
    BLOCKSIZE0: tl.constexpr, BLOCKSIZE1: tl.constexpr,
    b_rsos_ptr=None,
    b_rmsnorm_w_ptr=None,
    b_rsos_shape0: tl.constexpr = 0,
    b_rmsnorm_w_shape0: tl.constexpr = 0,
    b_rsos_stride0: tl.constexpr = 0,
    b_rmsnorm_w_stride0: tl.constexpr = 0,
    # these args are here so when optional tensors are None and launch doesn't decompose them, it doesn't error
    b_rsos: None = None,
    b_rmsnorm_w: None = None,
    # fmt: on
):
    tl.static_assert(OP == "add" or OP == "mul")
    tl.static_assert(a_shape1 == b_shape1)
    tl.static_assert(b_shape0 == 1 or b_shape0 == a_shape0)
    tl.static_assert(out_shape0 == a_shape0)
    tl.static_assert(out_shape1 == a_shape1)
    if b_rsos_ptr is not None:
        tl.static_assert(b_shape0 == b_rsos_shape0)
    if b_rmsnorm_w_ptr is not None:
        tl.static_assert(b_shape1 == b_rmsnorm_w_shape0)

    pid0 = tl.program_id(1)
    pid1 = tl.program_id(0)
    offs0 = pid0 * BLOCKSIZE0 + tl.arange(0, BLOCKSIZE0)[:, None]
    offs1 = pid1 * BLOCKSIZE1 + tl.arange(0, BLOCKSIZE1)[None, :]

    a = tl.load(a_ptr + offs0 * a_stride0 + offs1 * a_stride1, mask=(offs0 < a_shape0) & (offs1 < a_shape1))
    b_offs0 = 0 if b_shape0 == 1 else offs0
    b = tl.load(b_ptr + b_offs0 * b_stride0 + offs1 * b_stride1, mask=(b_offs0 < b_shape0) & (offs1 < b_shape1))

    if b_rsos_ptr is not None:
        b_rsos = tl.load(b_rsos_ptr + offs0 * b_rsos_stride0, mask=offs0 < b_rsos_shape0)
        rms = (b_rsos / b_shape1 + rmsnorm_eps).sqrt()
        b /= rms
    if b_rmsnorm_w_ptr is not None:
        b_rmsnorm_w = tl.load(b_rmsnorm_w_ptr + offs1 * b_rmsnorm_w_stride0, mask=(offs1 < b_rmsnorm_w_shape0))
        b *= b_rmsnorm_w

    if OP == "add":
        out = a + b
    else:
        out = a * b

    out = out * output_scale_factor

    tl.store(out_ptr + offs0 * out_stride0 + offs1 * out_stride1, out, mask=(offs0 < out_shape0) & (offs1 < out_shape1))


def _elementwise_2d(
    op: Literal["add", "mul"],
    *,
    a: Tensor,
    b: Tensor,
    out: Tensor,
    output_scale_factor: float,
    rmsnorm_eps: float,
    b_rsos: Tensor | None = None,
    b_rmsnorm_w: Tensor | None = None,
):
    a = a.reshape((-1, a.shape[-1]))
    out = out.reshape((-1, out.shape[-1]))
    b = b.reshape((-1, b.shape[-1]))

    if b_rsos is not None:
        b_rsos = b_rsos.reshape((-1,))
    if b_rmsnorm_w is not None:
        b_rmsnorm_w = b_rmsnorm_w.reshape((-1,))

    grid_fn = lambda META: (
        triton.cdiv(a.shape[1], META["BLOCKSIZE1"]),
        triton.cdiv(a.shape[0], META["BLOCKSIZE0"]),
    )
    return launch[_elementwise_2d_kernel, grid_fn](
        a=a,
        b=b,
        b_rsos=b_rsos,
        b_rmsnorm_w=b_rmsnorm_w,
        out=out,
        OP=op,
        output_scale_factor=output_scale_factor,
        rmsnorm_eps=rmsnorm_eps,
    )


def add(
        a: Tensor,
        b: Tensor,
        out: Tensor,
        output_scale_factor: float,
        rmsnorm_eps: float,
        b_rsos: Tensor | None = None,
        b_rmsnorm_w: Tensor | None = None,
):
    return _elementwise_2d(
        "add",
        a=a,
        b=b,
        out=out,
        b_rsos=b_rsos,
        b_rmsnorm_w=b_rmsnorm_w,
        output_scale_factor=output_scale_factor,
        rmsnorm_eps=rmsnorm_eps,
    )


def mul(
        a: Tensor,
        b: Tensor,
        out: Tensor,
        output_scale_factor: float,
        rmsnorm_eps: float,
        b_rsos: Tensor | None = None,
        b_rmsnorm_w: Tensor | None = None,
):
    return _elementwise_2d(
        "mul",
        a=a,
        b=b,
        out=out,
        b_rsos=b_rsos,
        b_rmsnorm_w=b_rmsnorm_w,
        output_scale_factor=output_scale_factor,
        rmsnorm_eps=rmsnorm_eps,
    )
