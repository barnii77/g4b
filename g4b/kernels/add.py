import triton
from triton import language as tl
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
def _add_2d_kernel(
    # fmt: off
    a_ptr, b_ptr, b_rsos_ptr, b_rmsnorm_w_ptr, out_ptr,
    output_scale_factor: tl.constexpr,
    rmsnorm_eps: tl.constexpr,
    a_shape0: tl.constexpr, a_shape1: tl.constexpr,
    b_shape0: tl.constexpr, b_shape1: tl.constexpr,
    b_rsos_shape0: tl.constexpr,
    b_rmsnorm_w_shape0: tl.constexpr,
    out_shape0: tl.constexpr, out_shape1: tl.constexpr,
    a_stride0: tl.constexpr, a_stride1: tl.constexpr,
    b_stride0: tl.constexpr, b_stride1: tl.constexpr,
    b_rsos_stride0: tl.constexpr,
    b_rmsnorm_w_stride0: tl.constexpr,
    out_stride0: tl.constexpr, out_stride1: tl.constexpr,
    BLOCKSIZE0: tl.constexpr, BLOCKSIZE1: tl.constexpr,
    # fmt: on
):
    tl.static_assert(a_shape0 == b_shape0)
    tl.static_assert(a_shape1 == b_shape1)
    tl.static_assert(b_shape0 == b_rsos_shape0)
    tl.static_assert(b_shape1 == b_rmsnorm_w_shape0)
    tl.static_assert(out_shape0 == a_shape0)
    tl.static_assert(out_shape1 == a_shape1)

    pid0 = tl.program_id(1)
    pid1 = tl.program_id(0)
    offs0 = pid0 * BLOCKSIZE0 + tl.arange(0, BLOCKSIZE0)[:, None]
    offs1 = pid1 * BLOCKSIZE1 + tl.arange(0, BLOCKSIZE1)[None, :]

    a = tl.load(a_ptr + offs0 * a_stride0 + offs1 * a_stride1, mask=(offs0 < a_shape0) & (offs1 < a_shape1))
    b = tl.load(b_ptr + offs0 * b_stride0 + offs1 * b_stride1, mask=(offs0 < b_shape0) & (offs1 < b_shape1))
    b_rsos = tl.load(b_rsos_ptr + offs0 * b_rsos_stride0, mask=offs0 < b_rsos_shape0)
    b_rmsnorm_w = tl.load(b_rmsnorm_w_ptr + offs1 * b_rmsnorm_w_stride0, mask=(offs1 < b_rmsnorm_w_shape0))

    rms = (b_rsos / b_shape1 + rmsnorm_eps).sqrt()

    out = (a + b / rms * b_rmsnorm_w) * output_scale_factor

    tl.store(out_ptr + offs0 * out_stride0 + offs1 * out_stride1, out, mask=(offs0 < out_shape0) & (offs1 < out_shape1))


def add(
    a: Tensor,
    b: Tensor,
    b_rsos: Tensor,
    b_rmsnorm_w: Tensor,
    out: Tensor,
    output_scale_factor: float,
    rmsnorm_eps: float,
):
    a = a.reshape((-1, a.shape[-1]))
    b = b.reshape((-1, b.shape[-1]))
    out = out.reshape((-1, out.shape[-1]))
    b_rsos = b_rsos.reshape((-1,))
    b_rmsnorm_w = b_rmsnorm_w.reshape((-1,))

    assert a.shape == b.shape == out.shape
    assert b.shape[:-1] == b_rsos.shape
    assert b.shape[-1:] == b_rmsnorm_w.shape

    grid_fn = lambda META: (
        triton.cdiv(a.shape[1], META["BLOCKSIZE1"]),
        triton.cdiv(a.shape[0], META["BLOCKSIZE0"]),
    )
    return launch[_add_2d_kernel, grid_fn](
        a=a,
        b=b,
        b_rsos=b_rsos,
        b_rmsnorm_w=b_rmsnorm_w,
        out=out,
        output_scale_factor=output_scale_factor,
        rmsnorm_eps=rmsnorm_eps,
    )
