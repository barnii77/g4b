import triton
from triton import language as tl
from g4b.tensor import Tensor
from g4b.kernels.utils import launch, default_bencher


@triton.autotune(
    configs=[
        triton.Config({"BLOCKSIZE0": 128}),
        triton.Config({"BLOCKSIZE0": 256}),
        triton.Config({"BLOCKSIZE0": 512}),
        triton.Config({"BLOCKSIZE0": 1024}),
    ],
    key=["x_shape0", "x_stride0"],
    do_bench=default_bencher,
    cache_results=True,
)
@triton.jit
def _increment_kernel(
    x_ptr,
    increment_by,
    modulus,
    x_shape0: tl.constexpr,
    x_stride0: tl.constexpr,
    BLOCKSIZE0: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCKSIZE0 + tl.arange(0, BLOCKSIZE0)
    x_ptrs = x_ptr + offs * x_stride0
    mask = offs < x_shape0
    x = tl.load(x_ptrs, mask=mask)
    tl.store(x_ptrs, (x + increment_by) % modulus, mask=mask)


def increment(
    x: Tensor,
    increment_by: int | float,
    modulus: int | float,
):
    grid_fn = lambda META: (triton.cdiv(x.shape[0], META["BLOCKSIZE0"]),)
    return launch[_increment_kernel, grid_fn](x=x, increment_by=increment_by, modulus=modulus)
