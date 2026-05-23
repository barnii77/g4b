import triton
from triton import language as tl
from g4b.tensor import Tensor, uint8
from g4b.kernels.utils import launch


@triton.autotune(
    configs=[
        triton.Config({"BLOCKSIZE": 64}),
        triton.Config({"BLOCKSIZE": 128}),
        triton.Config({"BLOCKSIZE": 256}),
        triton.Config({"BLOCKSIZE": 512}),
        triton.Config({"BLOCKSIZE": 1024}),
        triton.Config({"BLOCKSIZE": 2048}),
        triton.Config({"BLOCKSIZE": 4096}),
    ],
    key=["x_shape0"],
)
@triton.jit
def _memset_contiguous_kernel(x_ptr, x_shape0: tl.constexpr, value: tl.constexpr, BLOCKSIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCKSIZE + tl.arange(0, BLOCKSIZE)
    tl.store(x_ptr + offsets, value, mask=offsets < x_shape0)


def memset_contiguous(x: Tensor, value: int):
    assert x.is_contiguous()
    assert 0 <= value <= 255
    x = x.reshape((-1,)).view(uint8)
    grid_fn = lambda META: (triton.cdiv(x.shape[0], META["BLOCKSIZE"]),)
    return launch[_memset_contiguous_kernel, grid_fn](x=x, value=value)
