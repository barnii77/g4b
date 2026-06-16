import triton
from triton import language as tl
from g4b.tensor import Tensor, uint8
from g4b.kernels.utils import launch, default_bencher


@triton.autotune(
    configs=[triton.Config({"BLOCKSIZE": 256})],
    key=["x_shape0"],
    do_bench=default_bencher,
    cache_results=True,
)
@triton.jit
def _memset_contiguous_kernel(x_ptr, x_shape0: tl.constexpr, value: tl.constexpr, BLOCKSIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCKSIZE + tl.arange(0, BLOCKSIZE)
    tl.store(x_ptr + offsets, value, mask=offsets < x_shape0)


def memset_contiguous_by_ptr(x_ptr, x_size, value: int):
    assert 0 <= value <= 255
    grid_fn = lambda META: (triton.cdiv(x_size, META["BLOCKSIZE"]),)
    return _memset_contiguous_kernel[grid_fn](x_ptr, x_size, value)


def memset_contiguous(x: Tensor, value: int):
    assert x.is_contiguous()
    assert 0 <= value <= 255

    x = x.reshape((-1,))
    if isinstance(x, Tensor):
        x = x.view(uint8)
    else:
        import torch
        assert isinstance(x, torch.Tensor)
        x = x.view(torch.uint8)

    grid_fn = lambda META: (triton.cdiv(x.shape[0], META["BLOCKSIZE"]),)
    return launch[_memset_contiguous_kernel, grid_fn](x=x, value=value)
