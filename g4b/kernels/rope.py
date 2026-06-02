import triton
from triton import language as tl
from g4b.tensor import Tensor
from g4b.kernels.utils import launch


@triton.jit
def _populate_rope_frequencies_kernel(
    out_ptr,
    freq_base: float,
    out_shape0: tl.constexpr,
    out_stride0: tl.constexpr,
    freq_scalars_stride0: tl.constexpr,
    freq_scalars_ptr = None,
    BLOCKSIZE: tl.constexpr = 128,
):
    pid = tl.program_id(0)
    offs = pid * BLOCKSIZE + tl.arange(0, BLOCKSIZE)

    out = 1.0 / (freq_base ** (tl.arange(pid * BLOCKSIZE, (pid + 1) * BLOCKSIZE, 2) / out_shape0))

    if freq_scalars_ptr is not None:
        freq_scalars = tl.load(freq_scalars_ptr + offs * freq_scalars_stride0, mask=offs < out_shape0)
        out *= freq_scalars

    tl.store(out_ptr + offs * out_stride0, out, mask=offs < out_shape0)


def populate_rope_frequencies(out: Tensor, freq_scalars: Tensor | None, freq_base: float):
    assert freq_scalars is None or list(out.shape) == list(freq_scalars.shape)
    grid_fn = lambda META: (triton.cdiv(out.shape[0], META["BLOCKSIZE"]),)
    launch[_populate_rope_frequencies_kernel, grid_fn](out=out, freq_scalars=freq_scalars, freq_base=freq_base)
