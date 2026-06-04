import math
import triton
from triton import language as tl
from g4b.tensor import Tensor
from g4b.kernels.utils import launch


# TODO autotune
@triton.jit
def _rope_kernel(
    # fmt: off
    q_ptr, k_ptr, rope_freqs_ptr, time_dim_offsets_ptr,
    q_shape0: tl.constexpr, q_shape1: tl.constexpr, q_shape2: tl.constexpr, q_shape3: tl.constexpr,
    k_shape0: tl.constexpr, k_shape1: tl.constexpr, k_shape2: tl.constexpr, k_shape3: tl.constexpr,
    rope_freqs_shape0: tl.constexpr, rope_freqs_shape1: tl.constexpr, rope_freqs_shape2: tl.constexpr, rope_freqs_shape3: tl.constexpr,
    time_dim_offsets_shape0: tl.constexpr, time_dim_offsets_shape1: tl.constexpr, time_dim_offsets_shape2: tl.constexpr, time_dim_offsets_shape3: tl.constexpr,
    q_stride0: tl.constexpr, q_stride1: tl.constexpr, q_stride2: tl.constexpr, q_stride3: tl.constexpr,
    k_stride0: tl.constexpr, k_stride1: tl.constexpr, k_stride2: tl.constexpr, k_stride3: tl.constexpr,
    rope_freqs_stride0: tl.constexpr, rope_freqs_stride1: tl.constexpr, rope_freqs_stride2: tl.constexpr, rope_freqs_stride3: tl.constexpr,
    time_dim_offsets_stride0: tl.constexpr, time_dim_offsets_stride1: tl.constexpr, time_dim_offsets_stride2: tl.constexpr, time_dim_offsets_stride3: tl.constexpr,
    BLOCKSIZE0: tl.constexpr, BLOCKSIZE1: tl.constexpr, BLOCKSIZE2: tl.constexpr, BLOCKSIZE3: tl.constexpr,
    # fmt: on
): ...  # TODO


# Only used once during model loading
@triton.jit
def _populate_rope_frequencies_kernel(
    out_ptr,
    freq_base: tl.constexpr,
    out_shape0: tl.constexpr,
    out_stride0: tl.constexpr,
    freq_scalars_stride0: tl.constexpr = None,
    freq_scalars_ptr=None,
    BLOCKSIZE: tl.constexpr = 128,
    freq_scalars: None = None,  # sink for when freq_scalars=None arg to launch[...](...)
):
    pid = tl.program_id(0)
    offs = pid * BLOCKSIZE + tl.arange(0, BLOCKSIZE)

    idx = pid * BLOCKSIZE + tl.arange(0, BLOCKSIZE)
    powers = idx.to(tl.float32) / out_shape0
    out = 1.0 / tl.exp2(math.log2(freq_base) * powers)  # 1 / freq_base ** powers

    if freq_scalars_ptr is not None:
        freq_scalars = tl.load(freq_scalars_ptr + offs * freq_scalars_stride0, mask=offs < out_shape0)
        out *= freq_scalars

    tl.store(out_ptr + offs * out_stride0, out, mask=offs < out_shape0)


def populate_rope_frequencies(out: Tensor, freq_scalars: Tensor | None, freq_base: float):
    assert freq_scalars is None or list(out.shape) == list(freq_scalars.shape)
    grid_fn = lambda META: (triton.cdiv(out.shape[0], META["BLOCKSIZE"]),)
    launch[_populate_rope_frequencies_kernel, grid_fn](out=out, freq_scalars=freq_scalars, freq_base=freq_base)
