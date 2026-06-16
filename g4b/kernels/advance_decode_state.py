import triton
from triton import language as tl
from g4b.tensor import Tensor
from g4b.kernels.fa2 import PHASE_DECODE_CONSTEXPR
from g4b.kernels.utils import launch, default_bencher, gated_configs


@triton.autotune(
    configs=gated_configs(
        default=[
            triton.Config({"BLOCKSIZE0": 256}),
        ],
        tuned=[
            triton.Config({"BLOCKSIZE0": 128}),
            triton.Config({"BLOCKSIZE0": 512}),
        ],
    ),
    key=[
        # fmt: off
        "input_token_ids_shape0", "input_token_ids_shape1",
        "out_token_ids_shape0", "out_token_ids_shape1",
        "cache_offsets_shape0", "time_dim_sizes_shape0", "user_in_prefill_or_decode_shape0",
        "input_token_ids_stride0", "input_token_ids_stride1",
        "out_token_ids_stride0", "out_token_ids_stride1",
        "cache_offsets_stride0", "time_dim_sizes_stride0", "user_in_prefill_or_decode_stride0",
        # fmt: on
    ],
    do_bench=default_bencher,
    cache_results=True,
)
@triton.jit
def _advance_decode_state_kernel(
    # fmt: off
    input_token_ids_ptr, out_token_ids_ptr, cache_offsets_ptr, time_dim_sizes_ptr, user_in_prefill_or_decode_ptr,
    input_token_ids_shape0: tl.constexpr, input_token_ids_shape1: tl.constexpr,
    out_token_ids_shape0: tl.constexpr, out_token_ids_shape1: tl.constexpr,
    cache_offsets_shape0: tl.constexpr, time_dim_sizes_shape0: tl.constexpr, user_in_prefill_or_decode_shape0: tl.constexpr,
    input_token_ids_stride0: tl.constexpr, input_token_ids_stride1: tl.constexpr,
    out_token_ids_stride0: tl.constexpr, out_token_ids_stride1: tl.constexpr,
    cache_offsets_stride0: tl.constexpr, time_dim_sizes_stride0: tl.constexpr, user_in_prefill_or_decode_stride0: tl.constexpr,
    BLOCKSIZE0: tl.constexpr,
    # fmt: on
):
    tl.static_assert(input_token_ids_shape0 >= 1)
    tl.static_assert(input_token_ids_shape1 == out_token_ids_shape0)
    tl.static_assert(out_token_ids_shape1 >= 1)
    tl.static_assert(input_token_ids_shape1 == cache_offsets_shape0)
    tl.static_assert(input_token_ids_shape1 == time_dim_sizes_shape0)
    tl.static_assert(input_token_ids_shape1 == user_in_prefill_or_decode_shape0)

    offs_b = tl.program_id(0) * BLOCKSIZE0 + tl.arange(0, BLOCKSIZE0)
    mask = offs_b < input_token_ids_shape1
    phase = tl.load(user_in_prefill_or_decode_ptr + offs_b * user_in_prefill_or_decode_stride0, mask=mask)
    mask = mask & (phase == PHASE_DECODE_CONSTEXPR)

    tok = tl.load(out_token_ids_ptr + offs_b * out_token_ids_stride0, mask=mask)
    tl.store(input_token_ids_ptr + offs_b * input_token_ids_stride1, tok, mask=mask)

    cache_offsets = tl.load(cache_offsets_ptr + offs_b * cache_offsets_stride0, mask=mask)
    time_dim_sizes = tl.load(time_dim_sizes_ptr + offs_b * time_dim_sizes_stride0, mask=mask)
    tl.store(cache_offsets_ptr + offs_b * cache_offsets_stride0, cache_offsets + 1, mask=mask)
    tl.store(time_dim_sizes_ptr + offs_b * time_dim_sizes_stride0, time_dim_sizes + 1, mask=mask)


def advance_decode_state(
    input_token_ids: Tensor,
    out_token_ids: Tensor,
    cache_offsets: Tensor,
    time_dim_sizes: Tensor,
    user_in_prefill_or_decode: Tensor,
):
    grid_fn = lambda META: (triton.cdiv(input_token_ids.shape[1], META["BLOCKSIZE0"]),)
    return launch[_advance_decode_state_kernel, grid_fn](
        input_token_ids=input_token_ids,
        out_token_ids=out_token_ids,
        cache_offsets=cache_offsets,
        time_dim_sizes=time_dim_sizes,
        user_in_prefill_or_decode=user_in_prefill_or_decode,
    )
