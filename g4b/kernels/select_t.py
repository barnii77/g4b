import triton
from triton import language as tl
from g4b.tensor import Tensor
from g4b.kernels.utils import launch, default_bencher, gated_configs


def _cfg(b0: int, b2: int, *, warps: int, stages: int = 3):
    return triton.Config({"BLOCKSIZE0": b0, "BLOCKSIZE2": b2}, num_warps=warps, num_stages=stages)


@triton.autotune(
    configs=gated_configs(
        default=[
            _cfg(1, 128, warps=2),
        ],
        tuned=[
            _cfg(1, 64, warps=1),
            _cfg(1, 256, warps=4),
            _cfg(2, 128, warps=2),
            _cfg(4, 64, warps=2),
        ],
    ),
    key=[
        # fmt: off
        "x_shape0", "x_shape1", "x_shape2",
        "out_shape0", "out_shape1", "out_shape2",
        "positions_shape0",
        "x_stride0", "x_stride1", "x_stride2",
        "out_stride0", "out_stride1", "out_stride2",
        "positions_stride0",
        # fmt: on
    ],
    do_bench=default_bencher,
    cache_results=True,
)
@triton.jit
def _select_t_kernel(
    # fmt: off
    x_ptr, positions_ptr, out_ptr,
    x_shape0: tl.constexpr, x_shape1: tl.constexpr, x_shape2: tl.constexpr,
    positions_shape0: tl.constexpr,
    out_shape0: tl.constexpr, out_shape1: tl.constexpr, out_shape2: tl.constexpr,
    x_stride0: tl.constexpr, x_stride1: tl.constexpr, x_stride2: tl.constexpr,
    positions_stride0: tl.constexpr,
    out_stride0: tl.constexpr, out_stride1: tl.constexpr, out_stride2: tl.constexpr,
    BLOCKSIZE0: tl.constexpr, BLOCKSIZE2: tl.constexpr,
    # fmt: on
):
    tl.static_assert(x_shape0 == out_shape0)
    tl.static_assert(out_shape1 == 1)
    tl.static_assert(x_shape2 == out_shape2)
    tl.static_assert(x_shape0 == positions_shape0)

    pid_d = tl.program_id(0)
    pid_b = tl.program_id(1)

    offs_b = pid_b * BLOCKSIZE0 + tl.arange(0, BLOCKSIZE0)[:, None]
    offs_d = pid_d * BLOCKSIZE2 + tl.arange(0, BLOCKSIZE2)[None, :]
    positions = tl.load(positions_ptr + offs_b * positions_stride0, mask=offs_b < x_shape0, other=0)
    mask = (offs_b < x_shape0) & (offs_d < x_shape2) & (positions >= 0) & (positions < x_shape1)
    x = tl.load(x_ptr + offs_b * x_stride0 + positions * x_stride1 + offs_d * x_stride2, mask=mask)
    tl.store(out_ptr + offs_b * out_stride0 + offs_d * out_stride2, x, mask=mask)


def select_t(x: Tensor, positions: Tensor, out: Tensor):
    grid_fn = lambda META: (
        triton.cdiv(x.shape[2], META["BLOCKSIZE2"]),
        triton.cdiv(x.shape[0], META["BLOCKSIZE0"]),
    )
    return launch[_select_t_kernel, grid_fn](x=x, positions=positions, out=out)
