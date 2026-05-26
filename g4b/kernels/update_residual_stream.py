import triton
from triton import language as tl
from g4b.tensor import Tensor
from g4b.kernels.utils import launch


def _cfg(
    b0: int,
    b1: int,
    b2: int,
    *,
    warps: int,
    stages: int = 3,
):
    return triton.Config(
        {
            "BLOCKSIZE0": b0,
            "BLOCKSIZE1": b1,
            "BLOCKSIZE2": b2,
        },
        num_warps=warps,
        num_stages=stages,
    )


# TODO fuse rmsnorm w into this as well?
@triton.autotune(
    # fmt: off
    configs=[
        # ---- decode / tiny token count ----
        _cfg(1, 1, 64, warps=1),
        _cfg(1, 1, 128, warps=2),
        _cfg(1, 1, 256, warps=4),
        # ---- small prefill / a few positions per program ----
        _cfg(1, 2, 64, warps=1),
        _cfg(1, 2, 128, warps=2),
        _cfg(1, 2, 256, warps=4),
        _cfg(1, 4, 64, warps=2),
        _cfg(1, 4, 128, warps=4),
        _cfg(1, 4, 256, warps=4),
        # ---- more position batching ----
        _cfg(1, 8, 64, warps=4),
        _cfg(1, 8, 128, warps=4),
        # ---- batch batching ----
        _cfg(2, 1, 64, warps=1),
        _cfg(2, 1, 128, warps=2),
        _cfg(2, 2, 64, warps=2),
        _cfg(2, 2, 128, warps=4),
        _cfg(4, 1, 64, warps=2),
        _cfg(4, 1, 128, warps=4),
    ],
    # fmt: on
    key=[
        # fmt: off
        "residual_shape0", "residual_shape1", "residual_shape2",
        "act_buf_shape0", "act_buf_shape1", "act_buf_shape2",
        "act_rsos_shape0", "act_rsos_shape1",
        "residual_stride0", "residual_stride1", "residual_stride2",
        "act_buf_stride0", "act_buf_stride1", "act_buf_stride2",
        "act_rsos_stride0", "act_rsos_stride1",
        # fmt: on
    ],
)
@triton.jit
def _update_residual_stream_and_rsos_kernel(
    # fmt: off
    residual_ptr, act_buf_ptr, act_rsos_ptr,
    residual_shape0: tl.constexpr, residual_shape1: tl.constexpr, residual_shape2: tl.constexpr,
    act_buf_shape0: tl.constexpr, act_buf_shape1: tl.constexpr, act_buf_shape2: tl.constexpr,
    act_rsos_shape0: tl.constexpr, act_rsos_shape1: tl.constexpr,
    residual_stride0: tl.constexpr, residual_stride1: tl.constexpr, residual_stride2: tl.constexpr,
    act_buf_stride0: tl.constexpr, act_buf_stride1: tl.constexpr, act_buf_stride2: tl.constexpr,
    act_rsos_stride0: tl.constexpr, act_rsos_stride1: tl.constexpr,
    BLOCKSIZE0: tl.constexpr, BLOCKSIZE1: tl.constexpr, BLOCKSIZE2: tl.constexpr,
    eps: tl.constexpr,
    # fmt: on
):
    tl.static_assert(residual_shape0 == act_buf_shape0)
    tl.static_assert(residual_shape0 == act_rsos_shape0)
    tl.static_assert(residual_shape1 == act_buf_shape1)
    tl.static_assert(residual_shape1 == act_rsos_shape1)
    tl.static_assert(residual_shape2 == act_buf_shape2)

    pid_b = tl.program_id(2)
    pid_t = tl.program_id(1)
    pid_d = tl.program_id(0)

    off_b = pid_b * BLOCKSIZE0 + tl.arange(0, BLOCKSIZE0)[:, None, None]
    off_t = pid_t * BLOCKSIZE1 + tl.arange(0, BLOCKSIZE1)[None, :, None]
    off_d = pid_d * BLOCKSIZE2 + tl.arange(0, BLOCKSIZE2)[None, None, :]

    residual_off = off_b * residual_stride0 + off_t * residual_stride1 + off_d * residual_stride2
    residual = tl.load(
        residual_ptr + residual_off,
        mask=(off_b < residual_shape0) & (off_t < residual_shape1) & (off_d < residual_shape2),
    )

    act_buf_off = off_b * act_buf_stride0 + off_t * act_buf_stride1 + off_d * act_buf_stride2
    act_buf = tl.load(
        act_buf_ptr + act_buf_off, mask=(off_b < act_buf_shape0) & (off_t < act_buf_shape1) & (off_d < act_buf_shape2)
    )

    rsos_off = off_b * act_rsos_stride0 + off_t * act_rsos_stride1
    act_rsos = tl.load(act_rsos_ptr + rsos_off, mask=(off_b < act_rsos_shape0) & (off_t < act_rsos_shape1))

    inv_rms = tl.rsqrt(act_rsos / residual_shape2 + eps)

    residual += act_buf * inv_rms

    tl.store(
        residual_ptr + residual_off,
        residual,
        mask=(off_b < residual_shape0) & (off_t < residual_shape1) & (off_d < residual_shape2),
    )
    tl.store(
        act_rsos_ptr + rsos_off.reshape((BLOCKSIZE0, BLOCKSIZE1)),
        (residual * residual).sum(-1),
        mask=((off_b < act_rsos_shape0) & (off_t < act_rsos_shape1)).reshape((BLOCKSIZE0, BLOCKSIZE1)),
    )


def update_residual_stream_and_rsos(residual: Tensor, act_buf: Tensor, act_rsos: Tensor, eps: float):
    grid_fn = lambda META: (
        triton.cdiv(residual.shape[2], META["BLOCKSIZE2"]),
        triton.cdiv(residual.shape[1], META["BLOCKSIZE1"]),
        triton.cdiv(residual.shape[0], META["BLOCKSIZE0"]),
    )
    return launch[_update_residual_stream_and_rsos_kernel, grid_fn](
        residual=residual, act_buf=act_buf, act_rsos=act_rsos, eps=float(eps)
    )
