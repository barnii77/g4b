import triton
from triton import language as tl
from g4b.tensor import Tensor
from g4b.kernels.utils import launch

# TODO fix this kernel


def _gather_cfg(
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


@triton.autotune(
    # fmt: off
    configs=[
        # ---- decode / tiny token count ----
        # One token row per program, varying embedding-dim tile.
        _gather_cfg(1, 1, 64, warps=1),
        _gather_cfg(1, 1, 128, warps=2),
        _gather_cfg(1, 1, 256, warps=4),
        # ---- small prefill / a few positions per program ----
        _gather_cfg(1, 2, 64, warps=1),
        _gather_cfg(1, 2, 128, warps=2),
        _gather_cfg(1, 2, 256, warps=4),
        _gather_cfg(1, 4, 64, warps=2),
        _gather_cfg(1, 4, 128, warps=4),
        _gather_cfg(1, 4, 256, warps=4),
        # ---- more position batching ----
        # Useful if token_ids are reasonably contiguous / output write is nicely laid out.
        _gather_cfg(1, 8, 64, warps=4),
        _gather_cfg(1, 8, 128, warps=4),
        # ---- batch batching ----
        # Useful if z batch dimension is small-but-nontrivial and z_stride0/z_stride1 are sane.
        _gather_cfg(2, 1, 64, warps=1),
        _gather_cfg(2, 1, 128, warps=2),
        _gather_cfg(2, 2, 64, warps=2),
        _gather_cfg(2, 2, 128, warps=4),
        _gather_cfg(4, 1, 64, warps=2),
        _gather_cfg(4, 1, 128, warps=4),
    ],
    # fmt: on
    key=[
        # fmt: off
        "z_shape0", "z_shape1", "z_shape2",
        "embed_shape0", "embed_shape1",
        "token_ids_rb_shape0", "token_ids_rb_shape1",
        "token_ids_rb_offset_shape0",
        "z_stride0", "z_stride1", "z_stride2",
        "embed_stride0", "embed_stride1",
        "token_ids_rb_stride0", "token_ids_rb_stride1",
        # fmt: on
    ],
)
@triton.jit
def _gather_token_embeddings_kernel(
    # fmt: off
    z_ptr, embed_ptr, token_ids_rb_ptr, token_ids_rb_offset_ptr,
    z_shape0: tl.constexpr, z_shape1: tl.constexpr, z_shape2: tl.constexpr,
    embed_shape0: tl.constexpr, embed_shape1: tl.constexpr,
    token_ids_rb_shape0: tl.constexpr, token_ids_rb_shape1: tl.constexpr,
    token_ids_rb_offset_shape0: tl.constexpr,
    z_stride0: tl.constexpr, z_stride1: tl.constexpr, z_stride2: tl.constexpr,
    embed_stride0: tl.constexpr, embed_stride1: tl.constexpr,
    token_ids_rb_stride0: tl.constexpr, token_ids_rb_stride1: tl.constexpr,
    BLOCKSIZE0: tl.constexpr, BLOCKSIZE1: tl.constexpr, BLOCKSIZE2: tl.constexpr,
    # fmt: on
):
    tl.static_assert(z_shape2 == embed_shape1)  # residual size
    tl.static_assert(token_ids_rb_offset_shape0 == 1)  # scalar
    tl.static_assert(token_ids_rb_shape1 == z_shape0)  # batch size

    next_token_id_rb_time_offset = tl.load(token_ids_rb_offset_ptr)

    pid_d = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_t = tl.program_id(2)

    pid_off_t = pid_t * BLOCKSIZE1 + tl.arange(0, BLOCKSIZE1)
    pid_off_b = pid_b * BLOCKSIZE0 + tl.arange(0, BLOCKSIZE0)
    pid_off_d = pid_d * BLOCKSIZE2 + tl.arange(0, BLOCKSIZE2)

    # TODO does this layout give me gmem coalesced loads?
    rb_off_t = next_token_id_rb_time_offset + pid_off_t[None, :]
    rb_off_b = pid_off_b[:, None]
    rb_off = rb_off_t * token_ids_rb_stride0 + rb_off_b * token_ids_rb_stride1

    token_ids = tl.load(
        token_ids_rb_ptr + rb_off,
        mask=(rb_off_t < token_ids_rb_shape0) & (rb_off_b < token_ids_rb_shape1),
        other=embed_shape0,
    )

    embed_off_d = pid_off_d[None, None, :]
    embed_off = token_ids[:, :, None] * embed_stride0 + embed_off_d * embed_stride1
    embeddings = tl.load(
        embed_ptr + embed_off, mask=(token_ids[:, :, None] < embed_shape0) & (embed_off_d < embed_shape1)
    )

    z_off_b = pid_off_b[:, None, None]
    z_off_t = pid_off_t[None, :, None]
    z_off_d = pid_off_d[None, None, :]
    z_off = z_off_b * z_stride0 + z_off_t * z_stride1 + z_off_d * z_stride2
    tl.device_assert(z_off_b >= 0)
    tl.device_assert(z_off_t >= 0)
    tl.device_assert(z_off_d >= 0)

    tl.store(z_ptr + z_off, embeddings, mask=(z_off_b < z_shape0) & (z_off_t < z_shape1) & (z_off_d < z_shape2))


def gather_token_embeddings(z: Tensor, embed: Tensor, token_ids_rb: Tensor, token_ids_rb_offset: Tensor):
    grid_fn = lambda META: (
        triton.cdiv(z.shape[2], META["BLOCKSIZE2"]),
        triton.cdiv(z.shape[1], META["BLOCKSIZE1"]),
        triton.cdiv(z.shape[0], META["BLOCKSIZE0"]),
    )
    return launch[_gather_token_embeddings_kernel, grid_fn](
        z=z, embed=embed, token_ids_rb=token_ids_rb, token_ids_rb_offset=token_ids_rb_offset
    )


# TODO per layer embeddings per-layer application mixin jfn

