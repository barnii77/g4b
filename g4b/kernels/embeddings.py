import triton
from triton import language as tl
from g4b.tensor import Tensor
from g4b.kernels.utils import launch, default_bencher
from g4b.kernels.memset import memset_contiguous


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


@triton.autotune(
    # fmt: off
    configs=[
        # ---- decode / tiny token count ----
        # One token row per program, varying embedding-dim tile.
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
        # Useful if token_ids are reasonably contiguous / output write is nicely laid out.
        _cfg(1, 8, 64, warps=4),
        _cfg(1, 8, 128, warps=4),
        # ---- batch batching ----
        # Useful if z batch dimension is small-but-nontrivial and z_stride0/z_stride1 are sane.
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
        "scaling_factor",
        "z_shape0", "z_shape1", "z_shape2",
        "z_rsos_shape0", "z_rsos_shape1", "z_rsos_shape2",
        "embed_shape0", "embed_shape1",
        "token_ids_rb_shape0", "token_ids_rb_shape1",
        "token_ids_rb_offset_shape0",
        "z_stride0", "z_stride1", "z_stride2",
        "z_rsos_stride0", "z_rsos_stride1", "z_rsos_stride2",
        "embed_stride0", "embed_stride1",
        "token_ids_rb_stride0", "token_ids_rb_stride1",
        # fmt: on
    ],
    do_bench=default_bencher,
    cache_results=True,
)
@triton.jit
def _gather_token_embeddings_kernel(
    # fmt: off
    z_ptr, embed_ptr, token_ids_ptr,
    scaling_factor: tl.constexpr,
    z_shape0: tl.constexpr, z_shape1: tl.constexpr, z_shape2: tl.constexpr,
    embed_shape0: tl.constexpr, embed_shape1: tl.constexpr,
    token_ids_shape0: tl.constexpr, token_ids_shape1: tl.constexpr,
    z_stride0: tl.constexpr, z_stride1: tl.constexpr, z_stride2: tl.constexpr,
    embed_stride0: tl.constexpr, embed_stride1: tl.constexpr,
    token_ids_stride0: tl.constexpr, token_ids_stride1: tl.constexpr,
    BLOCKSIZE0: tl.constexpr, BLOCKSIZE1: tl.constexpr, BLOCKSIZE2: tl.constexpr,
    z_rsos_ptr = None,
    z_rsos_shape0: tl.constexpr = 0, z_rsos_shape1: tl.constexpr = 0,
    z_rsos_stride0: tl.constexpr = 0, z_rsos_stride1: tl.constexpr = 0,
    z_rsos: None = None,
    # fmt: on
):
    tl.static_assert(z_shape2 == embed_shape1)  # residual size
    tl.static_assert(token_ids_shape1 == z_shape0)  # batch size
    tl.static_assert(z_rsos_shape0 == 0 or z_rsos_shape0 == z_shape0)
    tl.static_assert(z_rsos_shape1 == 0 or z_rsos_shape1 == z_shape1)

    pid_d = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_b = tl.program_id(2)

    pid_off_t = pid_t * BLOCKSIZE1 + tl.arange(0, BLOCKSIZE1)
    pid_off_b = pid_b * BLOCKSIZE0 + tl.arange(0, BLOCKSIZE0)
    pid_off_d = pid_d * BLOCKSIZE2 + tl.arange(0, BLOCKSIZE2)

    rb_off_t = pid_off_t[None, :]
    rb_off_b = pid_off_b[:, None]
    rb_off = rb_off_t * token_ids_stride0 + rb_off_b * token_ids_stride1

    token_ids = tl.load(
        token_ids_ptr + rb_off,
        mask=(rb_off_t < token_ids_shape0) & (rb_off_b < token_ids_shape1),
        other=embed_shape0,
    )

    embed_off_d = pid_off_d[None, None, :]
    embed_off = token_ids[:, :, None] * embed_stride0 + embed_off_d * embed_stride1
    embeddings = scaling_factor * tl.load(
        embed_ptr + embed_off,
        mask=(token_ids[:, :, None] < embed_shape0) & (embed_off_d < embed_shape1),
    )

    z_off_b = pid_off_b[:, None, None]
    z_off_t = pid_off_t[None, :, None]
    z_off_d = pid_off_d[None, None, :]
    z_off = z_off_b * z_stride0 + z_off_t * z_stride1 + z_off_d * z_stride2
    tl.device_assert(z_off_b >= 0)
    tl.device_assert(z_off_t >= 0)
    tl.device_assert(z_off_d >= 0)

    tl.store(z_ptr + z_off, embeddings, mask=(z_off_b < z_shape0) & (z_off_t < z_shape1) & (z_off_d < z_shape2))
    if z_rsos_ptr is not None:
        tl.atomic_add(
            z_rsos_ptr + pid_off_b[:, None] * z_rsos_stride0 + pid_off_t[None, :] * z_rsos_stride1,
            (embeddings * embeddings).sum(-1),
            mask=(pid_off_b[:, None] < z_rsos_shape0) & (pid_off_t[None, :] < z_rsos_shape1),
        )


def gather_token_embeddings(
    z: Tensor,
    z_rsos: Tensor | None,
    embed: Tensor,
    token_ids: Tensor,
    scaling_factor: int | float,
):
    grid_fn = lambda META: (
        triton.cdiv(z.shape[2], META["BLOCKSIZE2"]),
        triton.cdiv(z.shape[1], META["BLOCKSIZE1"]),
        triton.cdiv(z.shape[0], META["BLOCKSIZE0"]),
    )
    k1 = memset_contiguous(z_rsos, 0) if z_rsos is not None else None
    k2 = launch[_gather_token_embeddings_kernel, grid_fn](
        z=z,
        z_rsos=z_rsos,
        embed=embed,
        token_ids=token_ids,
        scaling_factor=scaling_factor,
    )
    return k1, k2
