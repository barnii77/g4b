import triton
from triton import language as tl
from g4b.tensor import Tensor
from g4b.kernels.utils import launch, default_bencher, gated_configs
from g4b.kernels.memset import memset_contiguous
from g4b.kernels.matmul import matmul_a3d_b2d_b_loader_jfn


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
    configs=gated_configs(
        # The embedding table is K-quantized, so each program dequantizes ONE token row over a tile of the
        # embedding dim (BLOCKSIZE2 must be a superblock-aligned >=32 column block). BLOCKSIZE0/1 are forced
        # to 1 because the dequant loader handles a single row at a time.
        default=[
            _cfg(1, 1, 256, warps=4),
        ],
        tuned=[
            _cfg(1, 1, 256, warps=8),
        ],
    ),
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
    EMBED_DTYPE: tl.constexpr = None,
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
    tl.static_assert(BLOCKSIZE0 == 1 and BLOCKSIZE1 == 1)  # dequant loader handles one token row per program

    pid_d = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_b = tl.program_id(2)

    first_col = pid_d * BLOCKSIZE2
    pid_off_b = pid_b + tl.arange(0, 1)
    pid_off_t = pid_t + tl.arange(0, 1)
    pid_off_d = first_col + tl.arange(0, BLOCKSIZE2)

    # one (b, t) -> one row (token id) per program
    token_id = tl.load(token_ids_ptr + pid_t * token_ids_stride0 + pid_b * token_ids_stride1)

    # dequantize embed[token_id, first_col : first_col + BLOCKSIZE2] -> tile shape (1, 1, BLOCKSIZE2)
    embed_tile = matmul_a3d_b2d_b_loader_jfn(
        "embed", None, 0, token_id, first_col, embed_ptr,
        1, embed_shape0, embed_shape1,
        0, embed_stride0, embed_stride1,
        1, 1, BLOCKSIZE2,
        EMBED_DTYPE,
    )
    embeddings = scaling_factor * embed_tile.reshape((1, 1, BLOCKSIZE2)).to(z_ptr.dtype.element_ty)

    z_off_d = pid_off_d[None, None, :]
    z_off = pid_off_b[:, None, None] * z_stride0 + pid_off_t[None, :, None] * z_stride1 + z_off_d * z_stride2
    tl.store(z_ptr + z_off, embeddings, mask=z_off_d < z_shape2)
    if z_rsos_ptr is not None:
        tl.atomic_add(
            z_rsos_ptr + pid_off_b[:, None] * z_rsos_stride0 + pid_off_t[None, :] * z_rsos_stride1,
            (embeddings.to(tl.float32) * embeddings.to(tl.float32)).sum(-1),
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
        EMBED_DTYPE=embed.dtype.name,
    )
    return k1, k2
