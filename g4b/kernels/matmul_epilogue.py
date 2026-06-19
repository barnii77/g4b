import triton
from triton import language as tl
from g4b.kernels import matmul
from g4b.kernels.utils import tanh_jfn


@triton.jit
def gelu_jfn(x):
    """
    GeLU activation - Gaussian error linear unit.
    GeLU: https://arxiv.org/pdf/1606.08415.pdf
    """
    coef1 = 0.79788456  # sqrt(2 / pi) approx
    coef2 = 0.044715  # scale for x^3 term
    return 0.5 * x * (1 + tanh_jfn(coef1 * (x + coef2 * x * x * x)))


@triton.jit
def apply_input_rsos_in_epilogue_mixin_jfn(
    # fmt: off
    tile0, tile1, off0, off1,
    input_rsos_ptr,
    input_rsos_shape0: tl.constexpr, input_rsos_shape1: tl.constexpr,
    input_rsos_stride0: tl.constexpr, input_rsos_stride1: tl.constexpr,
    rmsnorm_dim: tl.constexpr, rmsnorm_eps: tl.constexpr,
    # fmt: on
):
    if input_rsos_ptr is not None:
        # Plain masked load instead of a TMA descriptor: the rsos is a [B, t] scalar-per-row tensor, so the
        # descriptor's contiguous block (tile0.shape[1] == A_BLOCKSIZE1, which can be 1) would be < 16 bytes and
        # violate the TMA minimum. The load is tiny, so TMA buys nothing here.
        o0 = off0 + tl.arange(0, tile0.shape[0])
        o1 = off1 + tl.arange(0, tile0.shape[1])
        rsos_offs = o0[:, None] * input_rsos_stride0 + o1[None, :] * input_rsos_stride1
        rsos_mask = (o0[:, None] < input_rsos_shape0) & (o1[None, :] < input_rsos_shape1)
        rsos = tl.load(input_rsos_ptr + rsos_offs, mask=rsos_mask, other=0.0)
        scale = tl.rsqrt(rsos[:, :, None] / rmsnorm_dim + rmsnorm_eps)
        tile0 *= scale
        if tile1 is not None:
            tile1 *= scale
    return tile0, tile1


@triton.jit
def gelu_tile_times_extra_mixin_jfn(
    # fmt: off
    tile, extra_ptr, off0, off1, off2,
    extra_shape0: tl.constexpr, extra_shape1: tl.constexpr, extra_shape2: tl.constexpr,
    extra_stride0: tl.constexpr, extra_stride1: tl.constexpr, extra_stride2: tl.constexpr,
    USE_MATVEC: tl.constexpr,
    # fmt: on
):
    if USE_MATVEC:
        # No TMA descriptor in matvec mode (desc path unavailable / skinny tile < 16B TMA min): direct
        # masked load of the extra (e.g. up/gate) projection tile.
        _o0 = off0 + tl.arange(0, tile.shape[0])[:, None, None]
        _o1 = off1 + tl.arange(0, tile.shape[1])[None, :, None]
        _o2 = off2 + tl.arange(0, tile.shape[2])[None, None, :]
        extra = tl.load(
            extra_ptr + _o0 * extra_stride0 + _o1 * extra_stride1 + _o2 * extra_stride2,
            mask=(_o0 < extra_shape0) & (_o1 < extra_shape1) & (_o2 < extra_shape2),
            other=0.0,
        )
    else:
        extra_desc = tl.make_tensor_descriptor(
            extra_ptr,
            (extra_shape0, extra_shape1, extra_shape2),
            (extra_stride0, extra_stride1, extra_stride2),
            tile.shape,
        )
        extra = extra_desc.load((off0, off1, off2))
    return gelu_jfn(tile) * extra


@triton.jit
def geglu_fusion_matmul_merge_tiles_mixin_jfn(
    # fmt: off
    up_tile, gate_tile, off0, off1, off2,
    NUM_K_SPLITS: tl.constexpr, C_DTYPE: tl.constexpr,
    input_rsos_ptr,
    input_rsos_shape0: tl.constexpr, input_rsos_shape1: tl.constexpr,
    input_rsos_stride0: tl.constexpr, input_rsos_stride1: tl.constexpr,
    rmsnorm_dim: tl.constexpr, rmsnorm_eps: tl.constexpr,
    # fmt: on
):
    # fmt: off
    up_tile, gate_tile = apply_input_rsos_in_epilogue_mixin_jfn(
        up_tile, gate_tile, off0, off1,
        input_rsos_ptr,
        input_rsos_shape0, input_rsos_shape1,
        input_rsos_stride0, input_rsos_stride1,
        rmsnorm_dim, rmsnorm_eps,
    )
    # fmt: on
    return up_tile * gelu_jfn(gate_tile)


@triton.jit
def ple_gate_storer_jfn(
    # fmt: off
    name: tl.constexpr, desc, ptr, tile, off0, off1, off2, rsos_ptr, extra_ptr,
    rsos_shape0: tl.constexpr, rsos_shape1: tl.constexpr,
    rsos_stride0: tl.constexpr, rsos_stride1: tl.constexpr,
    NUM_K_SPLITS: tl.constexpr, C_DTYPE: tl.constexpr,
    input_rsos_ptr,
    out_stride0: tl.constexpr, out_stride1: tl.constexpr, out_stride2: tl.constexpr, out_shape2: tl.constexpr,
    input_rsos_shape0: tl.constexpr, input_rsos_shape1: tl.constexpr,
    input_rsos_stride0: tl.constexpr, input_rsos_stride1: tl.constexpr,
    rmsnorm_dim: tl.constexpr, rmsnorm_eps: tl.constexpr,
    extra_shape0: tl.constexpr, extra_shape1: tl.constexpr, extra_shape2: tl.constexpr,
    extra_stride0: tl.constexpr, extra_stride1: tl.constexpr, extra_stride2: tl.constexpr,
    rsos_stride2: tl.constexpr = 0, RSOS_HEAD_DIM: tl.constexpr = 0,
    USE_MATVEC: tl.constexpr = False,
    # fmt: on
):
    # fmt: off
    tile, _ = apply_input_rsos_in_epilogue_mixin_jfn(
        tile, None, off0, off1,
        input_rsos_ptr,
        input_rsos_shape0, input_rsos_shape1,
        input_rsos_stride0, input_rsos_stride1,
        rmsnorm_dim, rmsnorm_eps,
    )
    tile = gelu_tile_times_extra_mixin_jfn(
        tile, extra_ptr, off0, off1, off2,
        extra_shape0, extra_shape1, extra_shape2,
        extra_stride0, extra_stride1, extra_stride2,
        USE_MATVEC=USE_MATVEC,
    )
    matmul.matmul_a3d_b2d_partial_rmsnorm_storer_jfn(
        name, desc, ptr, tile, off0, off1, off2, rsos_ptr, extra_ptr,
        rsos_shape0, rsos_shape1,
        rsos_stride0, rsos_stride1,
        NUM_K_SPLITS, C_DTYPE,
        None,
        out_stride0, out_stride1, out_stride2, out_shape2,
        input_rsos_shape0, input_rsos_shape1,
        input_rsos_stride0, input_rsos_stride1,
        rmsnorm_dim, rmsnorm_eps,
        extra_shape0, extra_shape1, extra_shape2,
        extra_stride0, extra_stride1, extra_stride2,
        rsos_stride2, RSOS_HEAD_DIM,
        USE_MATVEC,
    )
    # fmt: on


@triton.jit
def qkv_input_rmsnorm_per_head_rsos_storer_jfn(
    # fmt: off
    name: tl.constexpr, desc, ptr, tile, off0, off1, off2, rsos_ptr, extra_ptr,
    rsos_shape0: tl.constexpr, rsos_shape1: tl.constexpr,
    rsos_stride0: tl.constexpr, rsos_stride1: tl.constexpr,
    NUM_K_SPLITS: tl.constexpr, C_DTYPE: tl.constexpr,
    input_rsos_ptr,
    out_stride0: tl.constexpr, out_stride1: tl.constexpr, out_stride2: tl.constexpr, out_shape2: tl.constexpr,
    input_rsos_shape0: tl.constexpr, input_rsos_shape1: tl.constexpr,
    input_rsos_stride0: tl.constexpr, input_rsos_stride1: tl.constexpr,
    rmsnorm_dim: tl.constexpr, rmsnorm_eps: tl.constexpr,
    extra_shape0: tl.constexpr, extra_shape1: tl.constexpr, extra_shape2: tl.constexpr,
    extra_stride0: tl.constexpr, extra_stride1: tl.constexpr, extra_stride2: tl.constexpr,
    rsos_stride2: tl.constexpr = 0, RSOS_HEAD_DIM: tl.constexpr = 0,
    USE_MATVEC: tl.constexpr = False,
    # fmt: on
):
    # q/k/v projection epilogue: apply the input rmsnorm (using the residual's sum-of-squares) to the
    # projected tile, then compute the PER-HEAD output sum-of-squares (rsos_ptr is the [B, t, n_heads]
    # buffer; RSOS_HEAD_DIM is the head size) and store. This fuses what used to be a separate
    # kernels.rsos.compute_rsos call per q/k/v.
    # fmt: off
    tile, _ = apply_input_rsos_in_epilogue_mixin_jfn(
        tile, None, off0, off1,
        input_rsos_ptr,
        input_rsos_shape0, input_rsos_shape1,
        input_rsos_stride0, input_rsos_stride1,
        rmsnorm_dim, rmsnorm_eps,
    )
    matmul.matmul_a3d_b2d_partial_rmsnorm_storer_jfn(
        name, desc, ptr, tile, off0, off1, off2, rsos_ptr, extra_ptr,
        rsos_shape0, rsos_shape1,
        rsos_stride0, rsos_stride1,
        NUM_K_SPLITS, C_DTYPE,
        None,
        out_stride0, out_stride1, out_stride2, out_shape2,
        input_rsos_shape0, input_rsos_shape1,
        input_rsos_stride0, input_rsos_stride1,
        rmsnorm_dim, rmsnorm_eps,
        extra_shape0, extra_shape1, extra_shape2,
        extra_stride0, extra_stride1, extra_stride2,
        rsos_stride2, RSOS_HEAD_DIM,
        USE_MATVEC,
    )
    # fmt: on
