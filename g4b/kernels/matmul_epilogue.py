import triton
from triton import language as tl
from g4b.kernels.matmul import matmul_a3d_b2d_partial_rmsnorm_storer_jfn
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
        rsos_desc = tl.make_tensor_descriptor(
            input_rsos_ptr,
            (input_rsos_shape0, input_rsos_shape1),
            (input_rsos_stride0, input_rsos_stride1),
            (tile0.shape[0], tile0.shape[1]),
        )
        rsos = rsos_desc.load((off0, off1))
        scale = tl.rsqrt(rsos[:, :, None] / rmsnorm_dim + rmsnorm_eps)
        tile0 *= scale
        if tile1 is not None:
            tile1 *= scale
    return tile0, tile1


@triton.jit
def gelu_tile_times_extra_mixin_jfn(
    # fmt: off
    tile, c_extra_2_ptr, off0, off1, off2,
    c_extra_2_shape0: tl.constexpr, c_extra_2_shape1: tl.constexpr, c_extra_2_shape2: tl.constexpr,
    c_extra_2_stride0: tl.constexpr, c_extra_2_stride1: tl.constexpr, c_extra_2_stride2: tl.constexpr,
    # fmt: on
):
    extra_desc = tl.make_tensor_descriptor(
        c_extra_2_ptr,
        (c_extra_2_shape0, c_extra_2_shape1, c_extra_2_shape2),
        (c_extra_2_stride0, c_extra_2_stride1, c_extra_2_stride2),
        tile.shape,
    )
    return gelu_jfn(tile) * extra_desc.load((off0, off1, off2))


@triton.jit
def geglu_fusion_matmul_merge_tiles_mixin_jfn(
    # fmt: off
    up_tile, gate_tile, off0, off1, off2,
    NUM_K_SPLITS: tl.constexpr, C_DTYPE: tl.constexpr,
    input_rsos_ptr,
    input_rsos_shape0: tl.constexpr, input_rsos_shape1: tl.constexpr,
    input_rsos_stride0: tl.constexpr, input_rsos_stride1: tl.constexpr,
    rmsnorm_dim: tl.constexpr, rmsnorm_eps: tl.constexpr,
    c_extra_2_ptr,
    c_extra_2_shape0: tl.constexpr, c_extra_2_shape1: tl.constexpr, c_extra_2_shape2: tl.constexpr,
    c_extra_2_stride0: tl.constexpr, c_extra_2_stride1: tl.constexpr, c_extra_2_stride2: tl.constexpr,
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
    name: tl.constexpr, desc, tile, off0, off1, off2, rsos_ptr, extra_ptr,
    rsos_shape0: tl.constexpr, rsos_shape1: tl.constexpr,
    rsos_stride0: tl.constexpr, rsos_stride1: tl.constexpr,
    NUM_K_SPLITS: tl.constexpr, C_DTYPE: tl.constexpr,
    input_rsos_ptr,
    input_rsos_shape0: tl.constexpr, input_rsos_shape1: tl.constexpr,
    input_rsos_stride0: tl.constexpr, input_rsos_stride1: tl.constexpr,
    rmsnorm_dim: tl.constexpr, rmsnorm_eps: tl.constexpr,
    c_extra_2_ptr,
    c_extra_2_shape0: tl.constexpr, c_extra_2_shape1: tl.constexpr, c_extra_2_shape2: tl.constexpr,
    c_extra_2_stride0: tl.constexpr, c_extra_2_stride1: tl.constexpr, c_extra_2_stride2: tl.constexpr,
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
        tile, c_extra_2_ptr, off0, off1, off2,
        c_extra_2_shape0, c_extra_2_shape1, c_extra_2_shape2,
        c_extra_2_stride0, c_extra_2_stride1, c_extra_2_stride2,
    )
    matmul_a3d_b2d_partial_rmsnorm_storer_jfn(
        name, desc, tile, off0, off1, off2, rsos_ptr, extra_ptr,
        rsos_shape0, rsos_shape1,
        rsos_stride0, rsos_stride1,
        NUM_K_SPLITS, C_DTYPE,
        input_rsos_ptr,
        input_rsos_shape0, input_rsos_shape1,
        input_rsos_stride0, input_rsos_stride1,
        rmsnorm_dim, rmsnorm_eps,
        c_extra_2_ptr,
        c_extra_2_shape0, c_extra_2_shape1, c_extra_2_shape2,
        c_extra_2_stride0, c_extra_2_stride1, c_extra_2_stride2,
    )
    # fmt: on
