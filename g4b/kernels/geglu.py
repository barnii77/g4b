import triton
from triton import language as tl


@triton.jit
def tanh_jfn(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def gelu_jfn(x):
    """
    GeLU activation - Gaussian error linear unit.
    GeLU: https://arxiv.org/pdf/1606.08415.pdf
    """
    coef1 = 0.79788456  # sqrt(2 / pi) approx
    coef2 = 0.044715  # scale for x^3 term
    return 0.5 * x * (1 + tanh_jfn(coef1 * (x + coef2 * x * x * x)))


# TODO this combined with the matmul into a full fused geglu is ~3x slower than naive torch and I don't know why.
#  I'll have to investigate.
@triton.jit
def geglu_fusion_matmul_merge_tiles_mixin_jfn(up_tile, gate_tile, _, __, ___, ____, _____):
    return up_tile * gelu_jfn(gate_tile)
