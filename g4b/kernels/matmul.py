import math
import triton
from triton import language as tl
from g4b import tensor
from g4b.tensor import Tensor, DType
from g4b.kernels.utils import launch
from g4b.kernels.memset import memset_contiguous_by_ptr
from g4b.utils import contiguous_strides_for_shape

# TODO it appears that torch's matmul (without sum of squares fusion) using fp16 + fp32 accum is ~10% faster than mine.
#  With fp16 + fp16 accum, mine is significantly faster though. Maybe torch with set_float32_matmul_precision("high") is
#  not quite doing the same computation as my kernel? The docs do say something about it using either tf32 or 2x bf16...
# TODO INT8 on SM89 gives you ~4.5x the tensor core throughput of FP16, so I absolutely need to add support for INT8,
#  like llama.cpp does it:
#  Q4_K packed nibbles -> widened integer codes arranged as int8 MMA operands
#  F32 activations -> temporary Q8_1 activation blocks
#  int8 x int8 MMA -> int32 dot-products
#  then apply Q4_K scales/mins and Q8_1 scales/sums in FP32
#  ... sadly, I think for now I don't have the time to get this right, let's hope mem bandwidth bottlenecks enough for
#  bf16 mma to do the trick during decode.


def _make_config_pre_hook(split_k: int):
    def pre_hook(args):
        shape = [args["c_shape0"], args["c_shape1"], args["c_shape2"]]
        strides = [args["c_stride0"], args["c_stride1"], args["c_stride2"]]
        if "c_rmsnorm_sum_of_squares_stride0" in args:
            shape_norm = shape[:-1]
            strides_norm = [args["c_rmsnorm_sum_of_squares_stride0"], args["c_rmsnorm_sum_of_squares_stride1"]]
            assert contiguous_strides_for_shape(shape_norm) == strides_norm, "sum of squares buffer must be contiguous"
            memset_contiguous_by_ptr(args["c_rmsnorm_sum_of_squares_ptr"], math.prod(shape[:-1]), 0)
        if split_k > 1:
            # TODO: not inherently, but I don't want to write more memcpy kernels
            assert contiguous_strides_for_shape(shape) == strides, "split k requires contiguous output buffer"
            if not args["KEEP_C"]:
                memset_contiguous_by_ptr(args["c_ptr"], math.prod(shape), 0)

    return pre_hook


def _cfg(
    a0: int,
    a1: int,
    k: int,
    n: int,
    group1: int,
    split_k: int = 1,
    *,
    warps: int = 4,
    stages: int = 3,
):
    return triton.Config(
        {
            "A_BLOCKSIZE0": a0,
            "A_BLOCKSIZE1": a1,
            "A_BLOCKSIZE2": k,
            "B_BLOCKSIZE1": n,
            "GROUPSIZE1": group1,
            "NUM_K_SPLITS": split_k,
        },
        num_warps=warps,
        num_stages=stages,
        pre_hook=_make_config_pre_hook(split_k),
    )


def _matmul_3d_autotune_configs():
    # AI slop configs
    # TODO better configs, enable all configs, proper warmup mechanism (see todo.md)
    return [
        _cfg(1, 128, 32, 128, 8, warps=4, stages=3),
        ] + 0*[
        # ---- small / skinny-N / decode-ish ----
        # Effective MxN: 16x16, 16x32, 32x16, 32x32
        _cfg(1, 16, 64, 16, 1, warps=4, stages=3),
        _cfg(1, 16, 64, 32, 8, warps=4, stages=3),
        _cfg(2, 16, 64, 16, 8, warps=4, stages=3),
        _cfg(2, 16, 64, 32, 8, warps=4, stages=3),
        # ---- normal balanced tiles ----
        # Effective MxN: 32x64, 64x32, 64x64
        _cfg(2, 16, 64, 64, 8, warps=4, stages=3),
        _cfg(4, 16, 64, 32, 8, warps=4, stages=3),
        _cfg(4, 16, 64, 64, 8, warps=4, stages=3),
        # Larger K tile: usually good when K is big and register pressure is fine.
        _cfg(2, 16, 128, 64, 8, warps=4, stages=3),
        _cfg(4, 16, 128, 32, 8, warps=4, stages=3),
        _cfg(4, 16, 128, 64, 8, warps=4, stages=3),
        # ---- bigger output tiles ----
        # Effective MxN: 64x128, 128x64, 128x128
        _cfg(4, 16, 64, 128, 8, warps=4, stages=4),
        _cfg(8, 16, 64, 64, 8, warps=4, stages=4),
        _cfg(8, 16, 64, 128, 8, warps=8, stages=4),
        _cfg(4, 16, 128, 128, 8, warps=4, stages=4),
        _cfg(8, 16, 128, 64, 8, warps=4, stages=4),
        _cfg(8, 16, 128, 128, 8, warps=8, stages=4),
        # ---- split-K variants ----
        _cfg(2, 16, 64, 64, 8, split_k=2, warps=4, stages=3),
        _cfg(4, 16, 64, 64, 8, split_k=2, warps=4, stages=3),
        _cfg(4, 16, 128, 64, 8, split_k=2, warps=4, stages=4),
        _cfg(2, 16, 64, 64, 8, split_k=4, warps=4, stages=3),
        _cfg(4, 16, 64, 64, 8, split_k=4, warps=4, stages=3),
    ]


@triton.autotune(
    configs=_matmul_3d_autotune_configs(),
    key=[
        # fmt: off
        # output / input problem shape
        "c_shape0", "c_shape1", "c_shape2", "a_shape0", "a_shape1", "a_shape2", "b_shape0", "b_shape1",
        # memory layout
        "c_stride0", "c_stride1", "c_stride2", "a_stride0", "a_stride1", "a_stride2", "b_stride0", "b_stride1",
        # optional fused rmsnorm output layout
        "c_rmsnorm_sum_of_squares_stride0", "c_rmsnorm_sum_of_squares_stride1",
        # optional second B matrix for SwiGLU / GeGLU-style fusion
        "b2_stride0", "b2_stride1",
        # codegen-affecting constexpr callables
        "loader_fn", "storer_fn", "c_c2_merge_tiles_fn",
        # dtype / quantization specialization
        "A_DTYPE", "B_DTYPE", "B2_DTYPE", "C_DTYPE", "ACCUM_DTYPE",
        # fmt: on
    ],
)
@triton.jit
def _matmul_a3d_b2d_kernel(
    # fmt: off
    c_ptr, a_ptr, b_ptr,
    c_shape0: tl.constexpr, c_shape1: tl.constexpr, c_shape2: tl.constexpr,
    a_shape0: tl.constexpr, a_shape1: tl.constexpr, a_shape2: tl.constexpr,
    b_shape0: tl.constexpr, b_shape1: tl.constexpr,
    # b2_shape0 = b_shape0, b2_shape1 = b_shape1
    c_stride0: tl.constexpr, c_stride1: tl.constexpr, c_stride2: tl.constexpr,
    a_stride0: tl.constexpr, a_stride1: tl.constexpr, a_stride2: tl.constexpr,
    b_stride0: tl.constexpr, b_stride1: tl.constexpr,
    A_BLOCKSIZE0: tl.constexpr, A_BLOCKSIZE1: tl.constexpr, A_BLOCKSIZE2: tl.constexpr,
    B_BLOCKSIZE1: tl.constexpr,  # B_BLOCKSIZE0 = A_BLOCKSIZE2
    # C_BLOCKSIZE0 = A_BLOCKSIZE0, C_BLOCKSIZE1 = A_BLOCKSIZE1, C_BLOCKSIZE2 = B_BLOCKSIZE1
    GROUPSIZE1: tl.constexpr, NUM_K_SPLITS: tl.constexpr,
    loader_fn: tl.constexpr, storer_fn: tl.constexpr,
    A_DTYPE: tl.constexpr, B_DTYPE: tl.constexpr, B2_DTYPE: tl.constexpr | None, C_DTYPE: tl.constexpr,  # e.g. q4_k
    ACCUM_DTYPE: tl.constexpr,
    KEEP_C: tl.constexpr,  # if true, init c = c_desc.load(...)
    # the b2_ptr mechanism can be used for GeGLU fusion. storer_extra_ptr is used for the PLE layers.
    c_rmsnorm_sum_of_squares_ptr = None, b2_ptr = None, storer_extra_ptr = None,
    c_rmsnorm_sum_of_squares_stride0: tl.constexpr = 0, c_rmsnorm_sum_of_squares_stride1: tl.constexpr = 0,
    b2_stride0: tl.constexpr = 0, b2_stride1: tl.constexpr = 0,
    c_c2_merge_tiles_fn: tl.constexpr | None = None,
    # these args are here so when b2 = None and launch doesn't decompose tensor, it doesn't error
    b2: None = None, c_rmsnorm_sum_of_squares: None = None, storer_extra: None = None
    # fmt: on
):
    tl.static_assert(a_shape2 == b_shape0 and a_shape0 == c_shape0 and a_shape1 == c_shape1 and b_shape1 == c_shape2)
    tl.static_assert(a_shape2 % NUM_K_SPLITS == 0)

    if b2_ptr is not None:
        # disables k splits (it will also disable it in the grid, forcing only 1 k split program id 2)
        NUM_K_SPLITS = 1

    # TODO I think this indexing is actually wrong when GROUPSIZE does not evenly divide the number of tiles
    k_split_step = tl.cdiv(a_shape2, NUM_K_SPLITS)
    N = tl.cdiv(b_shape1, B_BLOCKSIZE1)
    pid = tl.program_id(0)
    tile_b = tl.program_id(1)
    tile_k_split = tl.program_id(2)
    tile_row = pid % GROUPSIZE1 + (pid // (N * GROUPSIZE1)) * GROUPSIZE1
    tile_col = pid // GROUPSIZE1 % N

    off_b = tile_b * A_BLOCKSIZE0
    off_row = tile_row * A_BLOCKSIZE1
    off_col = tile_col * B_BLOCKSIZE1
    k_split_start = k_split_step * tile_k_split

    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        (a_shape0, a_shape1, a_shape2),
        (a_stride0, a_stride1, a_stride2),
        (A_BLOCKSIZE0, A_BLOCKSIZE1, A_BLOCKSIZE2),
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        (1, b_shape0, b_shape1),
        (0, b_stride0, b_stride1),
        (1, A_BLOCKSIZE2, B_BLOCKSIZE1),
    )
    has_b2: tl.constexpr = b2_ptr is not None
    b2_desc = (
        tl.make_tensor_descriptor(
            b2_ptr,
            (1, b_shape0, b_shape1),
            (0, b2_stride0, b2_stride1),
            (1, A_BLOCKSIZE2, B_BLOCKSIZE1),
        )
        if has_b2
        else None
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        (c_shape0, c_shape1, c_shape2),
        (c_stride0, c_stride1, c_stride2),
        (A_BLOCKSIZE0, A_BLOCKSIZE1, B_BLOCKSIZE1),
    )

    c = (
        c_desc.load((off_b, off_row, off_col)).to(ACCUM_DTYPE)
        if KEEP_C
        else tl.zeros((A_BLOCKSIZE0, A_BLOCKSIZE1, B_BLOCKSIZE1), dtype=ACCUM_DTYPE)
    )
    c2 = tl.zeros((A_BLOCKSIZE0, A_BLOCKSIZE1, B_BLOCKSIZE1), dtype=ACCUM_DTYPE) if has_b2 else None

    for off_k in tl.range(k_split_start, k_split_start + k_split_step, A_BLOCKSIZE2):
        BLOCK_M: tl.constexpr = A_BLOCKSIZE0 * A_BLOCKSIZE1
        BLOCK_K: tl.constexpr = A_BLOCKSIZE2
        BLOCK_N: tl.constexpr = B_BLOCKSIZE1
        a = loader_fn("a", a_desc, off_b, off_row, off_k, A_DTYPE).reshape((BLOCK_M, BLOCK_K))
        b = loader_fn("b", b_desc, 0, off_k, off_col, B_DTYPE).reshape((BLOCK_K, BLOCK_N))
        c = tl.dot(a, b, c.reshape((BLOCK_M, BLOCK_N)), out_dtype=c.dtype).reshape(
            (A_BLOCKSIZE0, A_BLOCKSIZE1, B_BLOCKSIZE1)
        )
        if has_b2:
            b2_tile = loader_fn("b2", b2_desc, 0, off_k, off_col, B2_DTYPE).reshape(b.shape)
            c2 = tl.dot(a, b2_tile, c2.reshape((BLOCK_M, BLOCK_N)), out_dtype=c.dtype).reshape(c.shape)

    if has_b2:
        c = c_c2_merge_tiles_fn(c, c2, off_b, off_row, off_col, NUM_K_SPLITS, C_DTYPE)

    storer_fn(
        "c",
        c_desc,
        c,
        off_b,
        off_row,
        off_col,
        c_rmsnorm_sum_of_squares_ptr,
        storer_extra_ptr,
        c_shape0,
        c_shape1,
        c_rmsnorm_sum_of_squares_stride0,
        c_rmsnorm_sum_of_squares_stride1,
        NUM_K_SPLITS,
        C_DTYPE,
    )


@triton.jit
def matmul_a3d_b2d_loader_jfn(name: tl.constexpr, desc, off0, off1, off2, conceptual_dtype: tl.constexpr):
    tile = desc.load((off0, off1, off2))
    # TODO
    if conceptual_dtype == tensor.q4_k.name:
        ...
    elif conceptual_dtype == tensor.q5_k.name:
        ...
    elif conceptual_dtype == tensor.q6_k.name:
        ...
    return tile


@triton.jit
def matmul_a3d_b2d_partial_rmsnorm_storer_jfn(
    name: tl.constexpr,
    desc,
    tile,
    off0,
    off1,
    off2,
    rsos_ptr,
    extra_ptr,
    rsos_shape0: tl.constexpr,
    rsos_shape1: tl.constexpr,
    rsos_stride0: tl.constexpr,
    rsos_stride1: tl.constexpr,
    NUM_K_SPLITS: tl.constexpr,
    C_DTYPE: tl.constexpr,  # e.g. "q4_k", ignored here for convenience (no case where it differs from desc.dtype)
):
    if rsos_ptr is not None:
        rsos = (tile * tile).sum(-1)
        rsos_offsets0 = (off0 + tl.arange(0, tile.shape[0]))[:, None]
        rsos_offsets1 = (off1 + tl.arange(0, tile.shape[1]))[None, :]
        rsos_offsets = rsos_offsets0 * rsos_stride0 + rsos_offsets1 * rsos_stride1
        tl.atomic_add(
            rsos_ptr + rsos_offsets,
            rsos.to(rsos_ptr.dtype.element_ty),
            mask=(rsos_offsets0 < rsos_shape0) & (rsos_offsets1 < rsos_shape1),
        )
    if NUM_K_SPLITS == 1:
        desc.store((off0, off1, off2), tile.to(desc.dtype))
    else:
        desc.atomic_add((off0, off1, off2), tile.to(desc.dtype))


def matmul_a3d_b2d(
    c: Tensor,
    c_rmsnorm_sum_of_squares: Tensor | None,
    a: Tensor,
    b: Tensor,
    b2: Tensor | None = None,
    storer_extra: Tensor | None = None,
    loader_fn: tl.constexpr = matmul_a3d_b2d_loader_jfn,
    storer_fn: tl.constexpr = matmul_a3d_b2d_partial_rmsnorm_storer_jfn,
    c_c2_merge_tiles_fn: tl.constexpr | None = None,
    accum_dtype: DType | None = None,
    keep_c: bool = False,
):
    assert (b2 is None) == (c_c2_merge_tiles_fn is None)

    k_split_allowed = b2 is None

    grid_fn = lambda META: (
        triton.cdiv(META["a_shape1"], META["A_BLOCKSIZE1"]) * triton.cdiv(META["b_shape1"], META["B_BLOCKSIZE1"]),
        triton.cdiv(META["a_shape0"], META["A_BLOCKSIZE0"]),
        META["NUM_K_SPLITS"] if k_split_allowed else 1,
    )

    if accum_dtype is None:
        # TODO add parameters and/or heuristics for when to use what. This should depend on hardware architecture,
        #  e.g. on ada/hopper you should really really use INT8 for speed if possible,
        #  whereas on volta for example fp16 + fp32 accum is the only viable choice because of the limited tensor cores.
        accum_dtype = tl.float32

    launch[_matmul_a3d_b2d_kernel, grid_fn](
        c=c,
        c_rmsnorm_sum_of_squares=c_rmsnorm_sum_of_squares,
        a=a,
        b=b,
        b2=b2,
        storer_extra=storer_extra,
        loader_fn=loader_fn,
        storer_fn=storer_fn,
        c_c2_merge_tiles_fn=c_c2_merge_tiles_fn,
        A_DTYPE=_dtype_name(a.dtype),
        B_DTYPE=_dtype_name(b.dtype),
        B2_DTYPE=_dtype_name(b2.dtype) if b2 is not None else None,
        C_DTYPE=_dtype_name(c.dtype),
        ACCUM_DTYPE=accum_dtype,
        KEEP_C=keep_c,
    )


def _dtype_name(dtype):
    if isinstance(dtype, DType):
        return dtype.name
    return str(dtype).split(".")[1]  # torch dtype
