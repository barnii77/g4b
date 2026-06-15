import math
import triton
from triton import language as tl
from g4b import tensor
from g4b.tensor import Tensor, DType
from g4b.kernels.utils import launch, default_bencher, jfn_cache_key
from g4b.kernels.memset import memset_contiguous_by_ptr
from g4b.kernels.rsos import compute_rsos
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

# TODO It turns out K-quants are a retarded format for GPUs. There's room for optimization by repacking into a
#  custom format with better block layout at load time... However I fear such a repack would be lossy.


def _contiguous_ignoring_unit_dims(shape, strides):
    # size-1 dims may carry any stride (their data occupies nothing), so the underlying block is still
    # contiguous. This lets t_now=1 / B=1 prefix slices pass the contiguity requirement.
    expected = contiguous_strides_for_shape(shape)
    return all(strides[i] == expected[i] for i in range(len(shape)) if shape[i] != 1)


def _pre_hook(args):
    split_k = args["NUM_K_SPLITS"]
    shape = [args["c_shape0"], args["c_shape1"], args["c_shape2"]]
    strides = [args["c_stride0"], args["c_stride1"], args["c_stride2"]]
    if args.get("c_rmsnorm_sum_of_squares_ptr") is not None:
        shape_norm = shape[:-1]
        strides_norm = [args["c_rmsnorm_sum_of_squares_stride0"], args["c_rmsnorm_sum_of_squares_stride1"]]
        assert _contiguous_ignoring_unit_dims(shape_norm, strides_norm), "sum of squares buffer must be contiguous"
        memset_contiguous_by_ptr(args["c_rmsnorm_sum_of_squares_ptr"], math.prod(shape[:-1]), 0)
    if split_k > 1:
        # TODO not inherently, but I don't want to write more memcpy kernels
        assert _contiguous_ignoring_unit_dims(shape, strides), "split k requires contiguous output buffer"
        if not args["KEEP_C"]:
            memset_contiguous_by_ptr(args["c_ptr"], math.prod(shape), 0)


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
    )


def _matmul_3d_autotune_configs():
    return [
        # ---- aggressive ----
        _cfg(1, 32, 32, 32, 8, warps=4, stages=3),
        _cfg(1, 64, 32, 32, 8, warps=4, stages=3),
        _cfg(1, 64, 32, 128, 8, warps=4, stages=3),
        _cfg(1, 64, 32, 128, 8, split_k=2, warps=4, stages=3),
        _cfg(1, 128, 32, 128, 8, warps=4, stages=3),
        _cfg(1, 64, 64, 128, 8, warps=4, stages=3),
        _cfg(1, 64, 64, 64, 8, warps=4, stages=3),
        _cfg(1, 64, 64, 256, 8, warps=4, stages=3),
        _cfg(1, 64, 128, 64, 8, warps=4, stages=3),
        _cfg(1, 64, 128, 256, 8, warps=4, stages=3),
        _cfg(2, 64, 64, 128, 8, warps=4, stages=3),
        _cfg(2, 64, 128, 128, 8, warps=4, stages=3),
        # ---- small / skinny-N / decode-ish ----
        # Effective MxN: 16x16, 16x32, 32x16, 32x32
        # _cfg(1, 16, 64, 16, 1, warps=4, stages=3),
        # # _cfg(1, 16, 64, 32, 8, warps=4, stages=3),
        # # _cfg(2, 16, 64, 16, 8, warps=4, stages=3),
        # _cfg(2, 16, 64, 32, 8, warps=4, stages=3),
        # # ---- normal balanced tiles ----
        # # Effective MxN: 32x64, 64x32, 64x64
        # # _cfg(2, 16, 64, 64, 8, warps=4, stages=3),
        # # _cfg(4, 16, 64, 32, 8, warps=4, stages=3),
        # _cfg(4, 16, 64, 64, 8, warps=4, stages=3),
        # # Larger K tile: usually good when K is big and register pressure is fine.
        # # _cfg(2, 16, 128, 64, 8, warps=4, stages=3),
        # # _cfg(4, 16, 128, 32, 8, warps=4, stages=3),
        # _cfg(4, 16, 128, 64, 8, warps=4, stages=3),
        # # ---- bigger output tiles ----
        # # Effective MxN: 64x128, 128x64, 128x128
        # # _cfg(4, 16, 64, 128, 8, warps=4, stages=4),
        # # _cfg(8, 16, 64, 64, 8, warps=4, stages=4),
        # # _cfg(8, 16, 64, 128, 8, warps=8, stages=4),
        # _cfg(4, 16, 128, 128, 8, warps=4, stages=4),
        # # _cfg(8, 16, 128, 64, 8, warps=4, stages=4),
        # _cfg(8, 16, 128, 128, 8, warps=8, stages=4),
        # # ---- split-K variants ----
        # _cfg(2, 16, 64, 64, 8, split_k=2, warps=4, stages=3),
        # # _cfg(4, 16, 64, 64, 8, split_k=2, warps=4, stages=3),
        # # _cfg(4, 16, 128, 64, 8, split_k=2, warps=4, stages=4),
        # # _cfg(2, 16, 64, 64, 8, split_k=4, warps=4, stages=3),
        # _cfg(4, 16, 64, 64, 8, split_k=4, warps=4, stages=3),
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
        # optional epilogue inputs
        "input_rmsnorm_sum_of_squares_shape0", "input_rmsnorm_sum_of_squares_shape1",
        "input_rmsnorm_sum_of_squares_stride0", "input_rmsnorm_sum_of_squares_stride1",
        "storer_extra_shape0", "storer_extra_shape1", "storer_extra_shape2",
        "storer_extra_stride0", "storer_extra_stride1", "storer_extra_stride2",
        "rmsnorm_eps",
        "TRANSPOSE_B_BEFORE_MMA",
        # codegen-affecting constexpr callables
        # "a_loader_fn", "b_loader_fn", "storer_fn", "c_c2_merge_tiles_fn",
        # hack so autotune results are cacheable to disk
        "_a_loader_fn_key", "_b_loader_fn_key", "_storer_fn_key", "_c_c2_merge_tiles_fn_key",
        # dtype / quantization specialization
        "A_DTYPE", "B_DTYPE", "B2_DTYPE", "C_DTYPE", "_ACCUM_DTYPE_CACHE_KEY",
        # fmt: on
    ],
    do_bench=default_bencher,
    cache_results=True,
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
    a_loader_fn: tl.constexpr, b_loader_fn: tl.constexpr, storer_fn: tl.constexpr,
    A_DTYPE: tl.constexpr, B_DTYPE: tl.constexpr, B2_DTYPE: tl.constexpr | None, C_DTYPE: tl.constexpr,  # e.g. q4_k
    ACCUM_DTYPE: tl.constexpr,
    KEEP_C: tl.constexpr,  # if true, init c = c_desc.load(...)
    TRANSPOSE_B_BEFORE_MMA: tl.constexpr,
    # make caching to disk work (hacky)
    _a_loader_fn_key: tl.constexpr, _b_loader_fn_key: tl.constexpr, _storer_fn_key: tl.constexpr,
    _c_c2_merge_tiles_fn_key: tl.constexpr,
    _ACCUM_DTYPE_CACHE_KEY: tl.constexpr,
    # the b2_ptr mechanism can be used for GeGLU fusion. storer_extra_ptr is used by custom storers.
    c_rmsnorm_sum_of_squares_ptr = None, b2_ptr = None, storer_extra_ptr = None,
    input_rmsnorm_sum_of_squares_ptr = None,
    c_rmsnorm_sum_of_squares_stride0: tl.constexpr = 0, c_rmsnorm_sum_of_squares_stride1: tl.constexpr = 0,
    b2_stride0: tl.constexpr = 0, b2_stride1: tl.constexpr = 0,
    input_rmsnorm_sum_of_squares_shape0: tl.constexpr = 0, input_rmsnorm_sum_of_squares_shape1: tl.constexpr = 0,
    input_rmsnorm_sum_of_squares_stride0: tl.constexpr = 0, input_rmsnorm_sum_of_squares_stride1: tl.constexpr = 0,
    storer_extra_shape0: tl.constexpr = 0, storer_extra_shape1: tl.constexpr = 0,
    storer_extra_shape2: tl.constexpr = 0,
    storer_extra_stride0: tl.constexpr = 0, storer_extra_stride1: tl.constexpr = 0,
    storer_extra_stride2: tl.constexpr = 0,
    rmsnorm_eps: tl.constexpr = 0.0,
    c_c2_merge_tiles_fn: tl.constexpr | None = None,
    # these args are here so when b2 = None and launch doesn't decompose tensor, it doesn't error
    b2: None = None, c_rmsnorm_sum_of_squares: None = None, storer_extra: None = None,
    input_rmsnorm_sum_of_squares: None = None,
    # fmt: on
):
    if TRANSPOSE_B_BEFORE_MMA:
        tl.static_assert(
            a_shape2 == b_shape1 and a_shape0 == c_shape0 and a_shape1 == c_shape1 and b_shape0 == c_shape2
        )
    else:
        tl.static_assert(
            a_shape2 == b_shape0 and a_shape0 == c_shape0 and a_shape1 == c_shape1 and b_shape1 == c_shape2
        )
    tl.static_assert(a_shape2 % NUM_K_SPLITS == 0)

    if b2_ptr is not None:
        # disables k splits (it will also disable it in the grid, forcing only 1 k split program id 2)
        NUM_K_SPLITS = 1

    k_split_step = tl.cdiv(a_shape2, NUM_K_SPLITS)
    num_tile_rows = tl.cdiv(a_shape1, A_BLOCKSIZE1)
    num_tile_cols = tl.cdiv(c_shape2, B_BLOCKSIZE1)
    pid = tl.program_id(0)
    tile_b = tl.program_id(1)
    tile_k_split = tl.program_id(2)
    num_tiles_per_group = GROUPSIZE1 * num_tile_cols
    group_id = pid // num_tiles_per_group
    first_tile_row = group_id * GROUPSIZE1
    group_size_rows = tl.minimum(num_tile_rows - first_tile_row, GROUPSIZE1)
    pid_in_group = pid % num_tiles_per_group
    tile_row = first_tile_row + pid_in_group % group_size_rows
    tile_col = pid_in_group // group_size_rows

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
    if TRANSPOSE_B_BEFORE_MMA:
        b_desc = tl.make_tensor_descriptor(
            b_ptr,
            (1, b_shape0, b_shape1),
            (0, b_stride0, b_stride1),
            (1, B_BLOCKSIZE1, A_BLOCKSIZE2),
        )
    else:
        b_desc = tl.make_tensor_descriptor(
            b_ptr,
            (1, b_shape0, b_shape1),
            (0, b_stride0, b_stride1),
            (1, A_BLOCKSIZE2, B_BLOCKSIZE1),
        )
    has_b2: tl.constexpr = b2_ptr is not None
    if has_b2:
        if TRANSPOSE_B_BEFORE_MMA:
            b2_desc = tl.make_tensor_descriptor(
                b2_ptr,
                (1, b_shape0, b_shape1),
                (0, b2_stride0, b2_stride1),
                (1, B_BLOCKSIZE1, A_BLOCKSIZE2),
            )
        else:
            b2_desc = tl.make_tensor_descriptor(
                b2_ptr,
                (1, b_shape0, b_shape1),
                (0, b2_stride0, b2_stride1),
                (1, A_BLOCKSIZE2, B_BLOCKSIZE1),
            )
    else:
        b2_desc = None
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
        # fmt: off
        a = a_loader_fn(
            "a", a_desc, off_b, off_row, off_k, a_ptr,
            a_shape0, a_shape1, a_shape2,
            a_stride0, a_stride1, a_stride2,
            A_BLOCKSIZE0, A_BLOCKSIZE1, A_BLOCKSIZE2,
            A_DTYPE,
        ).reshape((BLOCK_M, BLOCK_K))
        if TRANSPOSE_B_BEFORE_MMA:
            b = b_loader_fn(
                "b", b_desc, 0, off_col, off_k, b_ptr,
                1, b_shape0, b_shape1,
                0, b_stride0, b_stride1,
                1, B_BLOCKSIZE1, A_BLOCKSIZE2,
                B_DTYPE,
            ).reshape((BLOCK_N, BLOCK_K)).T
        else:
            b = b_loader_fn(
                "b", b_desc, 0, off_k, off_col, b_ptr,
                1, b_shape0, b_shape1,
                0, b_stride0, b_stride1,
                1, A_BLOCKSIZE2, B_BLOCKSIZE1,
                B_DTYPE,
            ).reshape((BLOCK_K, BLOCK_N))
        # fmt: on

        # Handle upcasting.
        tl.static_assert(a.dtype == tl.float32 or a.dtype == tl.float16 or a.dtype == tl.bfloat16)
        tl.static_assert(b.dtype == tl.float32 or b.dtype == tl.float16 or b.dtype == tl.bfloat16)
        # If a and b have the same float dtype we keep it (fp16/bf16 -> tensor cores); otherwise (e.g. bf16
        # activations against fp16-dequantized weights) we upcast both to fp32 for a correct, if slower, mma.
        if a.dtype == b.dtype:
            UPCAST_TY = a.dtype
        else:
            UPCAST_TY = tl.float32

        a_upcast = a.to(UPCAST_TY)

        c = tl.dot(a_upcast, b.to(UPCAST_TY), c.reshape((BLOCK_M, BLOCK_N)), out_dtype=c.dtype).reshape(
            (A_BLOCKSIZE0, A_BLOCKSIZE1, B_BLOCKSIZE1)
        )
        if has_b2:
            # TODO maybe t.join'ing the b and b2 tiles and doing one tl.dot call then splitting again is faster? try!
            # fmt: off
            if TRANSPOSE_B_BEFORE_MMA:
                b2_tile = b_loader_fn(
                    "b2", b2_desc, 0, off_col, off_k, b2_ptr,
                    1, b_shape0, b_shape1,
                    0, b2_stride0, b2_stride1,
                    1, B_BLOCKSIZE1, A_BLOCKSIZE2,
                    B2_DTYPE,
                ).reshape((BLOCK_N, BLOCK_K)).T.to(UPCAST_TY)
            else:
                b2_tile = b_loader_fn(
                    "b2", b2_desc, 0, off_k, off_col, b2_ptr,
                    1, b_shape0, b_shape1,
                    0, b2_stride0, b2_stride1,
                    1, A_BLOCKSIZE2, B_BLOCKSIZE1,
                    B2_DTYPE,
                ).reshape((BLOCK_K, BLOCK_N)).to(UPCAST_TY)
            # fmt: on
            c2 = tl.dot(a_upcast, b2_tile, c2.reshape((BLOCK_M, BLOCK_N)), out_dtype=c.dtype).reshape(c.shape)

    if has_b2:
        # fmt: off
        c = c_c2_merge_tiles_fn(
            c, c2, off_b, off_row, off_col,
            NUM_K_SPLITS, C_DTYPE,
            input_rmsnorm_sum_of_squares_ptr,
            input_rmsnorm_sum_of_squares_shape0, input_rmsnorm_sum_of_squares_shape1,
            input_rmsnorm_sum_of_squares_stride0, input_rmsnorm_sum_of_squares_stride1,
            a_shape2, rmsnorm_eps,
        )
        # fmt: on

    # fmt: off
    storer_fn(
        "c", c_desc, c, off_b, off_row, off_col,
        c_rmsnorm_sum_of_squares_ptr, storer_extra_ptr,
        c_shape0, c_shape1,
        c_rmsnorm_sum_of_squares_stride0, c_rmsnorm_sum_of_squares_stride1,
        NUM_K_SPLITS, C_DTYPE,
        input_rmsnorm_sum_of_squares_ptr,
        input_rmsnorm_sum_of_squares_shape0, input_rmsnorm_sum_of_squares_shape1,
        input_rmsnorm_sum_of_squares_stride0, input_rmsnorm_sum_of_squares_stride1,
        a_shape2, rmsnorm_eps,
        storer_extra_shape0, storer_extra_shape1, storer_extra_shape2,
        storer_extra_stride0, storer_extra_stride1, storer_extra_stride2,
    )
    # fmt: on


@triton.jit
def matmul_a3d_b2d_a_loader_jfn(
    # fmt: off
    name: tl.constexpr, desc, off0, off1, off2, ptr,
    shape0: tl.constexpr, shape1: tl.constexpr, shape2: tl.constexpr,
    stride0: tl.constexpr, stride1: tl.constexpr, stride2: tl.constexpr,
    BLOCKSIZE0: tl.constexpr, BLOCKSIZE1: tl.constexpr, BLOCKSIZE2: tl.constexpr,
    conceptual_dtype: tl.constexpr,
    # fmt: on
):
    return desc.load((off0, off1, off2))


@triton.jit
def matmul_a3d_b2d_b_loader_jfn(
    # fmt: off
    name: tl.constexpr, desc, off0, off1, off2, ptr,
    shape0: tl.constexpr, shape1: tl.constexpr, shape2: tl.constexpr,
    stride0: tl.constexpr, stride1: tl.constexpr, stride2: tl.constexpr,
    BLOCKSIZE0: tl.constexpr, BLOCKSIZE1: tl.constexpr, BLOCKSIZE2: tl.constexpr,
    conceptual_dtype: tl.constexpr,
    # fmt: on
):
    is_quantized: tl.constexpr = (
        conceptual_dtype == tensor.q4_k.name
        or conceptual_dtype == tensor.q5_k.name
        or conceptual_dtype == tensor.q6_k.name
    )
    if not is_quantized:
        return desc.load((off0, off1, off2)).to(tl.float16)

    # desc not used, we manually tl.load
    ptr_u8 = ptr.to(tl.pointer_type(tl.uint8))
    tl.static_assert(not is_quantized or ptr_u8.dtype.element_ty == tl.uint8)

    # each row must fit in one superblock for now
    k_quant_superblock_num_elems: tl.constexpr = 256
    tl.static_assert(not is_quantized or BLOCKSIZE2 <= k_quant_superblock_num_elems)

    # dtype to which it upcasts values
    UPCAST_DTYPE: tl.constexpr = tl.float16

    # for quantized loads, tensor must be 2d for now
    tl.static_assert(not is_quantized or shape0 == 1)
    tl.static_assert(not is_quantized or stride0 == 0)
    tl.static_assert(not is_quantized or BLOCKSIZE0 == 1)
    tl.static_assert(
        not is_quantized or stride2 == 1
    )  # superblocks are physically laid out as contiguous 256-byte arrays
    tl.static_assert(not is_quantized or BLOCKSIZE2 >= 32)  # hacky assert but makes my life easier below

    stride_row: tl.constexpr = stride1
    BLOCKSIZE_ROW: tl.constexpr = BLOCKSIZE1
    BLOCKSIZE_COL: tl.constexpr = BLOCKSIZE2
    first_row = off1
    first_col = off2
    SUPERBLOCK_SIZE_ELEMS: tl.constexpr = 256
    offs_row = first_row + tl.arange(0, BLOCKSIZE_ROW)[:, None]

    # TODO a lot of the indexing can probably be optimized and maybe one could also avoid loading all sub-block scales.
    # TODO handle BLOCKSIZE_COL == 16 correctly

    # full ahead-of-mma dequant
    if conceptual_dtype == tensor.q4_k.name:
        # q4_k
        SUPERBLOCK_SIZE_BYTES: tl.constexpr = 144
        SUBBLOCK_SIZE_ELEMS: tl.constexpr = 32
        SUBBLOCK_SIZE_BYTES: tl.constexpr = 16

        rows: tl.constexpr = shape1
        cols: tl.constexpr = shape2 // SUPERBLOCK_SIZE_BYTES * SUPERBLOCK_SIZE_ELEMS
        tl.static_assert(not is_quantized or cols % SUPERBLOCK_SIZE_ELEMS == 0)

        ptr_u16 = ptr_u8.cast(tl.pointer_type(tl.uint16))
        sizeof_u16 = 2

        superblock_id_col = first_col // SUPERBLOCK_SIZE_ELEMS
        sb_first_byte_col_off = superblock_id_col * SUPERBLOCK_SIZE_BYTES
        dd_ptrs_u16 = ptr_u16 + offs_row * (stride_row // sizeof_u16) + sb_first_byte_col_off // sizeof_u16
        mask = offs_row < rows
        dd = tl.load(dd_ptrs_u16, mask=mask, other=0.0).cast(tl.float16, bitcast=True).to(UPCAST_DTYPE)
        md = tl.load(dd_ptrs_u16 + 1, mask=mask, other=0.0).cast(tl.float16, bitcast=True).to(UPCAST_DTYPE)

        dd_ptrs_u8 = ptr_u8 + offs_row * stride_row + sb_first_byte_col_off
        d_frags = tl.load(dd_ptrs_u8 + 4 + tl.arange(0, 4), mask=mask, other=0.0)
        m_frags = tl.load(dd_ptrs_u8 + 8 + tl.arange(0, 4), mask=mask, other=0.0)
        mixed_frags = tl.load(dd_ptrs_u8 + 12 + tl.arange(0, 4), mask=mask, other=0.0)

        sc = tl.cat(d_frags & 0x3F, ((d_frags & 0xC0) >> 2) | (mixed_frags & 0x0F), dim=-1)
        mins = tl.cat(m_frags & 0x3F, ((m_frags & 0xC0) >> 2) | (mixed_frags >> 4), dim=-1)

        all_ds = sc.to(UPCAST_DTYPE) * dd
        all_ms = mins.to(UPCAST_DTYPE) * md

        Q4_K_REQUIRES_INEFFICIENT_LOAD_PATTERN: tl.constexpr = BLOCKSIZE_COL <= SUBBLOCK_SIZE_ELEMS
        n_subblocks_to_load: tl.constexpr = (
            tl.cdiv(BLOCKSIZE_COL * 2, SUBBLOCK_SIZE_ELEMS)
            if Q4_K_REQUIRES_INEFFICIENT_LOAD_PATTERN
            else tl.cdiv(BLOCKSIZE_COL, SUBBLOCK_SIZE_ELEMS)
        )
        first_subblock_id = (
            # produces 0,1,0,1,2,3,2,3,4,5,4,5,... pattern - handles Q4_K_REQUIRES_INEFFICIENT_LOAD_PATTERN
            (first_col % SUPERBLOCK_SIZE_ELEMS) // (SUBBLOCK_SIZE_ELEMS * 2) * 2
            + (first_col % SUPERBLOCK_SIZE_ELEMS % SUBBLOCK_SIZE_ELEMS) // (SUBBLOCK_SIZE_ELEMS // 2)
        )
        ql_packed = tl.load(
            ptr_u8
            + offs_row * stride_row
            + sb_first_byte_col_off
            + 16  # skip first "subblock" which contains block scales/mins
            + first_subblock_id * SUBBLOCK_SIZE_BYTES
            + tl.arange(0, n_subblocks_to_load * SUBBLOCK_SIZE_BYTES)[None, :],
            mask=mask,
            other=0.0,
        )

        # pull high 4 bits and low 4 bits apart into 2 values (order: low then high)
        qs_unpacked = ql_packed.reshape((BLOCKSIZE_ROW, n_subblocks_to_load // 2, 1, 32)) >> (
            tl.arange(0, 2).to(tl.uint8) * 4
        ).reshape((1, 1, 2, 1))
        qs_unpacked &= 0x0F

        qs_reshaped = qs_unpacked.reshape((BLOCKSIZE_ROW, n_subblocks_to_load, 32))
        if Q4_K_REQUIRES_INEFFICIENT_LOAD_PATTERN:
            # discard either the top or bottom half (had to be loaded though for dequant because K-quants are dumb)
            required_idx_val = first_col % 64 // 32
            required_idx = tl.full((qs_reshaped.shape[0], 1, qs_reshaped.shape[2]), required_idx_val, dtype=tl.int32)
            required_idx_dm_offs = tl.full((1, 1), required_idx_val, dtype=tl.int32)
            assert required_idx <= 1
            qs_selected = qs_reshaped.gather(required_idx, axis=1)
            dm_offs = (
                (first_subblock_id + tl.arange(0, n_subblocks_to_load)[None, :])
                .gather(required_idx_dm_offs, axis=1)
                .broadcast_to((BLOCKSIZE_ROW, required_idx_dm_offs.shape[1]))
            )
        else:
            qs_selected = qs_reshaped
            dm_offs = (first_subblock_id + tl.arange(0, n_subblocks_to_load)[None, :]).broadcast_to(
                (BLOCKSIZE_ROW, n_subblocks_to_load)
            )
        qs = qs_selected.to(UPCAST_DTYPE)

        ds = all_ds.gather(dm_offs, axis=-1)
        ms = all_ms.gather(dm_offs, axis=-1)

        return (ds.expand_dims(-1) * qs - ms.expand_dims(-1)).reshape((1, BLOCKSIZE_ROW, BLOCKSIZE_COL))
    elif conceptual_dtype == tensor.q5_k.name:
        # q5_k
        SUPERBLOCK_SIZE_BYTES: tl.constexpr = 176
        SUBBLOCK_SIZE_ELEMS: tl.constexpr = 32
        SUBBLOCK_SIZE_BYTES: tl.constexpr = 16

        rows: tl.constexpr = shape1
        cols: tl.constexpr = shape2 // SUPERBLOCK_SIZE_BYTES * SUPERBLOCK_SIZE_ELEMS
        tl.static_assert(not is_quantized or cols % SUPERBLOCK_SIZE_ELEMS == 0)

        ptr_u16 = ptr_u8.cast(tl.pointer_type(tl.uint16))
        sizeof_u16 = 2

        superblock_id_col = first_col // SUPERBLOCK_SIZE_ELEMS
        sb_first_byte_col_off = superblock_id_col * SUPERBLOCK_SIZE_BYTES
        dd_ptrs_u16 = ptr_u16 + offs_row * (stride_row // sizeof_u16) + sb_first_byte_col_off // sizeof_u16
        mask = offs_row < rows
        dd = tl.load(dd_ptrs_u16, mask=mask, other=0.0).cast(tl.float16, bitcast=True).to(UPCAST_DTYPE)
        md = tl.load(dd_ptrs_u16 + 1, mask=mask, other=0.0).cast(tl.float16, bitcast=True).to(UPCAST_DTYPE)

        dd_ptrs_u8 = ptr_u8 + offs_row * stride_row + sb_first_byte_col_off
        d_frags = tl.load(dd_ptrs_u8 + 4 + tl.arange(0, 4), mask=mask, other=0.0)
        m_frags = tl.load(dd_ptrs_u8 + 8 + tl.arange(0, 4), mask=mask, other=0.0)
        mixed_frags = tl.load(dd_ptrs_u8 + 12 + tl.arange(0, 4), mask=mask, other=0.0)

        sc = tl.cat(d_frags & 0x3F, ((d_frags & 0xC0) >> 2) | (mixed_frags & 0x0F), dim=-1)
        mins = tl.cat(m_frags & 0x3F, ((m_frags & 0xC0) >> 2) | (mixed_frags >> 4), dim=-1)

        all_ds = sc.to(UPCAST_DTYPE) * dd
        all_ms = mins.to(UPCAST_DTYPE) * md

        Q4_K_REQUIRES_INEFFICIENT_LOAD_PATTERN: tl.constexpr = BLOCKSIZE_COL <= SUBBLOCK_SIZE_ELEMS
        n_subblocks_to_load: tl.constexpr = (
            tl.cdiv(BLOCKSIZE_COL * 2, SUBBLOCK_SIZE_ELEMS)
            if Q4_K_REQUIRES_INEFFICIENT_LOAD_PATTERN
            else tl.cdiv(BLOCKSIZE_COL, SUBBLOCK_SIZE_ELEMS)
        )
        first_subblock_id = (
            # produces 0,1,0,1,2,3,2,3,4,5,4,5,... pattern - handles Q4_K_REQUIRES_INEFFICIENT_LOAD_PATTERN
            (first_col % SUPERBLOCK_SIZE_ELEMS) // (SUBBLOCK_SIZE_ELEMS * 2) * 2
            + (first_col % SUPERBLOCK_SIZE_ELEMS % SUBBLOCK_SIZE_ELEMS) // (SUBBLOCK_SIZE_ELEMS // 2)
        )
        ql_packed = tl.load(
            ptr_u8
            + offs_row * stride_row
            + sb_first_byte_col_off
            + 48  # skip first "subblock" which contains block scales/mins
            + first_subblock_id * SUBBLOCK_SIZE_BYTES
            + tl.arange(0, n_subblocks_to_load * SUBBLOCK_SIZE_BYTES)[None, :],
            mask=mask,
            other=0.0,
        )
        qh_packed = tl.load(
            ptr_u8
            + offs_row * stride_row
            + sb_first_byte_col_off
            + 16  # skip first "subblock" which contains block scales/mins
            + tl.arange(0, SUBBLOCK_SIZE_ELEMS)[None, :],
            mask=mask,
            other=0.0,
        )

        # pull high 4 bits and low 4 bits apart into 2 values (order: low then high)
        ql_unpacked = ql_packed.reshape((BLOCKSIZE_ROW, n_subblocks_to_load // 2, 1, 32)) >> (
            tl.arange(0, 2).to(tl.uint8) * 4
        ).reshape((1, 1, 2, 1))
        ql_unpacked &= 0x0F
        qs_reshaped = ql_unpacked.reshape((BLOCKSIZE_ROW, n_subblocks_to_load, 32))

        qh_unpacked = qh_packed.reshape((BLOCKSIZE_ROW, 1, 32)) >> (
            (first_subblock_id + tl.arange(0, n_subblocks_to_load)).to(tl.uint8)
        ).reshape((1, n_subblocks_to_load, 1))
        qh_unpacked &= 0x01
        qs_reshaped |= qh_unpacked << 4

        if Q4_K_REQUIRES_INEFFICIENT_LOAD_PATTERN:
            # discard either the top or bottom half (had to be loaded though for dequant because K-quants are dumb)
            required_idx_val = first_col % 64 // 32
            required_idx = tl.full((qs_reshaped.shape[0], 1, qs_reshaped.shape[2]), required_idx_val, dtype=tl.int32)
            required_idx_dm_offs = tl.full((1, 1), required_idx_val, dtype=tl.int32)
            assert required_idx <= 1
            qs_selected = qs_reshaped.gather(required_idx, axis=1)
            dm_offs = (
                (first_subblock_id + tl.arange(0, n_subblocks_to_load)[None, :])
                .gather(required_idx_dm_offs, axis=1)
                .broadcast_to((BLOCKSIZE_ROW, required_idx_dm_offs.shape[1]))
            )
        else:
            qs_selected = qs_reshaped
            dm_offs = (first_subblock_id + tl.arange(0, n_subblocks_to_load)[None, :]).broadcast_to(
                (BLOCKSIZE_ROW, n_subblocks_to_load)
            )
        qs = qs_selected.to(UPCAST_DTYPE)

        ds = all_ds.gather(dm_offs, axis=-1)
        ms = all_ms.gather(dm_offs, axis=-1)

        return (ds.expand_dims(-1) * qs - ms.expand_dims(-1)).reshape((1, BLOCKSIZE_ROW, BLOCKSIZE_COL))
    else:
        # q6_k
        SUPERBLOCK_SIZE_BYTES: tl.constexpr = 210
        SUBBLOCK_SIZE_ELEMS: tl.constexpr = 16

        rows: tl.constexpr = shape1
        cols: tl.constexpr = shape2 // SUPERBLOCK_SIZE_BYTES * SUPERBLOCK_SIZE_ELEMS
        tl.static_assert(not is_quantized or cols % SUPERBLOCK_SIZE_ELEMS == 0)

        ptr_u16 = ptr_u8.cast(tl.pointer_type(tl.uint16))
        sizeof_u16 = 2

        superblock_id_col = first_col // SUPERBLOCK_SIZE_ELEMS
        sb_first_byte_col_off = superblock_id_col * SUPERBLOCK_SIZE_BYTES
        dd_ptrs_u16 = ptr_u16 + offs_row * (stride_row // sizeof_u16) + sb_first_byte_col_off // sizeof_u16 + 104
        mask = offs_row < rows
        dd = tl.load(dd_ptrs_u16, mask=mask, other=0.0).cast(tl.float16, bitcast=True).to(UPCAST_DTYPE)
        sc_ptrs_u8 = ptr_u8 + offs_row * stride_row + sb_first_byte_col_off + 192 + tl.arange(0, 16)[None, :]
        sc = tl.load(sc_ptrs_u8, mask=mask, other=0.0).cast(tl.int8, bitcast=True)

        all_ds = sc.to(UPCAST_DTYPE) * dd

        tl.static_assert(BLOCKSIZE_COL >= SUBBLOCK_SIZE_ELEMS)
        n_subblocks_to_load: tl.constexpr = tl.cdiv(BLOCKSIZE_COL, SUBBLOCK_SIZE_ELEMS)
        first_subblock_id = (first_col % SUPERBLOCK_SIZE_ELEMS) // SUBBLOCK_SIZE_ELEMS
        col_offs = (first_col % SUPERBLOCK_SIZE_ELEMS) + tl.arange(0, BLOCKSIZE_COL)[None, :]

        ql_byte_offs = (col_offs // 128) * 64 + (col_offs % 64)
        ql_shift = (((col_offs // 64) % 2) * 4).to(tl.uint8)
        ql_packed = tl.load(
            ptr_u8
            + offs_row * stride_row
            + sb_first_byte_col_off
            + ql_byte_offs,
            mask=mask,
            other=0.0,
        )

        qh_byte_offs = 128 + (col_offs // 128) * 32 + (col_offs % 32)
        qh_shift = (((col_offs % 128) // 32) * 2).to(tl.uint8)
        qh_packed = tl.load(
            ptr_u8
            + offs_row * stride_row
            + sb_first_byte_col_off
            + qh_byte_offs,
            mask=mask,
            other=0.0,
        )

        ql_unpacked = (ql_packed >> ql_shift) & 0x0F
        qh_unpacked = (qh_packed >> qh_shift) & 0x03
        qs_selected = ((qh_unpacked << 4) | ql_unpacked).reshape(
            (BLOCKSIZE_ROW, n_subblocks_to_load, SUBBLOCK_SIZE_ELEMS)
        )
        dm_offs = (first_subblock_id + tl.arange(0, n_subblocks_to_load)[None, :]).broadcast_to(
            (BLOCKSIZE_ROW, n_subblocks_to_load)
        )
        qs = (qs_selected.to(tl.int8) - 32).to(UPCAST_DTYPE)

        ds = all_ds.gather(dm_offs, axis=-1)

        return (ds.expand_dims(-1) * qs).reshape((1, BLOCKSIZE_ROW, BLOCKSIZE_COL))


@triton.jit
def matmul_a3d_b2d_partial_rmsnorm_storer_jfn(
    # fmt: off
    name: tl.constexpr, desc, tile, off0, off1, off2, rsos_ptr, extra_ptr,
    rsos_shape0: tl.constexpr, rsos_shape1: tl.constexpr,
    rsos_stride0: tl.constexpr, rsos_stride1: tl.constexpr,
    NUM_K_SPLITS: tl.constexpr, C_DTYPE: tl.constexpr,  # e.g. "q4_k", ignored here for convenience
    input_rsos_ptr=None,
    input_rsos_shape0: tl.constexpr = 0, input_rsos_shape1: tl.constexpr = 0,
    input_rsos_stride0: tl.constexpr = 0, input_rsos_stride1: tl.constexpr = 0,
    rmsnorm_dim: tl.constexpr = 0, rmsnorm_eps: tl.constexpr = 0.0,
    extra_shape0: tl.constexpr = 0, extra_shape1: tl.constexpr = 0, extra_shape2: tl.constexpr = 0,
    extra_stride0: tl.constexpr = 0, extra_stride1: tl.constexpr = 0, extra_stride2: tl.constexpr = 0,
    # fmt: on
):
    if rsos_ptr is not None and NUM_K_SPLITS == 1:
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
    a_loader_fn: tl.constexpr = matmul_a3d_b2d_a_loader_jfn,
    b_loader_fn: tl.constexpr = matmul_a3d_b2d_b_loader_jfn,
    storer_fn: tl.constexpr = matmul_a3d_b2d_partial_rmsnorm_storer_jfn,
    c_c2_merge_tiles_fn: tl.constexpr | None = None,
    accum_dtype: DType | None = None,
    keep_c: bool = False,
    transpose_b_before_mma: bool = False,
    *,
    input_rmsnorm_sum_of_squares: Tensor | None = None,
    rmsnorm_eps: float,
):
    assert (b2 is None) == (c_c2_merge_tiles_fn is None)

    k_split_allowed = b2 is None
    selected_num_k_splits = 1

    # TODO this grid_fn is hacky and I'm not sure launching the pre-hook in the grid_fn is ideal because it is part of
    #  the measurement of the autotuner, but I guess zeroing is a real downside and so it being measured by the
    #  autotuner isn't actually that undesirable?
    def grid_fn(META):
        nonlocal selected_num_k_splits
        selected_num_k_splits = META["NUM_K_SPLITS"] if k_split_allowed else 1
        return _pre_hook(META) or (
            triton.cdiv(META["a_shape1"], META["A_BLOCKSIZE1"]) * triton.cdiv(META["c_shape2"], META["B_BLOCKSIZE1"]),
            triton.cdiv(META["a_shape0"], META["A_BLOCKSIZE0"]),
            selected_num_k_splits,
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
        input_rmsnorm_sum_of_squares=input_rmsnorm_sum_of_squares,
        a_loader_fn=a_loader_fn,
        b_loader_fn=b_loader_fn,
        storer_fn=storer_fn,
        c_c2_merge_tiles_fn=c_c2_merge_tiles_fn,
        A_DTYPE=_dtype_name(a.dtype),
        B_DTYPE=_dtype_name(b.dtype),
        B2_DTYPE=_dtype_name(b2.dtype) if b2 is not None else None,
        C_DTYPE=_dtype_name(c.dtype),
        ACCUM_DTYPE=accum_dtype,
        KEEP_C=keep_c,
        TRANSPOSE_B_BEFORE_MMA=transpose_b_before_mma,
        rmsnorm_eps=float(rmsnorm_eps),
        _a_loader_fn_key=jfn_cache_key(a_loader_fn),
        _b_loader_fn_key=jfn_cache_key(b_loader_fn),
        _storer_fn_key=jfn_cache_key(storer_fn),
        _c_c2_merge_tiles_fn_key=jfn_cache_key(c_c2_merge_tiles_fn),
        _ACCUM_DTYPE_CACHE_KEY=accum_dtype.name,
    )
    if c_rmsnorm_sum_of_squares is not None and selected_num_k_splits > 1:
        compute_rsos(c, c_rmsnorm_sum_of_squares)


def _dtype_name(dtype):
    if isinstance(dtype, DType):
        return dtype.name
    return str(dtype).split(".")[1]  # torch dtype
