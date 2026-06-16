import torch
import time
import triton
from triton import language as tl


@triton.jit
def _matmul_kernel(
    z_ptr,
    a_ptr,
    b_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    epilogue: tl.constexpr,
    num_stages: tl.constexpr,
):
    pid_m_in_group = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_m = tl.program_id(2) * GROUP_M + pid_m_in_group

    a_desc = tl.make_tensor_descriptor(a_ptr, (M, K), (K, 1), (BLOCK_M, BLOCK_K))
    b_desc = tl.make_tensor_descriptor(b_ptr, (K, N), (N, 1), (BLOCK_K, BLOCK_N))
    z_desc = tl.make_tensor_descriptor(z_ptr, (M, N), (N, 1), (BLOCK_M, BLOCK_N))

    ACCUM_DTYPE = tl.float32
    z = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACCUM_DTYPE)
    for k_off in tl.range(0, K, BLOCK_K, num_stages=num_stages):
        a = a_desc.load((pid_m * BLOCK_M, k_off))
        b = b_desc.load((k_off, pid_n * BLOCK_N))
        z = tl.dot(a, b, z, allow_tf32=True, out_dtype=ACCUM_DTYPE)

    z = z.to(z_ptr.dtype.element_ty)
    if epilogue is not None:
        z = epilogue(z)
    z_desc.store((pid_m * BLOCK_M, pid_n * BLOCK_N), z)


@triton.jit
def matmul_kernel_ref(
        # Pointers to matrices
        c_ptr,
        a_ptr, b_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
        num_stages: tl.constexpr
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # -----------------------------------------------------------
    # Add some integer bound assumptions.
    # This helps to guide integer analysis in the backend to optimize
    # load/store offset address calculation
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * K + offs_k[None, :] * 1)
    b_ptrs = b_ptr + (offs_k[:, None] * N + offs_bn[None, :] * 1)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K), num_stages=num_stages):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * 1
        b_ptrs += BLOCK_SIZE_K * N
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    c = accumulator.to(tl.float16)
    # c = accumulator

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + N * offs_cm[:, None] + 1 * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def _matmul_kernel_block_ptrs(
        z_ptr,
        a_ptr,
        b_ptr,
        M,
        N,
        K,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
):
    pid_m_in_group = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_m = tl.program_id(2) * GROUP_M + pid_m_in_group

    a_blk = tl.make_block_ptr(a_ptr, (M, K), (K, 1), (pid_m * BLOCK_M, 0), (BLOCK_M, BLOCK_K), (1, 0))
    b_blk = tl.make_block_ptr(b_ptr, (K, N), (N, 1), (0, pid_n * BLOCK_N), (BLOCK_K, BLOCK_N), (1, 0))

    ACCUM_DTYPE = tl.float32
    z = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACCUM_DTYPE)
    for _ in range(tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_blk)
        b = tl.load(b_blk)
        z = tl.dot(a, b, z, allow_tf32=True, out_dtype=ACCUM_DTYPE)
        a_blk = a_blk.advance((0, BLOCK_K))
        b_blk = b_blk.advance((BLOCK_K, 0))

    z_blk = tl.make_block_ptr(z_ptr, (M, N), (N, 1), (pid_m * BLOCK_M, pid_n * BLOCK_N), (BLOCK_M, BLOCK_N), (1, 0))
    tl.store(z_blk, z.to(z_ptr.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def _relu_epilogue(tile):
    return tl.maximum(tile, 0)

M, N, K = 16 * 8192, 4096, 2048
a = torch.randn(M, K, dtype=torch.float16, device="cuda")
b = torch.randn(K, N, dtype=torch.float16, device="cuda")
z = torch.randn(M, N, dtype=torch.float16, device="cuda")

# warmup
torch.set_float32_matmul_precision("medium")
z_torch = (a @ b).relu()
torch.cuda.synchronize()
M, N, K = z.shape[0], z.shape[1], a.shape[1]
BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M = 128, 128, 32, 8
grid = (GROUP_M, triton.cdiv(N, BLOCK_N), triton.cdiv(M, BLOCK_M) // GROUP_M,)
# grid = (triton.cdiv(N, BLOCK_N) * triton.cdiv(M, BLOCK_M),)
_matmul_kernel[grid](z, a, b, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, _relu_epilogue, num_warps=4, num_stages=3)
torch.cuda.synchronize()

start = time.time()

M, N, K = z.shape[0], z.shape[1], a.shape[1]
# BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M = 64, 64, 64, 2
grid = (GROUP_M, triton.cdiv(N, BLOCK_N), triton.cdiv(M, BLOCK_M) // GROUP_M,)
# grid = (triton.cdiv(N, BLOCK_N) * triton.cdiv(M, BLOCK_M),)
_matmul_kernel[grid](z, a, b, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, _relu_epilogue, num_warps=4, num_stages=3)

torch.cuda.synchronize()

end = time.time()
print("custom", end - start)

start = time.time()

z_torch = (a @ b).relu()
torch.cuda.synchronize()

end = time.time()
print("torch", end - start)

print((z - z_torch).abs().max())
print(z.abs().max())
print(z_torch.abs().max())
print(z.min(), z.max())
