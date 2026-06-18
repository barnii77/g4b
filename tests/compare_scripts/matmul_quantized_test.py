import torch
import triton.runtime.driver
import time
import sys
from pathlib import Path
from typing import Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import g4b.device
from g4b import tensor as g4b_tensor
from g4b.gguf import GGUFType, GGUFTensor
from g4b.kernels.matmul import matmul_a3d_b2d
from g4b.kernels.fa2 import PHASE_PREFILL
from g4b.tensor import Tensor
from scripts.reference_impl import dequant_q4k_to_fp32, dequant_q5k_to_fp32, dequant_q6k_to_fp32


g4b.device.init(0)
triton.runtime.driver.active.utils.set_printf_fifo_size(1024 * 1024 * 256)
cuda_sync = lambda: g4b.device.stream.sync()


def plot_heatmap(
    x: torch.Tensor,
    *,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    xticklabels: Optional[Sequence[str]] = None,
    yticklabels: Optional[Sequence[str]] = None,
    annotate: bool = False,
    fmt: str = ".2f",
    colorbar: bool = True,
    cmap: Optional[str] = None,
    save_path: Optional[str | Path] = None,
    show: bool = True,
):
    import matplotlib.pyplot as plt

    if not isinstance(x, torch.Tensor):
        raise TypeError(f"x must be a torch.Tensor, got {type(x).__name__}")
    if x.ndim != 2:
        raise ValueError(f"x must be 2D, got shape {tuple(x.shape)}")

    data = x.detach().float().cpu().numpy()
    rows, cols = data.shape

    fig, ax = plt.subplots()
    im = ax.imshow(data, aspect="auto", cmap=cmap)

    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if xticklabels is not None:
        if len(xticklabels) != cols:
            raise ValueError(f"xticklabels has length {len(xticklabels)}, expected {cols}")
        ax.set_xticks(range(cols))
        ax.set_xticklabels(xticklabels, rotation=45, ha="right")

    if yticklabels is not None:
        if len(yticklabels) != rows:
            raise ValueError(f"yticklabels has length {len(yticklabels)}, expected {rows}")
        ax.set_yticks(range(rows))
        ax.set_yticklabels(yticklabels)

    if annotate:
        for i in range(rows):
            for j in range(cols):
                ax.text(j, i, format(data[i, j], fmt), ha="center", va="center")

    if colorbar:
        fig.colorbar(im, ax=ax)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=200)

    if show:
        plt.show()

    return fig, ax


def make_random_q4k_tensor(rows: int, cols: int) -> GGUFTensor:
    assert cols % 256 == 0
    n_blocks = rows * (cols // 256)
    raw = torch.empty((n_blocks, 144), dtype=torch.uint8)

    dd = (torch.rand((n_blocks,), dtype=torch.float16) * 0.25 + 0.01).view(torch.uint8).reshape(n_blocks, 2)
    md = (torch.rand((n_blocks,), dtype=torch.float16) * 0.25).view(torch.uint8).reshape(n_blocks, 2)
    raw[:, 0:2] = dd
    raw[:, 2:4] = md
    raw[:, 4:16] = torch.randint(0, 256, (n_blocks, 12), dtype=torch.uint8)
    raw[:, 16:] = torch.randint(0, 256, (n_blocks, 128), dtype=torch.uint8)

    return GGUFTensor("synthetic.q4_k", [cols, rows], GGUFType.GGML_TYPE_Q4_K, raw.numpy().tobytes())


def make_random_q5k_tensor(rows: int, cols: int) -> GGUFTensor:
    assert cols % 256 == 0
    n_blocks = rows * (cols // 256)
    raw = torch.empty((n_blocks, 176), dtype=torch.uint8)

    dd = (torch.rand((n_blocks,), dtype=torch.float16) * 0.25 + 0.01).view(torch.uint8).reshape(n_blocks, 2)
    md = (torch.rand((n_blocks,), dtype=torch.float16) * 0.25).view(torch.uint8).reshape(n_blocks, 2)
    raw[:, 0:2] = dd
    raw[:, 2:4] = md
    raw[:, 4:16] = torch.randint(0, 256, (n_blocks, 12), dtype=torch.uint8)
    raw[:, 16:48] = torch.randint(0, 256, (n_blocks, 32), dtype=torch.uint8)
    raw[:, 48:] = torch.randint(0, 256, (n_blocks, 128), dtype=torch.uint8)

    return GGUFTensor("synthetic.q5_k", [cols, rows], GGUFType.GGML_TYPE_Q5_K, raw.numpy().tobytes())


def make_random_q6k_tensor(rows: int, cols: int) -> GGUFTensor:
    assert cols % 256 == 0
    n_blocks = rows * (cols // 256)
    raw = torch.empty((n_blocks, 210), dtype=torch.uint8)

    raw[:, 0:128] = torch.randint(0, 256, (n_blocks, 128), dtype=torch.uint8)
    raw[:, 128:192] = torch.randint(0, 256, (n_blocks, 64), dtype=torch.uint8)
    raw[:, 192:208] = torch.randint(-128, 128, (n_blocks, 16), dtype=torch.int8).view(torch.uint8)
    dd = (torch.rand((n_blocks,), dtype=torch.float16) * 0.25 + 0.01).view(torch.uint8).reshape(n_blocks, 2)
    raw[:, 208:210] = dd

    return GGUFTensor("synthetic.q6_k", [cols, rows], GGUFType.GGML_TYPE_Q6_K, raw.numpy().tobytes())


def _pack_q4k_scales_and_mins(sc: torch.Tensor, mins: torch.Tensor) -> torch.Tensor:
    packed = torch.empty((sc.shape[0], 12), dtype=torch.uint8)
    packed[:, 0:4] = (sc[:, 0:4] & 0x3F) | ((sc[:, 4:8] & 0x30) << 2)
    packed[:, 4:8] = (mins[:, 0:4] & 0x3F) | ((mins[:, 4:8] & 0x30) << 2)
    packed[:, 8:12] = (sc[:, 4:8] & 0x0F) | ((mins[:, 4:8] & 0x0F) << 4)
    return packed


def _pack_q4k_qs(qs: torch.Tensor) -> torch.Tensor:
    low = qs[:, 0::2, :]
    high = qs[:, 1::2, :]
    return (low | (high << 4)).reshape((qs.shape[0], 128))


def _pack_q5k_qh(qs: torch.Tensor) -> torch.Tensor:
    high_bits = (qs >> 4) & 0x01
    bit_shifts = torch.arange(8, dtype=torch.uint8).reshape((1, 8, 1))
    return ((high_bits << bit_shifts).sum(dim=1) & 0xFF).to(torch.uint8)


def _pack_q6k_ql(qs: torch.Tensor) -> torch.Tensor:
    low = (qs & 0x0F).reshape((qs.shape[0], 2, 2, 64))
    return (low[:, :, 0, :] | (low[:, :, 1, :] << 4)).reshape((qs.shape[0], 128))


def _pack_q6k_qh(qs: torch.Tensor) -> torch.Tensor:
    high = ((qs >> 4) & 0x03).reshape((qs.shape[0], 2, 4, 32))
    return (
        high[:, :, 0, :]
        | (high[:, :, 1, :] << 2)
        | (high[:, :, 2, :] << 4)
        | (high[:, :, 3, :] << 6)
    ).reshape((qs.shape[0], 64))


def make_patterned_q4k_tensor(rows: int, cols: int) -> GGUFTensor:
    assert cols % 256 == 0
    n_col_blocks = cols // 256
    n_blocks = rows * n_col_blocks
    block_row = torch.arange(n_blocks, dtype=torch.int64) // n_col_blocks
    block_col = torch.arange(n_blocks, dtype=torch.int64) % n_col_blocks
    subblock = torch.arange(8, dtype=torch.uint8).reshape((1, 8))
    lane = torch.arange(32, dtype=torch.uint8).reshape((1, 1, 32))
    raw = torch.empty((n_blocks, 144), dtype=torch.uint8)

    dd = torch.ones((n_blocks,), dtype=torch.float16)
    md = torch.ones((n_blocks,), dtype=torch.float16)
    raw[:, 0:2] = dd.view(torch.uint8).reshape((n_blocks, 2))
    raw[:, 2:4] = md.view(torch.uint8).reshape((n_blocks, 2))
    sc = (1 + subblock * 3 + (block_col.to(torch.uint8).reshape((n_blocks, 1)) % 3)).to(torch.uint8)
    mins = (2 + subblock * 5 + (block_row.to(torch.uint8).reshape((n_blocks, 1)) % 5)).to(torch.uint8)
    raw[:, 4:16] = _pack_q4k_scales_and_mins(sc, mins)
    qs = ((subblock.reshape((1, 8, 1)) * 3 + lane + (lane // 16) * 5) % 16).to(torch.uint8).expand(
        (n_blocks, 8, 32)
    )
    raw[:, 16:] = _pack_q4k_qs(qs)
    return GGUFTensor("synthetic.patterned.q4_k", [cols, rows], GGUFType.GGML_TYPE_Q4_K, raw.numpy().tobytes())


def make_patterned_q5k_tensor(rows: int, cols: int) -> GGUFTensor:
    assert cols % 256 == 0
    n_col_blocks = cols // 256
    n_blocks = rows * n_col_blocks
    block_row = torch.arange(n_blocks, dtype=torch.int64) // n_col_blocks
    block_col = torch.arange(n_blocks, dtype=torch.int64) % n_col_blocks
    subblock = torch.arange(8, dtype=torch.uint8).reshape((1, 8))
    lane = torch.arange(32, dtype=torch.uint8).reshape((1, 1, 32))
    raw = torch.empty((n_blocks, 176), dtype=torch.uint8)

    dd = torch.ones((n_blocks,), dtype=torch.float16)
    md = torch.ones((n_blocks,), dtype=torch.float16)
    raw[:, 0:2] = dd.view(torch.uint8).reshape((n_blocks, 2))
    raw[:, 2:4] = md.view(torch.uint8).reshape((n_blocks, 2))
    sc = (3 + subblock * 4 + (block_col.to(torch.uint8).reshape((n_blocks, 1)) % 5)).to(torch.uint8)
    mins = (1 + subblock * 6 + (block_row.to(torch.uint8).reshape((n_blocks, 1)) % 7)).to(torch.uint8)
    raw[:, 4:16] = _pack_q4k_scales_and_mins(sc, mins)
    qs = ((subblock.reshape((1, 8, 1)) * 5 + lane + (lane // 16) * 9) % 32).to(torch.uint8).expand(
        (n_blocks, 8, 32)
    )
    raw[:, 16:48] = _pack_q5k_qh(qs)
    raw[:, 48:] = _pack_q4k_qs(qs & 0x0F)
    return GGUFTensor("synthetic.patterned.q5_k", [cols, rows], GGUFType.GGML_TYPE_Q5_K, raw.numpy().tobytes())


def make_patterned_q6k_tensor(rows: int, cols: int) -> GGUFTensor:
    assert cols % 256 == 0
    n_col_blocks = cols // 256
    n_blocks = rows * n_col_blocks
    block_row = torch.arange(n_blocks, dtype=torch.int64) // n_col_blocks
    block_col = torch.arange(n_blocks, dtype=torch.int64) % n_col_blocks
    subblock = torch.arange(16, dtype=torch.uint8).reshape((1, 16))
    lane = torch.arange(16, dtype=torch.uint8).reshape((1, 1, 16))
    raw = torch.empty((n_blocks, 210), dtype=torch.uint8)

    dd = torch.ones((n_blocks,), dtype=torch.float16)
    sc = (torch.arange(16, dtype=torch.int16).reshape((1, 16)) * 7 - 48).expand((n_blocks, 16)).clone()
    sc += (block_col.reshape((n_blocks, 1)) % 5).to(torch.int16)
    sc -= (block_row.reshape((n_blocks, 1)) % 3).to(torch.int16)
    sc = sc.clamp(-128, 127).to(torch.int8)
    qs = ((subblock.reshape((1, 16, 1)) * 5 + lane + (lane // 8) * 13) % 64).to(torch.uint8).expand(
        (n_blocks, 16, 16)
    )
    raw[:, 0:128] = _pack_q6k_ql(qs)
    raw[:, 128:192] = _pack_q6k_qh(qs)
    raw[:, 192:208] = sc.view(torch.uint8)
    raw[:, 208:210] = dd.view(torch.uint8).reshape((n_blocks, 2))
    return GGUFTensor("synthetic.patterned.q6_k", [cols, rows], GGUFType.GGML_TYPE_Q6_K, raw.numpy().tobytes())


QUANT_CASES = {
    "q4_k": {
        "dtype": g4b_tensor.q4_k,
        "random": make_random_q4k_tensor,
        "patterned": make_patterned_q4k_tensor,
        "dequant": dequant_q4k_to_fp32,
    },
    "q5_k": {
        "dtype": g4b_tensor.q5_k,
        "random": make_random_q5k_tensor,
        "patterned": make_patterned_q5k_tensor,
        "dequant": dequant_q5k_to_fp32,
    },
    "q6_k": {
        "dtype": g4b_tensor.q6_k,
        "random": make_random_q6k_tensor,
        "patterned": make_patterned_q6k_tensor,
        "dequant": dequant_q6k_to_fp32,
    },
}

SHAPE_CASES = [
    # B, M, K, N. N must be a multiple of the quant superblock width.
    (2, 32, 256, 256),
    (4, 128, 256, 512),
    (2, 32, 8192, 8192),
    (4, 128, 8192, 8192),
]

N_REPS = 100


def run_case(quant_name: str, input_name: str, transpose: bool, B: int, M: int, K: int, N: int):
    quant = QUANT_CASES[quant_name]
    torch.manual_seed(0)
    a = torch.randn((B, M, K), dtype=torch.float32, device="cuda") / K**0.5
    c = torch.empty((B, M, N), dtype=torch.float32, device="cuda")

    b_shape = (N, K) if transpose else (K, N)
    gguf_tensor = quant[input_name](*b_shape)
    b = Tensor.from_gguf_tensor(gguf_tensor)
    b_ref = quant["dequant"](gguf_tensor).reshape(b_shape).to("cuda")
    b_ref = b_ref.T if transpose else b_ref

    torch.set_float32_matmul_precision("medium")

    # Warmup also pays compilation/autotune cost before measuring correctness.
    matmul_a3d_b2d(c, None, a, b, transpose_b_before_mma=transpose, rmsnorm_eps=0.0)
    cuda_sync()
    torch.cuda.synchronize()
    _ = a @ b_ref
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(N_REPS):
        matmul_a3d_b2d(c, None, a, b, transpose_b_before_mma=transpose, rmsnorm_eps=0.0)
    cuda_sync()
    custom_seconds = time.time() - start

    start = time.time()
    for _ in range(N_REPS):
        ref = a @ b_ref
    torch.cuda.synchronize()
    torch_medium_seconds = time.time() - start

    # Capture one post-warmup output for correctness.
    matmul_a3d_b2d(c, None, a, b, transpose_b_before_mma=transpose, rmsnorm_eps=0.0)
    cuda_sync()
    torch.cuda.synchronize()

    ref_fp16_b = a @ b_ref.to(torch.float16).to(torch.float32)
    torch.cuda.synchronize()

    diff = (c - ref).abs()
    diff_fp16_b = (c - ref_fp16_b).abs()
    return {
        "quant": quant_name,
        "input": input_name,
        "transpose": transpose,
        "shape": (B, M, K, N),
        "max_diff": diff.max().item(),
        "mean_diff": diff.mean().item(),
        "max_diff_vs_fp16_b_ref": diff_fp16_b.max().item(),
        "mean_diff_vs_fp16_b_ref": diff_fp16_b.mean().item(),
        "expected_abs_max": ref.abs().max().item(),
        "real_abs_max": c.abs().max().item(),
        "expected_std": ref.float().std(unbiased=False).item(),
        "real_std": c.float().std(unbiased=False).item(),
        "custom_seconds": custom_seconds,
        "torch_medium_seconds": torch_medium_seconds,
        "diff": diff,
    }


results = []
for quant_name in QUANT_CASES:
    for input_name in ("patterned", "random"):
        for shape in SHAPE_CASES:
            for transpose in (False, True):
                results.append(run_case(quant_name, input_name, transpose, *shape))

print()
print("quantized matmul test report")
for result in results:
    B, M, K, N = result["shape"]
    print(
        f"- quant={result['quant']} input={result['input']} transpose_b_before_mma={result['transpose']} "
        f"B={B} M={M} K={K} N={N} "
        f"max_diff={result['max_diff']:.8g} mean_diff={result['mean_diff']:.8g} "
        f"max_diff_vs_fp16_b_ref={result.get('max_diff_vs_fp16_b_ref', float('nan')):.8g} "
        f"mean_diff_vs_fp16_b_ref={result.get('mean_diff_vs_fp16_b_ref', float('nan')):.8g} "
        f"expected_abs_max={result['expected_abs_max']:.8g} real_abs_max={result['real_abs_max']:.8g} "
        f"expected_std={result['expected_std']:.8g} real_std={result['real_std']:.8g} "
        f"custom={result['custom_seconds']:.6f}s torch_medium={result['torch_medium_seconds']:.6f}s"
    )

failed = [result for result in results if result["max_diff"] != 0.0]
if failed:
    worst = max(failed, key=lambda result: result["max_diff"])
    B, M, K, N = worst["shape"]
    diff = worst["diff"]
    print()
    print(
        f"plotting worst case quant={worst['quant']} input={worst['input']} "
        f"B={B} M={M} K={K} N={N} max_diff={worst['max_diff']:.8g}"
    )

    flat_diff = diff.reshape((B * M, N))
    plot_heatmap(
        flat_diff,
        title=f"{worst['quant']} {worst['input']} matmul abs diff",
        xlabel="N",
        ylabel="B*M",
        cmap="viridis",
    )

    plot_heatmap(
        diff.max(-1).values,
        title=f"{worst['quant']} {worst['input']} max abs diff by batch/M row",
        xlabel="M",
        ylabel="B",
        yticklabels=[str(i) for i in range(B)],
        cmap="viridis",
    )

    n_group = 32
    if N % n_group == 0:
        n_group_diff = diff.reshape((B, M, N // n_group, n_group)).max(-1).values
        plot_heatmap(
            n_group_diff.reshape((B * M, N // n_group)),
            title=f"{worst['quant']} {worst['input']} max abs diff by {n_group}-wide N group",
            xlabel=f"N groups x{n_group}",
            ylabel="B*M",
            cmap="viridis",
        )
else:
    print("all matmul cases had zero diff; no heatmaps generated")


# ---------------------------------------------------------------------------------------------------------------------
# Phase-skip benchmark: with ~3/4 of the batch slots "unallocated" (phase 0, skipped by the kernel), how does the
# in-kernel per-slot skip compare to (a) the full matmul over all slots and (b) compacting the active rows into a fresh
# contiguous tensor and doing a dense matmul on just those? Measured for both my kernel and torch.
#
# dim0 (B) is the batch/slot dim that the phase buffer indexes. The compacted variants are timed on a *pre-gathered*
# tensor (so each timed loop is single-stream: g4b kernels never read a tensor torch is still writing) and the gather
# cost is measured separately, so "compact total ~= gather + compact matmul".
# ---------------------------------------------------------------------------------------------------------------------

PHASE_SKIP_FRAC_UNALLOCATED = 0.75
PHASE_SKIP_QUANT = "q4_k"
PHASE_SKIP_SHAPE_CASES = [
    # B (slots), M (tokens/slot), K, N. N a multiple of the superblock width.
    (64, 1, 2048, 8192),  # decode-like: one token per slot
    (64, 16, 2048, 8192),  # small-prefill-like: a few tokens per slot
    (64, 512, 2048, 8192),  # big-prefill-like: many tokens per slot
]
PHASE_SKIP_REPS = 100


def _bench_g4b(fn, reps=PHASE_SKIP_REPS):
    for _ in range(5):
        fn()
    cuda_sync()
    start = time.time()
    for _ in range(reps):
        fn()
    cuda_sync()
    return (time.time() - start) / reps * 1e3  # ms/iter


def _bench_torch(fn, reps=PHASE_SKIP_REPS):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.time() - start) / reps * 1e3  # ms/iter


def run_phase_skip_case(quant_name: str, B: int, M: int, K: int, N: int, transpose: bool = True):
    quant = QUANT_CASES[quant_name]
    torch.manual_seed(0)
    a = torch.randn((B, M, K), dtype=torch.float32, device="cuda") / K**0.5
    c = torch.empty((B, M, N), dtype=torch.float32, device="cuda")

    b_shape = (N, K) if transpose else (K, N)
    gguf_tensor = quant["random"](*b_shape)
    b = Tensor.from_gguf_tensor(gguf_tensor)
    b_ref = quant["dequant"](gguf_tensor).reshape(b_shape).to("cuda")
    b_ref = b_ref.T if transpose else b_ref

    # random per-slot phase: ~PHASE_SKIP_FRAC_UNALLOCATED of slots are unallocated (0), the rest are active (prefill).
    active_mask = torch.rand((B,), device="cuda") >= PHASE_SKIP_FRAC_UNALLOCATED
    active_mask[0] = True  # guarantee at least one active slot
    user_phase = torch.where(active_mask, PHASE_PREFILL, 0).to(torch.uint8)
    active_idx = active_mask.nonzero(as_tuple=True)[0].to(torch.int32)
    n_active = int(active_idx.numel())

    # pre-gathered compact activation (gather cost measured separately, below).
    a_compact = a.index_select(0, active_idx.long()).contiguous()
    c_compact = torch.empty((n_active, M, N), dtype=torch.float32, device="cuda")

    torch.set_float32_matmul_precision("medium")

    mine_full = lambda: matmul_a3d_b2d(c, None, a, b, transpose_b_before_mma=transpose, rmsnorm_eps=0.0)
    mine_phase = lambda: matmul_a3d_b2d(
        c, None, a, b, transpose_b_before_mma=transpose, user_phase=user_phase, phase=PHASE_PREFILL, rmsnorm_eps=0.0
    )
    mine_compact = lambda: matmul_a3d_b2d(
        c_compact, None, a_compact, b, transpose_b_before_mma=transpose, rmsnorm_eps=0.0
    )
    torch_full = lambda: a @ b_ref
    torch_compact = lambda: a_compact @ b_ref
    gather = lambda: a.index_select(0, active_idx.long()).contiguous()

    ms = {
        "mine_full": _bench_g4b(mine_full),
        "mine_phase": _bench_g4b(mine_phase),
        "mine_compact": _bench_g4b(mine_compact),
        "torch_full": _bench_torch(torch_full),
        "torch_compact": _bench_torch(torch_compact),
        "gather": _bench_torch(gather),
    }
    return {"quant": quant_name, "shape": (B, M, K, N), "n_active": n_active, "ms": ms}


# warmup
for shape in PHASE_SKIP_SHAPE_CASES:
    run_phase_skip_case(PHASE_SKIP_QUANT, *shape)

print()
print(f"phase-skip benchmark (target {PHASE_SKIP_FRAC_UNALLOCATED:.0%} of slots unallocated, quant={PHASE_SKIP_QUANT})")
for shape in PHASE_SKIP_SHAPE_CASES:
    r = run_phase_skip_case(PHASE_SKIP_QUANT, *shape)
    B, M, K, N = r["shape"]
    ms = r["ms"]
    print(
        f"- B={B} M={M} K={K} N={N} active={r['n_active']}/{B}\n"
        f"    mine:  full={ms['mine_full']:.4f}ms  phase_skip={ms['mine_phase']:.4f}ms  "
        f"compact={ms['mine_compact']:.4f}ms (+gather={ms['gather']:.4f} = {ms['mine_compact']+ms['gather']:.4f}ms)\n"
        f"    torch: full={ms['torch_full']:.4f}ms  "
        f"compact={ms['torch_compact']:.4f}ms (+gather={ms['gather']:.4f} = {ms['torch_compact']+ms['gather']:.4f}ms)\n"
        f"    speedups vs mine_full: phase_skip={ms['mine_full']/ms['mine_phase']:.2f}x  "
        f"compact+gather={ms['mine_full']/(ms['mine_compact']+ms['gather']):.2f}x"
    )

g4b.device.teardown()
