import torch
import triton.runtime.driver
import sys
from pathlib import Path
from typing import Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import g4b.device
from g4b import tensor as g4b_tensor
from g4b.gguf import GGUFType, GGUFTensor
from g4b.kernels.matmul import matmul_a3d_b2d
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

# Inference-shaped cases for Gemma-4-E4B-it-UD-Q4_K_XL:
# D=2560, U=10240, h=8, g=2, qk/v=512, swa_qk/v=256, V=262144.
# All production weight matmuls use GGUF-transposed weights, so these cases keep transpose_b_before_mma=True.
SHAPE_CASES = [
    # name, quant, B, M, K, N, reps
    ("decode_q_proj", "q4_k", 1, 1, 2560, 4096, 40),
    ("decode_kv_proj", "q6_k", 1, 1, 2560, 1024, 40),
    ("decode_o_proj", "q4_k", 1, 1, 4096, 2560, 40),
    ("decode_up_proj", "q4_k", 1, 1, 2560, 10240, 40),
    ("decode_down_proj", "q6_k", 1, 1, 10240, 2560, 40),
    ("decode_lm_head", "q5_k", 1, 1, 2560, 262144, 10),
    ("prefill_q_proj", "q4_k", 1, 512, 2560, 4096, 10),
    ("prefill_up_proj", "q4_k", 1, 512, 2560, 10240, 6),
    ("prefill_down_proj", "q6_k", 1, 512, 10240, 2560, 6),
]

def _bench_g4b_event_ms(fn, reps: int) -> float:
    torch.cuda.synchronize()
    cuda_sync()
    start = g4b.device.event(timing_enabled=True)
    end = g4b.device.event(timing_enabled=True)
    g4b.device.stream.record(start)
    for _ in range(reps):
        fn()
    g4b.device.stream.record(end)
    end.sync()
    return end - start


def _bench_torch_event_ms(fn, reps: int) -> float:
    cuda_sync()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def run_case(name: str, quant_name: str, B: int, M: int, K: int, N: int, reps: int):
    quant = QUANT_CASES[quant_name]
    torch.manual_seed(0)
    a = torch.randn((B, M, K), dtype=torch.float32, device="cuda") / K**0.5
    c = torch.empty((B, M, N), dtype=torch.float32, device="cuda")

    b_shape = (N, K)
    gguf_tensor = quant["random"](*b_shape)
    b = Tensor.from_gguf_tensor(gguf_tensor)
    b_ref = quant["dequant"](gguf_tensor).reshape(b_shape).to("cuda")
    b_ref = b_ref.T

    torch.set_float32_matmul_precision("highest")

    # Warmup also pays compilation/autotune cost before measuring correctness.
    matmul_a3d_b2d(c, None, a, b, transpose_b_before_mma=True, rmsnorm_eps=0.0)
    cuda_sync()
    torch.cuda.synchronize()
    _ = a @ b_ref
    torch.cuda.synchronize()

    custom_ms = _bench_g4b_event_ms(
        lambda: matmul_a3d_b2d(c, None, a, b, transpose_b_before_mma=True, rmsnorm_eps=0.0), reps
    )

    ref = None
    torch_highest_ms = _bench_torch_event_ms(lambda: a @ b_ref, reps)
    ref = a @ b_ref
    torch.cuda.synchronize()

    # Capture one post-warmup output for correctness.
    matmul_a3d_b2d(c, None, a, b, transpose_b_before_mma=True, rmsnorm_eps=0.0)
    cuda_sync()
    torch.cuda.synchronize()

    diff = (c - ref).abs()
    return {
        "name": name,
        "quant": quant_name,
        "shape": (B, M, K, N),
        "reps": reps,
        "max_diff": diff.max().item(),
        "mean_diff": diff.mean().item(),
        "expected_abs_max": ref.abs().max().item(),
        "real_abs_max": c.abs().max().item(),
        "expected_std": ref.float().std(unbiased=False).item(),
        "real_std": c.float().std(unbiased=False).item(),
        "custom_ms": custom_ms,
        "torch_highest_ms": torch_highest_ms,
        "diff": diff,
    }


results = []
for case in SHAPE_CASES:
    results.append(run_case(*case))

print()
print("inference-shaped quantized matmul report")
for result in results:
    B, M, K, N = result["shape"]
    custom_ms = result["custom_ms"] / result["reps"]
    torch_ms = result["torch_highest_ms"] / result["reps"]
    print(
        f"- {result['name']} quant={result['quant']} B={B} M={M} K={K} N={N} reps={result['reps']} "
        f"max_diff={result['max_diff']:.8g} mean_diff={result['mean_diff']:.8g} "
        f"expected_abs_max={result['expected_abs_max']:.8g} real_abs_max={result['real_abs_max']:.8g} "
        f"expected_std={result['expected_std']:.8g} real_std={result['real_std']:.8g} "
        f"custom_cuda_event={custom_ms:.4f}ms "
        f"torch_highest_cuda_event={torch_ms:.4f}ms "
        f"custom/torch={custom_ms / torch_ms:.2f}x"
    )

failed = [result for result in results if result["max_diff"] != 0.0]
if failed:
    worst = max(failed, key=lambda result: result["max_diff"])
    B, M, K, N = worst["shape"]
    print()
    print(
        f"worst diff case={worst['name']} quant={worst['quant']} "
        f"B={B} M={M} K={K} N={N} max_diff={worst['max_diff']:.8g}"
    )
else:
    print("all matmul cases had zero diff")



g4b.device.teardown()
