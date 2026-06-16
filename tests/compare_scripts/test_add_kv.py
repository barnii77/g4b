import torch
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import g4b.device
from g4b.kernels.add_kv_to_cache import add_kv_to_cache

g4b.device.init(0)

import triton.runtime.driver

triton.runtime.driver.active.utils.set_printf_fifo_size(1024 * 1024 * 256)

cuda_sync = lambda: g4b.device.stream.sync()

B, G, t, T, D = 2, 4, 7, 17, 256

# TODO test with other dtypes than float32
embed_dtype = torch.float32
# embed_dtype = torch.float16
# embed_dtype = torch.int8

x = torch.randn(B, G, t, D, dtype=embed_dtype, device="cuda")
x_rsos = (x * x).sum(-1)
cache = torch.randn(B, G, T, D, dtype=embed_dtype, device="cuda")
cache_offsets = torch.tensor([15, 3], dtype=torch.int64, device="cuda")
eps = 1e-6

cache_orig = cache.clone()


def restore_inputs():
    global cache
    cache = cache_orig.clone()
    torch.cuda.synchronize()


def do_forward_plain():
    add_kv_to_cache(x, cache, cache_offsets, eps)


def do_forward_plain_torch():
    global cache
    for b in range(B):
        cache[b, :, (torch.arange(t, device="cuda") + cache_offsets[b]) % T, :] = x[b]


def do_forward_rmsnorm():
    add_kv_to_cache(x, cache, cache_offsets, eps, x_rsos)


def do_forward_rmsnorm_torch():
    global cache
    x_norm = x / (x_rsos.unsqueeze(-1) / D + eps).sqrt()
    for b in range(B):
        cache[b, :, (torch.arange(t, device="cuda") + cache_offsets[b]) % T, :] = x_norm[b]


def capture_out():
    return cache


N_reps = 1000

def measure(name, custom_fn, torch_fn):
    # warmup
    torch.cuda.synchronize()
    restore_inputs()
    for _ in range(N_reps): custom_fn()
    cuda_sync()
    restore_inputs()
    for _ in range(N_reps): torch_fn()
    torch.cuda.synchronize()

    # measure custom
    restore_inputs()
    start = time.time()
    for _ in range(N_reps): custom_fn()
    out_custom = capture_out()
    cuda_sync()
    end = time.time()
    print(name, "custom", end - start)

    # measure torch
    restore_inputs()
    start = time.time()
    for _ in range(N_reps): torch_fn()
    out_torch = capture_out()
    torch.cuda.synchronize()
    end = time.time()
    print(name, "torch", end - start)

    print(name)
    print("  abs diff", (out_custom - out_torch).abs().max())
    print("  custom abs max", out_custom.abs().max())
    print("  torch abs max", out_torch.abs().max())
    print("  custom min/max", out_custom.min(), out_custom.max())
    return out_custom, out_torch


torch.set_float32_matmul_precision("medium")
out_custom, out_torch = measure("plain", do_forward_plain, do_forward_plain_torch)
out_custom, out_torch = measure("rmsnorm", do_forward_rmsnorm, do_forward_rmsnorm_torch)




from pathlib import Path
from typing import Optional, Sequence

import torch
import matplotlib.pyplot as plt


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
    """
    Plot a 2D torch tensor as a heatmap.

    Args:
        x: 2D torch tensor.
        title: Optional plot title.
        xlabel/ylabel: Optional axis labels.
        xticklabels/yticklabels: Optional labels for columns/rows.
        annotate: If True, write values into cells.
        fmt: Format string for annotations.
        colorbar: Whether to show a colorbar.
        cmap: Optional matplotlib colormap name, e.g. "viridis", "magma".
        save_path: Optional path to save the figure.
        show: Whether to call plt.show().

    Returns:
        fig, ax
    """
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


plot_heatmap(
    (out_custom - out_torch).abs().reshape((B * G * T, D)),
    title="KV cache diff",
    xlabel="hidden",
    ylabel="BGT",
    cmap="viridis",
)

g4b.device.teardown()
