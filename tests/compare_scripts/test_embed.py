import torch
import torch.nn.functional as F
import time
import g4b.device
from g4b.kernels.embeddings import gather_token_embeddings

g4b.device.init(0)

import triton.runtime.driver

triton.runtime.driver.active.utils.set_printf_fifo_size(1024 * 1024 * 256)

cuda_sync = lambda: g4b.device.stream.sync()

# TODO test with T = 1
B, T = 2, 256
D = 2560
V = 1024

# TODO test with other dtypes than float32
embed_dtype = torch.float32
# embed_dtype = torch.float16
# embed_dtype = torch.int8

token_ids = torch.randint(0, V, (T, B), dtype=torch.int32, device="cuda")
embeddings = torch.randn(V, D, dtype=embed_dtype, device="cuda")
output = torch.randn(B, T, D, dtype=embed_dtype, device="cuda")
output_rsos = torch.randn(B, T, dtype=embed_dtype, device="cuda")
scaling_factor = 1.0

output_orig = output.clone()
output_rsos_orig = output_rsos.clone()


def do_forward():
    gather_token_embeddings(output, output_rsos, embeddings, token_ids, scaling_factor)


def do_forward_torch():
    gathered = scaling_factor * embeddings[token_ids.T]
    return gathered, (gathered * gathered).sum(-1)


def restore_inputs():
    global output, output_rsos
    output = output_orig.clone()
    output_rsos = output_rsos_orig.clone()
    torch.cuda.synchronize()


def capture_out():
    return output, output_rsos


# warmup
torch.set_float32_matmul_precision("medium")
torch.cuda.synchronize()
restore_inputs()
do_forward()
cuda_sync()
restore_inputs()
do_forward_torch()
torch.cuda.synchronize()

# measure custom
restore_inputs()
start = time.time()
do_forward()
output_custom, output_rsos_custom = capture_out()
cuda_sync()
end = time.time()
print("custom", end - start)

# measure torch
restore_inputs()
start = time.time()
output_torch, output_rsos_torch = do_forward_torch()
torch.cuda.synchronize()
end = time.time()
print("torch", end - start)

print("abs diff", (output_custom - output_torch).abs().max())
print("rsos diff", (output_rsos_custom - output_rsos_torch).abs().max())
print("custom abs max", output_custom.abs().max())
print("torch abs max", output_torch.abs().max())
print("custom rsos abs max", output_rsos_custom.abs().max())
print("torch rsos abs max", output_rsos_torch.abs().max())
print("custom min/max", output_custom.min(), output_custom.max())




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
    (output_custom - output_torch).abs().reshape((B * T, D)),
    title="Embedding abs diff",
    xlabel="Embedding dim",
    ylabel="Token row",
    cmap="viridis",
)

plot_heatmap(
    F.max_pool2d(
        (output_custom - output_torch).abs().reshape((1, 1, B * T, D)),
        kernel_size=16,
        stride=16,
    ).squeeze(0).squeeze(0),
    title="Embedding 16x16 max-pooled abs diff",
    xlabel="Embedding dim / 16",
    ylabel="Token row / 16",
    cmap="viridis",
)

plot_heatmap(
    (output_rsos_custom - output_rsos_torch).abs(),
    title="Embedding rsos abs diff",
    xlabel="Token position",
    ylabel="Batch",
    cmap="viridis",
)

g4b.device.teardown()
