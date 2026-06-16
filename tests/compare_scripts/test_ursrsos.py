import torch
import time
import g4b.device
from g4b.kernels.update_residual_stream import update_residual_stream

# TODO optimize this further... not faster than pytorch

g4b.device.init(0)

import triton.runtime.driver

triton.runtime.driver.active.utils.set_printf_fifo_size(1024 * 1024 * 256)

cuda_sync = lambda: g4b.device.stream.sync()

# TODO test with T = 1
B, T, D = 2, 256, 2560

# TODO test with other dtypes than float32
embed_dtype = torch.float32
# embed_dtype = torch.float16
# embed_dtype = torch.int8

residual = torch.randn(B, T, D, dtype=embed_dtype, device="cuda") * D ** -.5
act_buf = torch.randn(B, T, D, dtype=embed_dtype, device="cuda")
act_rsos = torch.randn(B, T, dtype=embed_dtype, device="cuda").abs()
out_rsos = torch.randn_like(act_rsos)
current_rmsnorm_w = torch.randn(D, dtype=embed_dtype, device="cuda") / 10
next_layer_input_rmsnorm_w = torch.randn(D, dtype=embed_dtype, device="cuda") / 10
layer_output_scale = torch.tensor([0.7], dtype=embed_dtype, device="cuda")
eps = 1e-6

residual_orig = residual.clone()
act_buf_orig = act_buf.clone()
act_rsos_orig = act_rsos.clone()


def restore_inputs():
    global residual, act_buf, act_rsos
    residual = residual_orig.clone()
    act_buf = act_buf_orig.clone()
    act_rsos = act_rsos_orig.clone()
    torch.cuda.synchronize()


def do_forward():
    update_residual_stream(
        residual,
        act_buf,
        act_rsos,
        out_rsos,
        current_rmsnorm_w,
        next_layer_input_rmsnorm_w,
        eps,
        layer_output_scale,
    )


def do_forward_torch():
    global residual, act_buf, act_rsos, out_rsos
    inv_rms = (act_rsos / residual.shape[-1] + eps).rsqrt().unsqueeze(-1)
    residual += act_buf * inv_rms * current_rmsnorm_w
    residual *= layer_output_scale
    act_buf = residual * next_layer_input_rmsnorm_w
    out_rsos = (residual * residual).sum(-1)


def capture_out():
    return residual, act_buf, act_rsos, out_rsos


N_reps = 1

# warmup
torch.set_float32_matmul_precision("medium")
torch.cuda.synchronize()
restore_inputs()
for _ in range(N_reps): do_forward()
cuda_sync()
restore_inputs()
for _ in range(N_reps): do_forward_torch()
torch.cuda.synchronize()

# measure custom
restore_inputs()
start = time.time()
for _ in range(N_reps): do_forward()
out_custom = capture_out()
cuda_sync()
end = time.time()
print("custom", end - start)

# measure torch
restore_inputs()
start = time.time()
for _ in range(N_reps): do_forward_torch()
out_torch = capture_out()
torch.cuda.synchronize()
end = time.time()
print("torch", end - start)

print("abs max input resid", residual_orig.abs().max())
for name, x_custom, x_torch in zip(("residual", "act_buf", "act_rsos", "out_rsos"), out_custom, out_torch):
    print(name)
    print("  abs diff", (x_custom - x_torch).abs().max())
    print("  custom abs max", x_custom.abs().max())
    print("  torch abs max", x_torch.abs().max())
    print("  custom min/max", x_custom.min(), x_custom.max())




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


for name, x_custom, x_torch in zip(("residual", "act_buf", "act_rsos", "out_rsos"), out_custom, out_torch):
    import math
    plot_heatmap(
        (x_custom - x_torch).abs().reshape((math.prod(x_custom.shape[:-1]), x_custom.shape[-1])),
        title=name,
        xlabel="hidden",
        ylabel="BT",
        cmap="viridis",
    )

g4b.device.teardown()
