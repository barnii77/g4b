import time
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

import g4b.device
import g4b.tensor
from g4b.kernels.rmsnorm import finish_rmsnorm_inplace, rmsnorm_x_4d_write_to_y

g4b.device.init(0)

cuda_sync = lambda: g4b.device.stream.sync()

# Shape configs to try:
#B, H, T, D = 1, 1, 256, 2560
#B, H, T, D = 1, 1, 1, 2560
#B, H, T, D = 4, 48, 1, 512
#B, H, T, D = 1, 1, 1024, 2560
B, H, T, D = 1, 1, 16384, 16384

N_REPS = 100
epsilon = 1e-6

embed_dtype = torch.float32
# embed_dtype = torch.float16
# embed_dtype = torch.bfloat16

x = torch.randn(B, H, T, D, dtype=embed_dtype, device="cuda") * 3
y = torch.empty_like(x)
x_rsos = (x * x).sum(-1)
x_rmsnorm_w = torch.randn(D, dtype=embed_dtype, device="cuda")

x_orig = x.clone()
y_orig = y.clone()

_is_first_invocation = True


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


def max_pool_heatmap(x: torch.Tensor, kernel_size: int = 256) -> torch.Tensor:
    kh = min(kernel_size, x.shape[0])
    kw = min(kernel_size, x.shape[1])
    return F.max_pool2d(x.reshape((1, 1, *x.shape)), kernel_size=(kh, kw), stride=(kh, kw)).reshape(
        (x.shape[0] // kh, x.shape[1] // kw)
    )


def min_pool_heatmap(x: torch.Tensor, kernel_size: int = 16) -> torch.Tensor:
    return -max_pool_heatmap(-x, kernel_size)


def restore_inputs():
    global x, y, x_rsos
    x = x_orig.clone()
    y = y_orig.clone()
    x_rsos = (x * x).sum(-1)
    torch.cuda.synchronize()


def do_write_to_y():
    rmsnorm_x_4d_write_to_y(y, x, g4b.tensor.float32)


def do_write_to_y_torch():
    global y
    y = torch.nn.functional.rms_norm(x, (x.shape[-1],))


def do_finish_inplace():
    global _is_first_invocation
    k = finish_rmsnorm_inplace(x, x_rsos, x_rmsnorm_w, epsilon)
    # if _is_first_invocation:
    #     with open("/home/david/Projects/g4b/tmp/source.txt", "w") as f:
    #         f.write(k.asm["source"])
    #     with open("/home/david/Projects/g4b/tmp/cubin.bin", "wb") as f:
    #         f.write(k.asm["cubin"])
    _is_first_invocation = False



def do_finish_inplace_torch():
    global x
    x = x / (x_rsos.unsqueeze(-1) / x.shape[-1] + epsilon).sqrt() * x_rmsnorm_w


def measure(name, custom_fn, torch_fn):
    # warmup
    torch.cuda.synchronize()
    restore_inputs()
    for _ in range(N_REPS):
        custom_fn()
    cuda_sync()
    restore_inputs()
    for _ in range(N_REPS):
        torch_fn()
    torch.cuda.synchronize()

    # measure custom
    restore_inputs()
    start = time.time()
    for _ in range(N_REPS):
        custom_fn()
    cuda_sync()
    end = time.time()
    print(name, "custom", end - start)

    # measure torch
    restore_inputs()
    start = time.time()
    for _ in range(N_REPS):
        torch_fn()
    torch.cuda.synchronize()
    end = time.time()
    print(name, "torch", end - start)

    # capture single-step outputs so repeated in-place application does not affect correctness numbers.
    restore_inputs()
    custom_fn()
    cuda_sync()
    out_custom = x.clone() if name == "finish_rmsnorm_inplace" else y.clone()
    restore_inputs()
    torch_fn()
    torch.cuda.synchronize()
    out_torch = x.clone() if name == "finish_rmsnorm_inplace" else y.clone()

    print(name)
    print("  abs diff", (out_custom - out_torch).abs().max())
    print("  custom abs max", out_custom.abs().max())
    print("  torch abs max", out_torch.abs().max())
    print("  custom min/max", out_custom.min(), out_custom.max())
    print("  rsos min", x_rsos.min())
    abs_diff = (out_custom - out_torch).abs().reshape((B * H * T, D))
    argmax_flat = abs_diff.argmax()
    argmax_row = argmax_flat // D
    argmax_d = argmax_flat % D
    argmax_b = argmax_row // (H * T)
    argmax_ht = argmax_row % (H * T)
    argmax_h = argmax_ht // T
    argmax_t = argmax_ht % T
    print("  max diff index", (argmax_b, argmax_h, argmax_t, argmax_d))
    print("  max diff", abs_diff[argmax_row, argmax_d])
    print("  x at max diff", x_orig[argmax_b, argmax_h, argmax_t, argmax_d])
    print("  rsos at max diff", (x_orig * x_orig).sum(-1)[argmax_b, argmax_h, argmax_t])
    print("  w at max diff", x_rmsnorm_w[argmax_d])
    print("  custom output at max diff", out_custom.reshape((B, H, T, D))[argmax_b, argmax_h, argmax_t, argmax_d])
    print("  torch output at max diff", out_torch.reshape((B, H, T, D))[argmax_b, argmax_h, argmax_t, argmax_d])
    plot_heatmap(
        abs_diff,
        title=name + " abs diff",
        xlabel="D",
        ylabel="BHT",
        cmap="viridis",
    )
    plot_heatmap(
        max_pool_heatmap(abs_diff),
        title=name + " 256x256 max-pooled abs diff",
        xlabel="D / 16",
        ylabel="BHT / 16",
        cmap="viridis",
    )
    plot_heatmap(
        min_pool_heatmap(x_rsos.reshape((B * H, T))),
        title=name + " 256x256 min-pooled rsos",
        xlabel="T / 16",
        ylabel="BH / 16",
        cmap="viridis",
    )


torch.set_float32_matmul_precision("medium")

measure("rmsnorm_x_4d_write_to_y", do_write_to_y, do_write_to_y_torch)
measure("finish_rmsnorm_inplace", do_finish_inplace, do_finish_inplace_torch)

g4b.device.teardown()
