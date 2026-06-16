import torch
import time
import g4b.device
from g4b.kernels.matmul import matmul_a3d_b2d
from cuda.core import EventOptions

g4b.device.init(0)

import triton.runtime.driver

triton.runtime.driver.active.utils.set_printf_fifo_size(1024 * 1024 * 256)

# import triton.knobs
# triton.knobs.autotuning.print = True

cuda_sync = lambda: g4b.device.stream.sync()

dt = torch.float16
# B, M, N, K = 16, 4 * 1024, 4 * 1024, 2048
B, M, N, K = 5, 8 * 1024 + 342, 4 * 1024 + 18, 2048 * 10
# B, M, N, K = 4, 8 * 1024, 4 * 1024, 2048 * 10
a = torch.randn(B, M, K, dtype=dt, device="cuda") / 10.0
b = torch.randn(K, N, dtype=dt, device="cuda") / 10.0
z = torch.randn(B, M, N, dtype=dt, device="cuda")


#### BEGIN SLOP

import torch
import triton

stream = g4b.device.stream

# warmup
for _ in range(10):
    matmul_a3d_b2d(z, None, a, b, rmsnorm_eps=0.0)
cuda_sync()

start_t = time.time()
start = g4b.device.device.create_event(options=EventOptions(timing_enabled=True))
end = g4b.device.device.create_event(options=EventOptions(timing_enabled=True))

stream.record(start)

for _ in range(100):
    matmul_a3d_b2d(z, None, a, b, rmsnorm_eps=0.0)

stream.record(end)
end.sync()
end_t = time.time()

print("custom avg ms:", (end - start) / 100)
print("(wall", end_t - start_t, ")")

#### END SLOP


# warmup
torch.set_float32_matmul_precision("high")
z_torch0 = a @ b
z_rsos = (z_torch0**2).sum(-1)
z_rsos_torch = z_rsos.clone()
torch.cuda.synchronize()
matmul_a3d_b2d(z, None, a, b, rmsnorm_eps=0.0)
cuda_sync()
torch.cuda.synchronize()

start = time.time()

matmul_a3d_b2d(z, None, a, b, rmsnorm_eps=0.0)
cuda_sync()
torch.cuda.synchronize()

end = time.time()
print("custom", end - start)

start = time.time()

z_torch = a @ b
torch.cuda.synchronize()

end = time.time()
print("torch", end - start)

print("abs diff", (z - z_torch).abs().max())
print("custom abs max", z.abs().max())
print("torch abs max", z_torch.abs().max())
print("custom min/max", z.min(), z.max())
print("abs diff torch vs torch", (z_torch0 - z_torch).abs().max())
print("abs diff rsos", (z_rsos - z_rsos_torch).abs().max())
print("absmax rsos", z_rsos.abs().max(), "and torch", z_rsos_torch.abs().max())


# NOW TRY WITH keep_c = true
def restore_keep_c_inputs():
    z.zero_()
    z_rsos.zero_()
    torch.cuda.synchronize()


restore_keep_c_inputs()
matmul_a3d_b2d(z, z_rsos, a, b, keep_c=True, rmsnorm_eps=0.0)  # warmup
cuda_sync()

restore_keep_c_inputs()
start = time.time()
matmul_a3d_b2d(z, z_rsos, a, b, keep_c=True, rmsnorm_eps=0.0)
cuda_sync()
end = time.time()
print("with keep_c = true, it took", end - start)
print("max abs diff", (z_torch - z).abs().max())


# NOW TRY THE GEGLU FUSION PAIR MATMUL FEATURE
@triton.jit
def add_tiles(
    c,
    c2,
    off0,
    off1,
    off2,
    NUM_K_SPLITS,
    C_DTYPE,
    input_rsos_ptr,
    input_rsos_shape0,
    input_rsos_shape1,
    input_rsos_stride0,
    input_rsos_stride1,
    rmsnorm_dim,
    rmsnorm_eps,
):
    return c + c2


# warmup
matmul_a3d_b2d(z, z_rsos, a, b, b, c_c2_merge_tiles_fn=add_tiles, rmsnorm_eps=0.0)
cuda_sync()
torch.cuda.synchronize()

# real run
start = time.time()
matmul_a3d_b2d(z, z_rsos, a, b, b, c_c2_merge_tiles_fn=add_tiles, rmsnorm_eps=0.0)
cuda_sync()
torch.cuda.synchronize()
end = time.time()
print("pair mma (merge function is elementwise add)", end - start)
print("pair mma diff", (z - 2 * z_torch).abs().max())


exit()
z_torch *= 2  # TODO this is hacky

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
    (z - z_torch).abs().log().reshape((B * M, N)),
    title="diff",
    xlabel="BM",
    ylabel="N",
    cmap="viridis",
)
# plot_heatmap(
#     (z - z_torch).abs().log().reshape((B * M // 256, 256, N // 256, 256)).transpose(1, 2).reshape((B * M // 256, N // 256, -1)).max(-1).values,
#     title="max-pooled diff",
#     xlabel="BM",
#     ylabel="N",
#     cmap="viridis",
# )

g4b.device.teardown()
