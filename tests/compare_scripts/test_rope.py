import torch
import time
import g4b.device
from g4b.kernels.rope import populate_rope_frequencies, apply_rope

g4b.device.init(0)

import triton.runtime.driver

triton.runtime.driver.active.utils.set_printf_fifo_size(1024 * 1024 * 256)

cuda_sync = lambda: g4b.device.stream.sync()

#B, T, V = 300, 17, 256
#B, T, V = 1, 1, 512
B, h, T, k = 1, 48, 1, 512

embed_dtype = torch.float32
#embed_dtype = torch.float16
#embed_dtype = torch.bfloat16
#embed_dtype = torch.int8

# Test case 1
rope_freq_scales = torch.cat([torch.ones((k // 4,), device="cuda"), torch.full((k // 4,), float("-inf"), device="cuda")], dim=-1)
rope_freq_scales_orig = rope_freq_scales.clone()

# Test case 2
# rope_freq_scales = None
# rope_freq_scales_orig = None


assert k % 2 == 0
is_first_iter = True
is_first_iter_torch = True
always_recompute_rope = False
rope_freqs_out = torch.empty((k // 2,), device="cuda")
freq_base = 100000.0
A = torch.randn((B, h, T, k), device="cuda")
K = torch.randn((B, h, T, k), device="cuda")
Q_rsos = (A * A).sum(-1)
K_rsos = (K * K).sum(-1)
Q_rmsnorm_w = torch.randn((k,), device="cuda")
K_rmsnorm_w = torch.randn((k,), device="cuda")
Q_orig = A.clone()
K_orig = K.clone()
Q_rsos_orig = Q_rsos.clone()
K_rsos_orig = K_rsos.clone()
base_time_offset = 1000
base_time_offsets_tensor = torch.tensor([base_time_offset], device="cuda")
rmsnorm_eps = 1e-6


class RoPE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rope_freq_base = freq_base
        self.head_dim = k
        self.freqs = None

    def _init(self):
        self.freqs = 1.0 / (self.rope_freq_base ** (torch.arange(0, self.head_dim, 2).cuda() / self.head_dim))
        if rope_freq_scales is not None:
            self.freqs /= rope_freq_scales

    @staticmethod
    def rotate_half(x: torch.Tensor):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, *tensors: torch.Tensor, base_time_offset: int = 0) -> tuple[torch.Tensor, ...]:
        T = tensors[0].shape[-2]
        assert all(t.shape[-2] == T for t in tensors)

        theta = torch.arange(base_time_offset, base_time_offset + T, device="cuda").reshape((T, 1)) * self.freqs
        theta = torch.cat([theta, theta], dim=-1)

        sin = theta.sin()  # shape: (seq_len, head_dim)
        cos = theta.cos()  # shape: (seq_len, head_dim)

        return tuple(tensor * cos + RoPE.rotate_half(tensor) * sin for tensor in tensors)


torch_rope_mod = RoPE()


def restore_inputs():
    global rope_freq_scales, A, K, Q_rsos, K_rsos
    if rope_freq_scales_orig is not None:
        rope_freq_scales = rope_freq_scales_orig.clone()
    A = Q_orig.clone()
    K = K_orig.clone()
    Q_rsos = Q_rsos_orig.clone()
    K_rsos = K_rsos_orig.clone()
    torch.cuda.synchronize()


def do_forward():
    global rope_freqs_out, is_first_iter
    if is_first_iter or always_recompute_rope:
        populate_rope_frequencies(rope_freqs_out, rope_freq_scales, freq_base)
        is_first_iter = False
    apply_rope(
        A,
        K,
        rope_freqs_out,
        base_time_offsets_tensor,
        Q_rsos,
        K_rsos,
        Q_rmsnorm_w,
        K_rmsnorm_w,
        rmsnorm_eps,
    )


def do_forward_torch():
    global rope_freqs_out, is_first_iter_torch, A, K
    if is_first_iter_torch or always_recompute_rope:
        torch_rope_mod._init()
        rope_freqs_out = torch_rope_mod.freqs
        is_first_iter_torch = False
    q = A / (Q_rsos.unsqueeze(-1) / k + rmsnorm_eps).sqrt() * Q_rmsnorm_w
    k_norm = K / (K_rsos.unsqueeze(-1) / k + rmsnorm_eps).sqrt() * K_rmsnorm_w
    A, K = torch_rope_mod(q, k_norm, base_time_offset=base_time_offset)


def capture_out():
    return rope_freqs_out, A, K


N_reps = 1000

# warmup
torch.set_float32_matmul_precision("medium")
torch.cuda.synchronize()
restore_inputs()
for _ in range(N_reps):
    do_forward()
cuda_sync()
restore_inputs()
for _ in range(N_reps):
    do_forward_torch()
torch.cuda.synchronize()

# measure custom
restore_inputs()
start = time.time()
for _ in range(N_reps):
    do_forward()
outs_custom = capture_out()
cuda_sync()
end = time.time()
print("custom", end - start)

# measure torch
restore_inputs()
start = time.time()
for _ in range(N_reps):
    do_forward_torch()
outs_torch = capture_out()
torch.cuda.synchronize()
end = time.time()
print("torch", end - start)


# capture outputs so small errors don't accumulate over 1000s of steps (small diff to torch is ok after all)
restore_inputs()
do_forward()
outs_custom = capture_out()
cuda_sync()
restore_inputs()
do_forward_torch()
outs_torch = capture_out()
torch.cuda.synchronize()

for name, out_custom, out_torch in zip(["rope_freqs", "Q", "K"], outs_custom, outs_torch):
    print(name + ":")
    print("  abs diff", (out_custom - out_torch).abs().max())
    print("  custom abs max", out_custom.abs().max())
    print("  torch abs max", out_torch.abs().max())
    print("  custom min/max", out_custom.min(), out_custom.max())


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


for i, out_custom, out_torch in zip(range(len(outs_custom)), outs_custom, outs_torch):
    plot_heatmap(
        (out_custom - out_torch).abs().reshape((B * T * h, k) if i > 0 else (1, k // 2)),
        title=["rope_freqs", "Q", "K"][i] + " diff",
        xlabel="k",
        ylabel="bth",
        cmap="viridis",
    )

g4b.device.teardown()
