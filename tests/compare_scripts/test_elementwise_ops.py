import time

import matplotlib.pyplot as plt
import torch

import g4b.device
from g4b.kernels.elementwise_ops import add, mul

g4b.device.init(0)

import triton.runtime.driver

triton.runtime.driver.active.utils.set_printf_fifo_size(1024 * 1024 * 256)

cuda_sync = lambda: g4b.device.stream.sync()

# B, T, V = 300, 17, 256
# B, T, V = 1, 1, 512
B, h, T, k = 4, 48, 1, 512

embed_dtype = torch.float32
# embed_dtype = torch.float16
# embed_dtype = torch.bfloat16
# embed_dtype = torch.int8

A = torch.randn((B, h, T, k), dtype=embed_dtype, device="cuda")
K = torch.randn((B, h, T, k), dtype=embed_dtype, device="cuda")
K_rsos = (torch.randn((B, h, T), dtype=embed_dtype, device="cuda") * 20).abs() + 0.1
K_rmsnorm_w = torch.randn((k,), dtype=embed_dtype, device="cuda")
Y = torch.randn((B, h, T, k), dtype=embed_dtype, device="cuda")
eps = 1e-6
output_scale_factor = 5.0
Y_orig = Y.clone()

N_reps = 1000


def restore_inputs():
    global Y
    Y = Y_orig.clone()
    torch.cuda.synchronize()


def measure(name, custom_fn, torch_fn):
    # warmup
    torch.cuda.synchronize()
    restore_inputs()
    for _ in range(N_reps):
        custom_fn()
    cuda_sync()
    restore_inputs()
    for _ in range(N_reps):
        torch_fn()
    torch.cuda.synchronize()

    # measure custom
    restore_inputs()
    start = time.time()
    for _ in range(N_reps):
        custom_fn()
    cuda_sync()
    end = time.time()
    print(name, "custom", end - start)

    # measure torch
    restore_inputs()
    start = time.time()
    for _ in range(N_reps):
        torch_fn()
    torch.cuda.synchronize()
    end = time.time()
    print(name, "torch", end - start)

    # capture one-step outputs so small errors do not accumulate over repeated application.
    restore_inputs()
    custom_fn()
    cuda_sync()
    out_custom = Y.reshape((-1, k)).clone()
    restore_inputs()
    torch_fn()
    torch.cuda.synchronize()
    out_torch = Y.reshape((-1, k)).clone()

    print(name)
    print("  abs diff", (out_custom - out_torch).abs().max())
    print("  custom abs max", out_custom.abs().max())
    print("  torch abs max", out_torch.abs().max())
    print("  custom min/max", out_custom.min(), out_custom.max())

    plot_heatmap(
        (out_custom - out_torch).abs().reshape((B * T * h, k)),
        title=name + " diff",
        xlabel="k",
        ylabel="bth",
        cmap="viridis",
    )


def do_add():
    add(A, K, Y, output_scale_factor, eps, K_rsos, K_rmsnorm_w)


def do_add_torch():
    global Y
    rms = (K_rsos.flatten().unsqueeze(-1) / k + eps).sqrt()
    Y = output_scale_factor * (A.reshape((-1, k)) + K.reshape((-1, k)) / rms * K_rmsnorm_w.flatten())


def do_mul():
    mul(A, K_rmsnorm_w, Y, output_scale_factor=output_scale_factor, rmsnorm_eps=eps)


def do_mul_torch():
    global Y
    Y = output_scale_factor * A.reshape((-1, k)) * K_rmsnorm_w.flatten()


def do_mul_with_rmsnorm():
    mul(
        A,
        K,
        Y,
        output_scale_factor=output_scale_factor,
        b_rsos=K_rsos,
        b_rmsnorm_w=K_rmsnorm_w,
        rmsnorm_eps=eps,
    )


def do_mul_with_rmsnorm_torch():
    global Y
    rms = (K_rsos.flatten().unsqueeze(-1) / k + eps).sqrt()
    Y = output_scale_factor * A.reshape((-1, k)) * (K.reshape((-1, k)) / rms * K_rmsnorm_w.flatten())


def plot_heatmap(
    x: torch.Tensor,
    *,
    title=None,
    xlabel=None,
    ylabel=None,
    cmap=None,
):
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"x must be a torch.Tensor, got {type(x).__name__}")

    if x.ndim != 2:
        raise ValueError(f"x must be 2D, got shape {tuple(x.shape)}")

    fig, ax = plt.subplots()
    im = ax.imshow(x.detach().float().cpu().numpy(), aspect="auto", cmap=cmap)

    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    plt.show()
    return fig, ax


torch.set_float32_matmul_precision("medium")

measure("add", do_add, do_add_torch)
measure("mul", do_mul, do_mul_torch)
measure("mul_with_rmsnorm", do_mul_with_rmsnorm, do_mul_with_rmsnorm_torch)

g4b.device.teardown()
