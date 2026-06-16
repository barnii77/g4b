import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import g4b.device
from g4b.kernels.matmul import matmul_a3d_b2d
from g4b.kernels.matmul_epilogue import geglu_fusion_matmul_merge_tiles_mixin_jfn, ple_gate_storer_jfn


g4b.device.init(0)

import triton.runtime.driver

triton.runtime.driver.active.utils.set_printf_fifo_size(1024 * 1024 * 256)

cuda_sync = lambda: g4b.device.stream.sync()

dt = torch.float16
B, T = 2, 129
D, U = 512, 1536
P = 256
eps = 1e-6
N_REPS = 100

torch.set_float32_matmul_precision("high")

X = torch.randn(B, T, D, dtype=dt, device="cuda")
X_rsos = (X * X).sum(-1)
W_up = torch.randn(D, U, dtype=dt, device="cuda") * D**-0.5
W_gate = torch.randn(D, U, dtype=dt, device="cuda") * D**-0.5
W_down = torch.randn(U, D, dtype=dt, device="cuda") * U**-0.5
W_ple_gate = torch.randn(D, P, dtype=dt, device="cuda") * D**-0.5
W_ple_down = torch.randn(P, D, dtype=dt, device="cuda") * P**-0.5
PLE = torch.randn(B, T, P, dtype=dt, device="cuda")

H = torch.empty(B, T, U, dtype=dt, device="cuda")
H_ple = torch.empty(B, T, P, dtype=dt, device="cuda")
Y = torch.empty(B, T, D, dtype=dt, device="cuda")


def rmsnorm_input():
    return X / (X_rsos.unsqueeze(-1) / D + eps).sqrt()


def check(name, custom_fn, torch_fn, out_fn):
    # warmup / autotune
    torch.cuda.synchronize()
    for _ in range(10):
        custom_fn()
    cuda_sync()
    for _ in range(10):
        torch_fn()
    torch.cuda.synchronize()

    # measure custom
    start = time.time()
    for _ in range(N_REPS):
        custom_fn()
    cuda_sync()
    end = time.time()
    print(name, "custom", end - start)

    # measure torch
    start = time.time()
    for _ in range(N_REPS):
        torch_fn()
    torch.cuda.synchronize()
    end = time.time()
    print(name, "torch", end - start)

    # capture one-step outputs after autotune/warmup so correctness does not depend on autotuner candidates.
    custom_fn()
    cuda_sync()
    out_custom = out_fn().clone()
    out_torch = torch_fn()
    torch.cuda.synchronize()

    diff = (out_custom - out_torch).abs()
    print(name)
    print("  abs diff", diff.max())
    print("  custom abs max", out_custom.abs().max())
    print("  torch abs max", out_torch.abs().max())
    print("  custom min/max", out_custom.min(), out_custom.max())


def do_geglu():
    matmul_a3d_b2d(
        H,
        None,
        X,
        W_up,
        W_gate,
        c_c2_merge_tiles_fn=geglu_fusion_matmul_merge_tiles_mixin_jfn,
        rmsnorm_eps=eps,
    )
    matmul_a3d_b2d(Y, None, H, W_down, rmsnorm_eps=eps)


def do_geglu_torch():
    return (torch.nn.functional.gelu(X @ W_gate) * (X @ W_up)) @ W_down


def do_geglu_with_input_rsos():
    matmul_a3d_b2d(
        H,
        None,
        X,
        W_up,
        W_gate,
        c_c2_merge_tiles_fn=geglu_fusion_matmul_merge_tiles_mixin_jfn,
        input_rmsnorm_sum_of_squares=X_rsos,
        rmsnorm_eps=eps,
    )
    matmul_a3d_b2d(Y, None, H, W_down, rmsnorm_eps=eps)


def do_geglu_with_input_rsos_torch():
    X_norm = rmsnorm_input()
    return (torch.nn.functional.gelu(X_norm @ W_gate) * (X_norm @ W_up)) @ W_down


def do_ple_gate_storer():
    matmul_a3d_b2d(
        H_ple,
        None,
        X,
        W_ple_gate,
        storer_extra=PLE,
        storer_fn=ple_gate_storer_jfn,
        rmsnorm_eps=eps,
    )
    matmul_a3d_b2d(Y, None, H_ple, W_ple_down, rmsnorm_eps=eps)


def do_ple_gate_storer_torch():
    return (torch.nn.functional.gelu(X @ W_ple_gate) * PLE) @ W_ple_down


check("geglu", do_geglu, do_geglu_torch, lambda: Y)
check("geglu_with_input_rsos", do_geglu_with_input_rsos, do_geglu_with_input_rsos_torch, lambda: Y)
check("ple_gate_storer", do_ple_gate_storer, do_ple_gate_storer_torch, lambda: Y)

g4b.device.teardown()
