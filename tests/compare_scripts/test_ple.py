import torch
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import g4b.device
from g4b.kernels.matmul import matmul_a3d_b2d
from g4b.kernels.matmul_epilogue import ple_gate_storer_jfn

g4b.device.init(0)

import triton.runtime.driver

triton.runtime.driver.active.utils.set_printf_fifo_size(1024 * 1024 * 256)

cuda_sync = lambda: g4b.device.stream.sync()

dt = torch.float16
B, T = 2, 256
D, U = 2560, 256
X = torch.randn(B, T, D, dtype=dt, device="cuda")
ple = torch.randn(B, T, U, dtype=dt, device="cuda") * D**-0.5
W_gate = torch.randn(D, U, dtype=dt, device="cuda") * D**-0.5
W_down = torch.randn(U, D, dtype=dt, device="cuda") * U**-0.5
H = torch.randn(B, T, U, dtype=dt, device="cuda")

Y = torch.randn(B, T, D, dtype=dt, device="cuda")


def gated_ple(x):
    return (torch.nn.functional.gelu(x @ W_gate) * ple) @ W_down


def do_forward():
    matmul_a3d_b2d(
        H,
        None,
        X,
        W_gate,
        storer_extra=ple,
        storer_fn=ple_gate_storer_jfn,
        rmsnorm_eps=0.0,
    )
    matmul_a3d_b2d(Y, None, H, W_down, rmsnorm_eps=0.0)


def do_forward_torch():
    return gated_ple(X)


# warmup
torch.set_float32_matmul_precision("medium")
torch.cuda.synchronize()
do_forward()
cuda_sync()
do_forward_torch()
torch.cuda.synchronize()

# measure custom
start = time.time()
do_forward()
cuda_sync()
end = time.time()
print("custom", end - start)

# measure torch
start = time.time()
Y_torch = do_forward_torch()
torch.cuda.synchronize()
end = time.time()
print("torch", end - start)

print("abs diff", (Y - Y_torch).abs().max())
print("custom abs max", Y.abs().max())
print("torch abs max", Y_torch.abs().max())
print("custom min/max", Y.min(), Y.max())
