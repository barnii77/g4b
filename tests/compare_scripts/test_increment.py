import time

import torch

import g4b.device
from g4b.kernels.increment import increment

g4b.device.init(0)

import triton.runtime.driver

triton.runtime.driver.active.utils.set_printf_fifo_size(1024 * 1024 * 256)

cuda_sync = lambda: g4b.device.stream.sync()

# 1D integer tensor test, matching the kernel's current contract.
N = 4096 ** 2
increment_by = 17
modulus = 257

x_dtype = torch.int32
x = torch.randint(0, modulus, (N,), dtype=x_dtype, device="cuda")
x_orig = x.clone()


def restore_inputs():
    global x
    x = x_orig.clone()
    torch.cuda.synchronize()


def do_forward():
    increment(x, increment_by, modulus)


def do_forward_torch():
    global x
    x = (x + increment_by) % modulus


def capture_out():
    return x


N_reps = 1000

# warmup
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
out_custom = capture_out()
cuda_sync()
end = time.time()
print("custom", end - start)

# measure torch
restore_inputs()
start = time.time()
for _ in range(N_reps):
    do_forward_torch()
out_torch = capture_out()
torch.cuda.synchronize()
end = time.time()
print("torch", end - start)

# capture outputs so we compare a single-step result, not accumulated drift
restore_inputs()
do_forward()
out_custom = capture_out().clone()
cuda_sync()
restore_inputs()
do_forward_torch()
out_torch = capture_out().clone()
torch.cuda.synchronize()

diff = (out_custom.to(torch.int64) - out_torch.to(torch.int64)).abs()
print("abs diff", diff.max())
print("custom abs max", out_custom.abs().max())
print("torch abs max", out_torch.abs().max())
print("custom min/max", out_custom.min(), out_custom.max())
