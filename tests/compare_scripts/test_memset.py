import torch
import triton
import time
import g4b.tensor
import g4b.device
from g4b.kernels.memset import memset_contiguous

# TODO I really should be using cuda events probably to time this because currently the timings I get out of this
#  feel mainly caused by overhead. This should not take 5ms.

g4b.device.init(0)
triton.runtime.driver.active.utils.set_printf_fifo_size(1024 * 1024 * 1024 * 8)

cuda_sync = lambda: g4b.device.stream.sync()

x_cpu = torch.randn(1, 1, 16384, 16384) * 3

start = time.time()
raw = x_cpu.detach().cpu().contiguous().numpy().tobytes()
end = time.time()
print("time for move", end - start)

x = x_cpu.to("cuda")
y = torch.empty_like(x)
raw_x = g4b.tensor.Tensor.from_bytes_sync(
    raw, g4b.tensor.float32, tuple(x_cpu.shape),
    # not required because contiguous: tuple(x_cpu.stride(i) for i in range(x_cpu.ndim))
)
torch.cuda.synchronize()
memset_contiguous(raw_x, 0)
#torch.cuda.synchronize()
cuda_sync()

y[...] = x[...]
torch.cuda.synchronize()

start = time.time()
y.zero_()
torch.cuda.synchronize()
end = time.time()
print("torch:", end - start)

y[...] = x[...]
torch.cuda.synchronize()

start = time.time()
memset_contiguous(raw_x, 0)
#torch.cuda.synchronize()
cuda_sync()
end = time.time()
print("kernel w/ torch:", end - start)
x2 = torch.frombuffer(raw_x.to_bytes_sync(), dtype=torch.uint8)
print("min/max:", x2.min(), x2.max(), x2.to(torch.float32).mean())

g4b.device.teardown()
