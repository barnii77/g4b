import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import g4b.device
from g4b.kernels.rsos import compute_rsos


g4b.device.init(0)

cuda_sync = lambda: g4b.device.stream.sync()


def check(shape, scale):
    x = torch.randn(shape, dtype=torch.float32, device="cuda")
    rsos = torch.empty(shape[:-1], dtype=torch.float32, device="cuda")

    compute_rsos(x, rsos, scale=scale)
    cuda_sync()

    ref = ((x * scale) ** 2).sum(-1)
    torch.cuda.synchronize()

    print(f"shape={shape} scale={scale}")
    print("  abs diff", (rsos - ref).abs().max())
    print("  custom abs max", rsos.abs().max())
    print("  torch abs max", ref.abs().max())


check((17, 513), 1.0)
check((2, 3, 17, 513), 0.125)

g4b.device.teardown()
