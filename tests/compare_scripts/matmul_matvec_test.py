"""Correctness tests for the USE_MATVEC matmul path (single-token decode, no tl.dot).

Runs small shapes so it iterates fast. Each case checks matmul_a3d_b2d(..., use_matvec=True)
against a torch dense reference built from the SAME fp16-dequantized weights the kernel uses
(UPCAST_DTYPE=fp16), so the only expected error is fp32 reduction-order noise.

Run: G4B_SKIP_TUNING=1 python tests/compare_scripts/matmul_matvec_test.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import g4b.device
from g4b.gguf import GGUFType, GGUFTensor
from g4b.kernels.matmul import matmul_a3d_b2d
from g4b.kernels.matmul_epilogue import geglu_fusion_matmul_merge_tiles_mixin_jfn, ple_gate_storer_jfn
from g4b.tensor import Tensor
from scripts.reference_impl import dequant_q4k_to_fp32, dequant_q5k_to_fp32, dequant_q6k_to_fp32

g4b.device.init(0)


# ---- small synthetic K-quant weight builders ([N, K] stored, K a multiple of 256) ----
def _q4k(N, K):
    nb = N * (K // 256)
    raw = torch.empty((nb, 144), dtype=torch.uint8)
    raw[:, 0:2] = (torch.rand(nb, dtype=torch.float16) * 0.25 + 0.01).view(torch.uint8).reshape(nb, 2)
    raw[:, 2:4] = (torch.rand(nb, dtype=torch.float16) * 0.25).view(torch.uint8).reshape(nb, 2)
    raw[:, 4:16] = torch.randint(0, 256, (nb, 12), dtype=torch.uint8)
    raw[:, 16:] = torch.randint(0, 256, (nb, 128), dtype=torch.uint8)
    return GGUFTensor("syn.q4_k", [K, N], GGUFType.GGML_TYPE_Q4_K, raw.numpy().tobytes())


def _q5k(N, K):
    nb = N * (K // 256)
    raw = torch.empty((nb, 176), dtype=torch.uint8)
    raw[:, 0:2] = (torch.rand(nb, dtype=torch.float16) * 0.25 + 0.01).view(torch.uint8).reshape(nb, 2)
    raw[:, 2:4] = (torch.rand(nb, dtype=torch.float16) * 0.25).view(torch.uint8).reshape(nb, 2)
    raw[:, 4:16] = torch.randint(0, 256, (nb, 12), dtype=torch.uint8)
    raw[:, 16:48] = torch.randint(0, 256, (nb, 32), dtype=torch.uint8)
    raw[:, 48:] = torch.randint(0, 256, (nb, 128), dtype=torch.uint8)
    return GGUFTensor("syn.q5_k", [K, N], GGUFType.GGML_TYPE_Q5_K, raw.numpy().tobytes())


def _q6k(N, K):
    nb = N * (K // 256)
    raw = torch.empty((nb, 210), dtype=torch.uint8)
    raw[:, 0:128] = torch.randint(0, 256, (nb, 128), dtype=torch.uint8)
    raw[:, 128:192] = torch.randint(0, 256, (nb, 64), dtype=torch.uint8)
    raw[:, 192:208] = torch.randint(-128, 128, (nb, 16), dtype=torch.int8).view(torch.uint8)
    raw[:, 208:210] = (torch.rand(nb, dtype=torch.float16) * 0.25 + 0.01).view(torch.uint8).reshape(nb, 2)
    return GGUFTensor("syn.q6_k", [K, N], GGUFType.GGML_TYPE_Q6_K, raw.numpy().tobytes())


BUILD = {"q4_k": (_q4k, dequant_q4k_to_fp32), "q5_k": (_q5k, dequant_q5k_to_fp32), "q6_k": (_q6k, dequant_q6k_to_fp32)}


def _ref(a, gguf, deq, N, K):
    # Match the kernel's fp16 weight upcast, accumulate in fp32: isolates reduction-order noise.
    w = deq(gguf).reshape(N, K).to("cuda").to(torch.float16).to(torch.float32)
    return (a.reshape(1, K) @ w.T).reshape(1, 1, N)


def run_case(name, quant, K, N, *, keep_c=False, tol=2e-2):
    torch.manual_seed(0)
    build, deq = BUILD[quant]
    a = torch.randn((1, 1, K), dtype=torch.float32, device="cuda") / (K**0.5)
    gguf = build(N, K)
    b = Tensor.from_gguf_tensor(gguf)
    ref = _ref(a, gguf, deq, N, K)

    if keep_c:
        # KEEP_C accumulates onto the existing c: seed c with a known tensor and add it to the reference.
        c = torch.randn((1, 1, N), dtype=torch.float32, device="cuda")
        ref = ref + c
    else:
        c = torch.empty((1, 1, N), dtype=torch.float32, device="cuda")

    matmul_a3d_b2d(c, None, a, b, transpose_b_before_mma=True, use_matvec=True, keep_c=keep_c, rmsnorm_eps=0.0)
    g4b.device.stream.sync()
    torch.cuda.synchronize()

    max_diff = (c - ref).abs().max().item()
    scale = ref.abs().max().item() + 1e-6
    ok = max_diff <= tol * scale
    print(
        f"[{'PASS' if ok else 'FAIL'}] {name:<28} {quant} K={K} N={N} "
        f"max_diff={max_diff:.5g} rel={max_diff / scale:.2e}"
    )
    assert ok, f"{name}: max_diff {max_diff} exceeds tol {tol} * {scale}"


def run_fused(name, *, fn, tol=5e-3):
    # Compare the matvec path against the (known-good) tl.dot path for a fused epilogue. Catches the
    # split-K vs nonlinear-epilogue hazard: gelu(partial_K) summed over splits != gelu(full).
    out_dot = fn(use_matvec=False).float()
    out_mv = fn(use_matvec=True).float()
    md = (out_dot - out_mv).abs().max().item()
    scale = out_dot.abs().max().item() + 1e-6
    ok = md <= tol * scale
    print(f"[{'PASS' if ok else 'FAIL'}] {name:<28} dot_vs_matvec max_diff={md:.5g} rel={md / scale:.2e}")
    assert ok, f"{name}: matvec vs dot max_diff {md} exceeds tol {tol} * {scale}"


def _geglu(use_matvec, *, with_input_rsos):
    torch.manual_seed(1)
    D, U = 512, 512
    X = torch.randn(1, 1, D, dtype=torch.float16, device="cuda")
    rsos = (X.float() ** 2).sum(-1).to(torch.float32) if with_input_rsos else None
    Wup = torch.randn(D, U, dtype=torch.float16, device="cuda") * D**-0.5
    Wg = torch.randn(D, U, dtype=torch.float16, device="cuda") * D**-0.5
    H = torch.empty(1, 1, U, dtype=torch.float32, device="cuda")
    matmul_a3d_b2d(
        H,
        None,
        X,
        Wup,
        Wg,
        c_c2_merge_tiles_fn=geglu_fusion_matmul_merge_tiles_mixin_jfn,
        input_rmsnorm_sum_of_squares=rsos,
        use_matvec=use_matvec,
        rmsnorm_eps=1e-6,
    )
    g4b.device.stream.sync()
    torch.cuda.synchronize()
    return H


def _ple_gate(use_matvec):
    torch.manual_seed(2)
    D, P = 512, 256
    X = torch.randn(1, 1, D, dtype=torch.float16, device="cuda")
    Wpg = torch.randn(D, P, dtype=torch.float16, device="cuda") * D**-0.5
    PLE = torch.randn(1, 1, P, dtype=torch.float16, device="cuda")
    H = torch.empty(1, 1, P, dtype=torch.float32, device="cuda")
    matmul_a3d_b2d(
        H,
        None,
        X,
        Wpg,
        storer_extra=PLE,
        storer_fn=ple_gate_storer_jfn,
        use_matvec=use_matvec,
        rmsnorm_eps=1e-6,
    )
    g4b.device.stream.sync()
    torch.cuda.synchronize()
    return H


def main():
    # The default use_matvec config (G4B_SKIP_TUNING) is a split-K matvec (split_k=4), so these exercise the
    # split-K atomic-add + pre-hook zeroing path across quant types.
    run_case("q6 decode", "q6_k", 512, 256)
    run_case("q4 decode", "q4_k", 512, 256)
    run_case("q5 decode", "q5_k", 512, 256)
    # N not a multiple of the default N-tile (32) -> partial last tile, exercises the store mask.
    run_case("q6 ragged-N", "q6_k", 512, 200)
    # KEEP_C accumulate path (the masked _c_ptrs load); KEEP_C forces NUM_K_SPLITS=1 internally.
    run_case("q6 keep_c", "q6_k", 512, 256, keep_c=True)
    # Fused epilogues vs the dot path (nonlinear gelu must not be split across K).
    run_fused("geglu", fn=lambda use_matvec: _geglu(use_matvec, with_input_rsos=False))
    run_fused("geglu+input_rsos", fn=lambda use_matvec: _geglu(use_matvec, with_input_rsos=True))
    run_fused("ple_gate (storer_extra)", fn=_ple_gate)
    print("all matvec tests passed")
    g4b.device.teardown()


if __name__ == "__main__":
    main()
