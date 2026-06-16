import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn.attention.varlen import varlen_attn_out

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import g4b.device
from g4b.kernels.fa2 import STAGE_CAUSAL, STAGE_FULL, flash_attention

g4b.device.init(0)

import triton.runtime.driver

triton.runtime.driver.active.utils.set_printf_fifo_size(1024 * 1024 * 256)


def sync_all():
    g4b.device.stream.sync()
    torch.cuda.synchronize()


dt = torch.float16
N_WARMUP = 10
torch.set_float32_matmul_precision("high")


@dataclass(frozen=True)
class Case:
    name: str
    B: int
    H: int
    G: int
    Tq: int
    Tcache: int
    D: int
    ctx_window_size: int
    stage: int = STAGE_CAUSAL
    phase: str = "prefill"
    reps: int = 100
    warmups: int = N_WARMUP
    max_kv_splits: int = 4
    compare_torch: bool = True
    varlen_num_splits: int | None = None


CASES = (
    Case("decode_small_w1024",    B=4, H=8,  G=2, Tq=1, Tcache=1024,  D=128, ctx_window_size=1024,  phase="decode", reps=200),
    Case("decode_gemma_w4096",    B=8, H=16, G=4, Tq=1, Tcache=4096,  D=256, ctx_window_size=4096,  phase="decode", reps=50),
    Case("decode_gemma_w8192",    B=4, H=16, G=4, Tq=1, Tcache=8192,  D=256, ctx_window_size=8192,  phase="decode", reps=20),
    Case("decode_gemma_w16384",   B=2, H=16, G=4, Tq=1, Tcache=16384, D=256, ctx_window_size=16384, phase="decode", reps=10, warmups=3),
    Case("decode_gemma_w32768",   B=1, H=16, G=4, Tq=1, Tcache=32768, D=256, ctx_window_size=32768, phase="decode", reps=5, warmups=2),
    Case("decode_gemma_w65536",   B=1, H=16, G=4, Tq=1, Tcache=65536, D=256, ctx_window_size=65536, phase="decode", reps=3, warmups=1, max_kv_splits=64),
    Case("decode_biggroup_w4096", B=4, H=32, G=4, Tq=1, Tcache=4096,  D=256, ctx_window_size=4096,  phase="decode", reps=30),

    Case("prefill_t128_w4096",    B=2, H=16, G=4, Tq=128,   Tcache=4096,  D=256, ctx_window_size=4096,  reps=10),
    Case("prefill_t256_w4096",    B=1, H=16, G=4, Tq=256,   Tcache=4096,  D=256, ctx_window_size=4096,  reps=6),
    Case("prefill_t256_w8192",    B=1, H=16, G=4, Tq=256,   Tcache=8192,  D=256, ctx_window_size=8192,  reps=3),
    Case("prefill_t256_w65536",   B=1, H=16, G=4, Tq=256,   Tcache=65536, D=256, ctx_window_size=65536, reps=1, warmups=1),
    Case(
        "prefill_t65536_w65536",
        B=1,
        H=16,
        G=4,
        Tq=65536,
        Tcache=65536,
        D=256,
        ctx_window_size=65536,
        reps=1,
        warmups=0,
        max_kv_splits=1,
    ),

    Case("full_t256_w256", B=2, H=16, G=4, Tq=256, Tcache=256, D=256, ctx_window_size=256, stage=STAGE_FULL, reps=10),
)


def restore_output(y, y_orig):
    y.copy_(y_orig)
    sync_all()


def choose_varlen_num_splits(case: Case) -> int | None:
    """
    Choose torch varlen num_splits using the same style of heuristic as fa2.py.

    Important:
    - Does NOT touch/force FA2 configs.
    - Only forces split-KV for decode Tq=1.
    - Leaves prefill/full larger-query cases to torch's normal path by returning None.
    - Env override exists only for varlen side:
        TORCH_VARLEN_NUM_SPLITS=16 python ...
        TORCH_VARLEN_NUM_SPLITS=none python ...
    """
    return None  # TODO requires FA3 because pytorch stupid
    override = os.environ.get("TORCH_VARLEN_NUM_SPLITS")
    if override is not None and override != "":
        if override.lower() in {"none", "auto", "default"}:
            return None
        return int(override)

    if case.varlen_num_splits is not None:
        return case.varlen_num_splits

    # Split-KV is relevant for decode, not prefill.
    if case.phase != "decode" or case.Tq != 1:
        return None

    max_kv_splits = case.max_kv_splits
    ctx_window_size = case.ctx_window_size

    if max_kv_splits <= 1 or ctx_window_size <= 4096:
        target_splits = 1
    elif ctx_window_size < 8192:
        target_splits = 4
    elif ctx_window_size < 32768:
        target_splits = 8
    else:
        target_splits = 16

    return min(target_splits, max_kv_splits)


def logical_cache_one(x_b, time_dim_size: int, ctx_window_size: int):
    # x_b: [G, Tcache, D]
    window_size = min(time_dim_size, ctx_window_size)
    ring_start = max(time_dim_size - ctx_window_size, 0) % x_b.shape[1]
    return torch.roll(x_b, shifts=-ring_start, dims=1)[:, :window_size]


def pack_q(q):
    # q: [B, H, Tq, D] -> [sum_q, H, D]
    B, H, Tq, D = q.shape
    pieces = [q[b].permute(1, 0, 2).contiguous() for b in range(B)]
    q_packed = torch.cat(pieces, dim=0)
    cu_q = torch.arange(0, (B + 1) * Tq, Tq, dtype=torch.int32, device=q.device)
    return q_packed, cu_q, Tq


def pack_logical_kv(k_cache, v_cache, time_dim_sizes, ctx_window_size):
    # k/v cache: [B, G, Tcache, D] -> [sum_k, G, D]
    B, G, Tcache, D = k_cache.shape

    k_pieces = []
    v_pieces = []
    lengths = []

    time_dim_sizes_cpu = time_dim_sizes.detach().cpu().tolist()
    for b in range(B):
        time_dim_size = int(time_dim_sizes_cpu[b])

        k_b = logical_cache_one(k_cache[b], time_dim_size, ctx_window_size)
        v_b = logical_cache_one(v_cache[b], time_dim_size, ctx_window_size)

        Tk = k_b.shape[1]
        lengths.append(Tk)

        k_pieces.append(k_b.permute(1, 0, 2).contiguous())
        v_pieces.append(v_b.permute(1, 0, 2).contiguous())

    k_packed = torch.cat(k_pieces, dim=0)
    v_packed = torch.cat(v_pieces, dim=0)

    cu_k_vals = [0]
    for n in lengths:
        cu_k_vals.append(cu_k_vals[-1] + n)

    cu_k = torch.tensor(cu_k_vals, dtype=torch.int32, device=k_cache.device)
    return k_packed, v_packed, cu_k, max(lengths)


def unpack_q_output(out_packed, B: int, H: int, Tq: int, D: int):
    # [B*Tq, H, D] -> [B, H, Tq, D]
    ys = []
    off = 0
    for _ in range(B):
        y_b = out_packed[off : off + Tq].permute(1, 0, 2).contiguous()
        ys.append(y_b)
        off += Tq
    return torch.stack(ys, dim=0)


def make_varlen_ref(q, k_cache, v_cache, time_dim_sizes, ctx_window_size, stage, num_splits: int | None):
    """
    Pre-packs Q/K/V once outside timing, matching the old torch-ref behavior:
    torch/varlen is not charged for ring-cache linearization.
    """
    B, H, Tq, D = q.shape
    G = k_cache.shape[1]

    q_packed, cu_q, max_q = pack_q(q)
    k_packed, v_packed, cu_k, max_k = pack_logical_kv(
        k_cache,
        v_cache,
        time_dim_sizes,
        ctx_window_size,
    )

    out_packed = torch.empty_like(q_packed)

    # For decode with STAGE_FULL, no mask.
    # For causal prefill/decode, FlashAttention-style varlen causal is expressed as (-1, 0).
    window_size = (-1, -1) if stage == STAGE_FULL else (-1, 0)

    def ref():
        return varlen_attn_out(
            out_packed,
            q_packed,
            k_packed,
            v_packed,
            cu_q,
            cu_k,
            max_q,
            max_k,
            scale=1.0,
            window_size=window_size,
            enable_gqa=H != G,
            num_splits=num_splits,
        )

    def unpack():
        return unpack_q_output(out_packed, B, H, Tq, D)

    return ref, unpack


def time_fn(fn, reps):
    sync_all()
    start = time.perf_counter()
    for _ in range(reps):
        fn()
    sync_all()
    return time.perf_counter() - start


def bench_case(case: Case):
    varlen_num_splits = choose_varlen_num_splits(case)

    print(
        f"\n{case.name}: B={case.B} H={case.H} G={case.G} Tq={case.Tq} "
        f"Tcache={case.Tcache} D={case.D} window={case.ctx_window_size} "
        f"stage={'causal' if case.stage == STAGE_CAUSAL else 'full'} phase={case.phase} "
        f"varlen_num_splits={varlen_num_splits}"
    )

    q = torch.randn(case.B, case.H, case.Tq, case.D, dtype=dt, device="cuda")
    k_cache = torch.randn(case.B, case.G, case.Tcache, case.D, dtype=dt, device="cuda")
    v_cache = torch.randn(case.B, case.G, case.Tcache, case.D, dtype=dt, device="cuda")

    y_baseline = torch.empty(case.B, case.H, case.Tq, case.D, dtype=dt, device="cuda")
    y_grouped = torch.empty_like(y_baseline)
    y_grouped_split = torch.empty_like(y_baseline)
    y_orig = torch.zeros_like(y_baseline)

    partial_o = torch.empty(
        case.max_kv_splits,
        case.B,
        case.H,
        case.Tq,
        case.D,
        dtype=dt,
        device="cuda",
    )
    partial_l = torch.empty(
        case.max_kv_splits,
        case.B,
        case.H,
        case.Tq,
        dtype=torch.float32,
        device="cuda",
    )
    partial_m = torch.empty(
        case.max_kv_splits,
        case.B,
        case.H,
        case.Tq,
        dtype=torch.float32,
        device="cuda",
    )

    # Cover both no-wrap and wrapped users when possible.
    time_dim_sizes = torch.tensor(
        [case.ctx_window_size if b % 2 == 0 else case.ctx_window_size + 17 for b in range(case.B)],
        dtype=torch.int32,
        device="cuda",
    )

    user_phase = 1 if case.phase == "decode" else 0
    user_in_prefill_or_decode = torch.full(
        (case.B,),
        user_phase,
        dtype=torch.uint8,
        device="cuda",
    )

    def baseline():
        flash_attention(
            q,
            k_cache,
            v_cache,
            y_baseline,
            time_dim_sizes,
            user_in_prefill_or_decode,
            case.ctx_window_size,
            case.phase,
            stage=case.stage,
            use_grouped_query_tile=False,
        )

    def grouped():
        flash_attention(
            q,
            k_cache,
            v_cache,
            y_grouped,
            time_dim_sizes,
            user_in_prefill_or_decode,
            case.ctx_window_size,
            case.phase,
            stage=case.stage,
            use_grouped_query_tile=True,
        )

    def grouped_split():
        flash_attention(
            q,
            k_cache,
            v_cache,
            y_grouped_split,
            time_dim_sizes,
            user_in_prefill_or_decode,
            case.ctx_window_size,
            case.phase,
            partial_o,
            partial_l,
            partial_m,
            stage=case.stage,
            use_grouped_query_tile=True,
        )

    if case.compare_torch:
        varlen_ref, unpack_varlen = make_varlen_ref(
            q,
            k_cache,
            v_cache,
            time_dim_sizes,
            case.ctx_window_size,
            case.stage,
            varlen_num_splits,
        )
    else:
        varlen_ref, unpack_varlen = None, None

    sync_all()

    for _ in range(case.warmups):
        restore_output(y_baseline, y_orig)
        baseline()
    sync_all()

    for _ in range(case.warmups):
        restore_output(y_grouped, y_orig)
        grouped()
    sync_all()

    for _ in range(case.warmups):
        restore_output(y_grouped_split, y_orig)
        grouped_split()
    sync_all()

    if varlen_ref is not None:
        for _ in range(case.warmups):
            varlen_ref()
        sync_all()

    restore_output(y_baseline, y_orig)
    baseline_s = time_fn(baseline, case.reps)

    restore_output(y_grouped, y_orig)
    grouped_s = time_fn(grouped, case.reps)

    restore_output(y_grouped_split, y_orig)
    grouped_split_s = time_fn(grouped_split, case.reps)

    varlen_reps = max(1, min(case.reps, 20))
    varlen_s = time_fn(varlen_ref, varlen_reps) if varlen_ref is not None else None

    restore_output(y_baseline, y_orig)
    baseline()
    sync_all()
    out_baseline = y_baseline.clone()

    restore_output(y_grouped, y_orig)
    grouped()
    sync_all()
    out_grouped = y_grouped.clone()

    restore_output(y_grouped_split, y_orig)
    grouped_split()
    sync_all()
    out_grouped_split = y_grouped_split.clone()

    if varlen_ref is not None:
        varlen_ref()
        sync_all()
        out_varlen = unpack_varlen()
    else:
        out_varlen = None

    print(f"  baseline      {baseline_s:.6f}s ({baseline_s / case.reps * 1e3:.4f} ms/iter)")
    print(f"  grouped       {grouped_s:.6f}s ({grouped_s / case.reps * 1e3:.4f} ms/iter)")
    print(f"  split         {grouped_split_s:.6f}s ({grouped_split_s / case.reps * 1e3:.4f} ms/iter)")

    if varlen_s is not None:
        print(f"  varlen        {varlen_s:.6f}s ({varlen_s / varlen_reps * 1e3:.4f} ms/iter)")
        print("  split/varlen", grouped_split_s / varlen_s)
    else:
        print("  varlen skipped")

    print("  grouped/baseline", grouped_s / baseline_s)
    print("  split/grouped", grouped_split_s / grouped_s)

    if out_varlen is not None:
        print("  baseline abs diff vs varlen", (out_baseline - out_varlen).abs().max())
        print("  grouped  abs diff vs varlen", (out_grouped - out_varlen).abs().max())
        print("  split    abs diff vs varlen", (out_grouped_split - out_varlen).abs().max())

    print("  grouped vs baseline abs diff", (out_grouped - out_baseline).abs().max())
    print("  split vs grouped abs diff", (out_grouped_split - out_grouped).abs().max())
    print("  grouped min/max", out_grouped.min(), out_grouped.max())


try:
    for case in CASES:
        bench_case(case)
finally:
    g4b.device.teardown()