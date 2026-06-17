import argparse
import math
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from g4b import device, gguf, lifecycle
from g4b.config import Config
from g4b.models import models
from g4b.scheduler import Request, Scheduler
from g4b.tokenizer import ChatTemplate, PromptFragment, Tokenizer
from scripts import reference_impl as R


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gguf", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--context-length", type=int, default=-1)
    p.add_argument("--prefill-chunk-size", type=int, default=512)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--prompt", type=str, default="Write one short sentence about CUDA.")
    p.add_argument("--steps", type=int, default=48)
    p.add_argument("--topn", type=int, default=12)
    p.add_argument("--no-ref", action="store_true")
    p.add_argument("--no-trace-ref-layers", action="store_true")
    p.add_argument("--no-trace-layer-diffs", action="store_true")
    return p.parse_args()


def make_config(args, meta):
    arch = meta["general.architecture"]
    context_len = args.context_length
    supported_ctx_len = meta[f"{arch}.context_length"]
    if context_len == -1:
        context_len = supported_ctx_len - args.prefill_chunk_size + 1
    return Config(
        batch_size=args.batch_size,
        context_len=context_len,
        model_arch=arch,
        gguf_path=args.gguf,
        prefill_chunk_size=args.prefill_chunk_size,
        host="127.0.0.1",
        port=0,
        seed=42,
    )


def tensor_to_torch(t, dtype):
    raw = t.to_bytes_sync()
    n = math.prod(t.shape)
    return torch.frombuffer(bytearray(raw[: n * dtype.itemsize]), dtype=dtype).reshape(tuple(t.shape))


def g4b_topk(model, t_idx: int, topn: int):
    partial_logits = tensor_to_torch(
        model.sampling_state.top_k_logits_scratchpad_B1__num_splits__top_k__dtsamp,
        torch.float32,
    )[0, 0].flatten()
    partial_idx = tensor_to_torch(
        model.sampling_state.top_k_idx_scratchpad_B1__num_splits__top_k__int32,
        torch.int32,
    )[0, 0].flatten().long()
    vals, order = torch.topk(partial_logits, min(topn, partial_logits.numel()))
    ids = partial_idx[order]
    softcap = model.lm_head.logit_softcap
    vals = softcap * torch.tanh(vals / softcap)

    # TODO rm should match closely (max abs diff < 5e-2) without this
    vals -= vals.max()

    return ids, vals


def ref_forward_ids(ref_model, ids, kv_cache, *, trace_layers: bool, collect_layer_residuals: bool):
    x, ple = ref_model.embeddings(ids)
    layer_residuals = []
    if trace_layers:
        ref_stats("embed", x)
    for i, layer in enumerate(ref_model.layers):
        x = layer(x, ple[:, :, i, :], kv_cache)
        if collect_layer_residuals:
            layer_residuals.append(x[0, -1].float().detach().clone())
        if trace_layers and (i < 3 or i % 10 == 0 or i == len(ref_model.layers) - 1):
            ref_stats(f"L{i} residual", x)
    final_hidden = x[0, -1].float()
    logits = ref_model.lm_head(x)
    if trace_layers:
        ref_stats("logits", logits)
    return logits[0, -1].float(), final_hidden, layer_residuals


def ref_forward_prefill(ref_model, ref_conf, input_ids, *, trace_layers: bool, collect_layer_residuals: bool):
    kv_cache = R.make_kv_caches(ref_conf)
    for kv in kv_cache:
        kv.set_prefill()
    ids = torch.tensor(input_ids).reshape((1, -1))
    logits, final_hidden, layer_residuals = ref_forward_ids(
        ref_model,
        ids,
        kv_cache,
        trace_layers=trace_layers,
        collect_layer_residuals=collect_layer_residuals,
    )
    for kv in kv_cache:
        kv.set_decode()
    return logits, final_hidden, layer_residuals, kv_cache


def ref_forward_decode(ref_model, kv_cache, tok: int, *, trace_layers: bool, collect_layer_residuals: bool):
    ids = torch.tensor([[tok]])
    return ref_forward_ids(
        ref_model,
        ids,
        kv_cache,
        trace_layers=trace_layers,
        collect_layer_residuals=collect_layer_residuals,
    )


def ref_stats(name, t):
    t = t.float()
    finite = torch.isfinite(t)
    if finite.any():
        vals = t[finite]
        mn, mx, am = vals.min().item(), vals.max().item(), vals.abs().mean().item()
    else:
        mn = mx = am = float("nan")
    print(f"  [ref] {name:28s} shape={list(t.shape)} min={mn:.5g} max={mx:.5g} absmean={am:.5g}")


def print_top(label, ids, vals, tokenizer, n):
    pieces = []
    for tok, val in zip(ids[:n].tolist(), vals[:n].tolist()):
        pieces.append(f"{tok}:{val:.4g}:{tokenizer.detokenize([tok])!r}")
    print(f"  {label}: " + " | ".join(pieces))


def g4b_lm_state(model):
    act = tensor_to_torch(model.lm_head.input_B1D_dtr, torch.float32)[0, 0].float()
    rsos = tensor_to_torch(model.lm_head.input_rsos_B1_dtss, torch.float32)[0, 0].float()
    logits = tensor_to_torch(model.lm_head.logits_B1V_dtsamp, torch.float32)[0, 0].float()
    softcap = model.lm_head.logit_softcap
    logits = softcap * torch.tanh(logits / softcap)
    return act, rsos, logits


def print_tensor_diff(label, a, b):
    d = (a - b).abs()
    print(
        f"  {label}: absdiff_max={float(d.max()):.5g} absdiff_mean={float(d.mean()):.5g} "
        f"a_absmax={float(a.abs().max()):.5g} b_absmax={float(b.abs().max()):.5g}"
    )


def compare_lm_state(model, ref_model, ref_hidden, ref_logits):
    g4b_act, g4b_rsos, g4b_logits = g4b_lm_state(model)
    ref_w = ref_model.lm_head.norm.w.float()
    ref_act = ref_hidden.float() * ref_w
    ref_rsos = (ref_hidden.float() * ref_hidden.float()).sum()
    print_tensor_diff("lm_input_weighted", g4b_act, ref_act)
    print(
        f"  lm_input_rsos: g4b={float(g4b_rsos):.7g} ref={float(ref_rsos):.7g} "
        f"absdiff={float((g4b_rsos - ref_rsos).abs()):.5g}"
    )
    print_tensor_diff("lm_logits_softcapped_full", g4b_logits, ref_logits.float())


def compare_layer_residuals(model, ref_layer_residuals):
    if not ref_layer_residuals:
        return
    g4b_layers = tensor_to_torch(model.debug_layer_residuals_LB1D_dtr, torch.float32)[:, 0, 0].float()
    g4b_rsos = tensor_to_torch(model.debug_layer_rsos_LB1_dtss, torch.float32)[:, 0, 0].float()
    rows = []
    for i, ref_resid in enumerate(ref_layer_residuals):
        g = g4b_layers[i]
        r = ref_resid.float()
        d = (g - r).abs()
        ref_rsos = (r * r).sum()
        rows.append(
            (
                float(d.mean()),
                float(d.max()),
                i,
                float(g.abs().max()),
                float(r.abs().max()),
                float((g4b_rsos[i] - ref_rsos).abs()),
            )
        )
    rows.sort(reverse=True)
    rendered = " | ".join(
        f"L{i}:mean={mean:.4g},max={mx:.4g},g_abs={ga:.4g},r_abs={ra:.4g},rsos_d={rsd:.4g}"
        for mean, mx, i, ga, ra, rsd in rows[:8]
    )
    by_layer = sorted(rows, key=lambda x: x[2])
    first_bad = next((i for mean, _mx, i, _ga, _ra, _rsd in by_layer if mean > 0.5), None)
    means = " ".join(f"{i}:{mean:.3g}" for mean, _mx, i, _ga, _ra, _rsd in by_layer)
    print(f"  layer_residual_worst: {rendered}")
    print(f"  layer_residual_means: {means}")
    print(f"  layer_residual_first_mean_gt_0.5: {first_bad}")


def compare_step(step, phase, g4b_ids, g4b_vals, ref_logits, sampled_tok, tokenizer, topn):
    ref_vals, ref_ids = torch.topk(ref_logits, topn)
    sampled_rank = (torch.argsort(ref_logits, descending=True) == sampled_tok).nonzero()
    rank_s = int(sampled_rank[0, 0].item()) if sampled_rank.numel() else -1

    common = []
    ref_by_id = {int(tok): float(val) for tok, val in zip(ref_ids.tolist(), ref_vals.tolist())}
    for tok, val in zip(g4b_ids.tolist(), g4b_vals.tolist()):
        tok = int(tok)
        if tok in ref_by_id:
            common.append(abs(float(val) - ref_by_id[tok]))
    max_common = max(common) if common else float("nan")
    mean_common = sum(common) / len(common) if common else float("nan")
    overlap = len(set(map(int, g4b_ids[:topn].tolist())) & set(map(int, ref_ids[:topn].tolist())))

    print(
        f"step {step:02d} phase={phase} sampled={sampled_tok}:{tokenizer.detokenize([sampled_tok])!r} "
        f"ref_argmax={int(ref_ids[0])}:{tokenizer.detokenize([int(ref_ids[0])])!r} "
        f"sampled_ref_rank={rank_s} sampled_ref_logit={float(ref_logits[sampled_tok]):.5g} "
        f"top{topn}_overlap={overlap}/{topn} common_absdiff_max={max_common:.5g} common_absdiff_mean={mean_common:.5g}"
    )
    print_top("g4b", g4b_ids, g4b_vals, tokenizer, topn)
    print_top("ref", ref_ids, ref_vals, tokenizer, topn)


def main():
    args = parse_args()
    if not args.no_trace_layer_diffs:
        os.environ["G4B_CAPTURE_LAYER_RESIDUALS"] = "1"
    device.init(args.device)
    try:
        print("loading gguf")
        meta, tensors = gguf.load(args.gguf)
        config = make_config(args, meta)
        assert config.batch_size == 1, "comparison script currently expects B=1"

        print("loading g4b model")
        model = models[config.model_arch].load(meta, tensors, config)
        scheduler = Scheduler(model)
        tokenizer = Tokenizer(config, meta)
        chat_template = ChatTemplate(config, meta)
        lifecycle.complete_phase("init")

        ref_model = ref_conf = None
        if not args.no_ref:
            print("loading reference model")
            ref_model, ref_conf, _, _, _ = R.load_model(args.gguf)
            ref_model.eval()

        text = chat_template.apply([PromptFragment(args.prompt)])
        toks = tokenizer.tokenize(text)[: config.context_len]
        if len(toks) < 2:
            toks = [tokenizer.bos, *toks, tokenizer.eos]
        print("prompt tokens:", toks)

        request = Request(toks)
        scheduler.submit(request)

        ref_next_input: int | None = None
        ref_kv = None
        for step in range(args.steps):
            scheduler.step()
            new = request.get_new_tokens()
            if not new:
                print(f"step {step:02d}: no output")
                continue
            sampled_tok = int(new[-1])
            t_idx = request._last_sample_t_idx
            g4b_ids, g4b_vals = g4b_topk(model, t_idx, args.topn)
            if args.no_ref:
                print(
                    f"step {step:02d} phase={scheduler._last_phase} "
                    f"sampled={sampled_tok}:{tokenizer.detokenize([sampled_tok])!r}"
                )
                print_top("g4b", g4b_ids, g4b_vals, tokenizer, args.topn)
            else:
                assert ref_model is not None and ref_conf is not None
                if step == 0:
                    ref_logits, ref_hidden, ref_layer_residuals, ref_kv = ref_forward_prefill(
                        ref_model,
                        ref_conf,
                        toks,
                        trace_layers=not args.no_trace_ref_layers,
                        collect_layer_residuals=not args.no_trace_layer_diffs,
                    )
                else:
                    assert ref_kv is not None
                    assert ref_next_input is not None
                    ref_logits, ref_hidden, ref_layer_residuals = ref_forward_decode(
                        ref_model,
                        ref_kv,
                        ref_next_input,
                        trace_layers=not args.no_trace_ref_layers,
                        collect_layer_residuals=not args.no_trace_layer_diffs,
                    )
                compare_step(
                    step, scheduler._last_phase, g4b_ids, g4b_vals, ref_logits, sampled_tok, tokenizer, args.topn
                )
                compare_lm_state(model, ref_model, ref_hidden, ref_logits)
                compare_layer_residuals(model, ref_layer_residuals)
            ref_next_input = sampled_tok
            if sampled_tok == tokenizer.eos:
                break
    except Exception:
        traceback.print_exc()
        raise
    finally:
        device.teardown()


if __name__ == "__main__":
    main()
