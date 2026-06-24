import os
import argparse
import traceback
import warnings
from pathlib import Path
from g4b.config import Config
from g4b.scheduler import Scheduler
from g4b.tokenizer import Tokenizer, ChatTemplate
from g4b.models import models
from g4b import gguf, device, serve, lifecycle
from g4b.interaction_generator import (
    submit_generated_interactions,
    GuaranteePrefillFirstGenTokenDoesNotPreventDecodeWarmupGenEndingTokensProvider,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--context-length", type=int, default=-1)
    parser.add_argument("--prefill-chunk-size", type=int, default=512)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--drop-thoughts-from-history", action="store_true")
    parser.add_argument(
        "--allow-sliding-global-context",
        action="store_true",
        help="continue generation by rolling the global KV window instead of returning error code 2",
    )
    args = parser.parse_args()
    return Config(
        args.batch_size,
        args.context_length,
        "unknown",
        Path(args.gguf),
        args.prefill_chunk_size,
        args.host,
        args.port,
        args.seed,
        args.drop_thoughts_from_history,
        args.allow_sliding_global_context,
    )


def main():
    device.init(0)

    config = parse_args()
    gguf_meta, gguf_tensors = gguf.load(config.gguf_path)
    config.model_arch = gguf_meta["general.architecture"]

    supported_ctx_len = gguf_meta[f"{config.model_arch}.context_length"]
    if config.context_len == -1:
        config.context_len = supported_ctx_len - config.prefill_chunk_size + 1
    if supported_ctx_len < config.context_len:
        warnings.warn(
            f"The gguf says this model only supports {supported_ctx_len}, but you passed {config.context_len}"
        )

    tokenizer = Tokenizer(gguf_meta)
    model = models[config.model_arch].load(gguf_meta, gguf_tensors, config)
    scheduler = Scheduler(model, tokenizer)
    chat_template = ChatTemplate(gguf_meta, tokenizer)

    lifecycle.complete_phase("init")
    tokenizer._gen_ending_tokens_provider = (
        GuaranteePrefillFirstGenTokenDoesNotPreventDecodeWarmupGenEndingTokensProvider(tokenizer)
    )
    submit_generated_interactions(
        scheduler,
        tokenizer,
        chat_template,
        config.batch_size,
        max_prompt_tokens=config.prefill_chunk_size,
        max_context_len=config.context_len,
    )
    scheduler.step()  # warmup prefill graph path / autotune
    scheduler.step()  # warmup decode graph path / autotune

    lifecycle.complete_phase("warmup")
    scheduler.reset()
    submit_generated_interactions(
        scheduler,
        tokenizer,
        chat_template,
        config.batch_size,
        max_prompt_tokens=config.prefill_chunk_size,
        max_context_len=config.context_len,
    )
    scheduler.step()  # record prefill graph
    scheduler.step()  # record decode graph
    lifecycle.complete_phase("record")
    tokenizer._gen_ending_tokens_provider = None
    scheduler.reset()

    serve.register_scheduler(scheduler)
    serve.register_tokenizer(tokenizer)
    serve.register_chat_template(chat_template)
    serve.register_config(config)
    uvicorn = serve.Uvicorn.start(config.host, config.port)

    is_profiling = bool(os.environ.get("G4B_PROFILE"))

    if is_profiling:
        device.cuda_profiler_start()

    try:
        while True:
            scheduler.step()
    except Exception:
        traceback.print_exc()

    if is_profiling:
        device.cuda_profiler_stop()

    uvicorn.stop()
    device.teardown()


if __name__ == "__main__":
    main()
