import argparse
import traceback
import warnings
from pathlib import Path
from g4b.config import Config
from g4b.scheduler import Scheduler
from g4b.tokenizer import Tokenizer, ChatTemplate
from g4b.models import models
from g4b import gguf, device, serve, lifecycle
from g4b.interaction_generator import submit_generated_interactions


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--context-length", type=int, default=-1)
    parser.add_argument("--prefill-chunk-size", type=int, default=512)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    return Config(
        args.batch_size,
        args.context_length,
        "unknown",
        Path(args.gguf),
        args.prefill_chunk_size,
        args.host,
        args.port,
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

    model = models[config.model_arch].load(gguf_meta, gguf_tensors, config)
    scheduler = Scheduler(model)
    tokenizer = Tokenizer(config, gguf_meta)
    chat_template = ChatTemplate(config, gguf_meta)

    lifecycle.complete_phase("init")
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
    scheduler.reset()

    serve.register_scheduler(scheduler)
    serve.register_tokenizer(tokenizer)
    serve.register_chat_template(chat_template)
    serve.register_max_ctx_len(config.context_len)
    uvicorn = serve.Uvicorn.start(config.host, config.port)

    try:
        while True:
            scheduler.step()
    except Exception:
        traceback.print_exc()

    uvicorn.stop()
    device.teardown()


if __name__ == "__main__":
    main()
