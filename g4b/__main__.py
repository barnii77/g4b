import argparse
import warnings
from pathlib import Path
from g4b.config import Config
from g4b.scheduler import Scheduler
from g4b.models import models
from g4b import gguf, device


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--context-length", type=int, default=-1)
    parser.add_argument("--prefill-chunk-size", type=int, default=512)
    args = parser.parse_args()
    return Config(args.batch_size, args.context_length, "unknown", Path(args.gguf), args.prefill_chunk_size)


def main():
    device.init(0)

    config = parse_args()
    gguf_meta, gguf_tensors = gguf.load(config.gguf_path)
    config.model_arch = gguf_meta["general.architecture"]

    supported_ctx_len = gguf_meta[f"{config.model_arch}.context_length"]
    if config.context_len == -1:
        config.context_len = supported_ctx_len - config.prefill_chunk_size + 1
    if supported_ctx_len < config.context_len:
        warnings.warn(f"The gguf says this model only supports {supported_ctx_len}, but you passed {config.context_len}")

    model = models[config.model_arch].load(gguf_meta, gguf_tensors, config)
    scheduler = Scheduler(model)
    # TODO run http server

    device.teardown()


if __name__ == "__main__":
    main()
