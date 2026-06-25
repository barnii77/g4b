from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    batch_size: int
    context_len: int
    model_arch: str
    gguf_path: Path
    prefill_chunk_size: int
    host: str
    port: int
    seed: int
    allow_sliding_global_context: bool = False
