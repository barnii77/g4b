from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    batch_size: int
    context_len: int  # TODO emit warning if gguf context len < this value
    ...  # TODO
