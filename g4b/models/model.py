from abc import ABC, abstractmethod
from g4b.gguf import GGUFMeta, GGUFTensor
from g4b.config import Config
from g4b import scheduler


class Model(ABC):
    @abstractmethod
    def max_batch_size(self) -> int: ...

    @abstractmethod
    def max_prefill_chunk_size(self) -> int: ...

    @abstractmethod
    def stop_token_id(self) -> int: ...

    @abstractmethod
    def prepare_prefill_inputs(
        self,
        token_cols: list[list[int]],
        cache_offsets: list[int],
        time_sizes_after: list[int],
    ): ...

    @abstractmethod
    def prepare_decode_inputs(
        self,
        token_cols: list[list[int]],
        cache_offsets: list[int],
        time_sizes_after: list[int],
    ): ...

    @abstractmethod
    def collect_output_token_ids(self) -> list[int]: ...

    @abstractmethod
    def decode(self, sched: "scheduler.Scheduler"): ...

    @abstractmethod
    def prefill_chunk(self, sched: "scheduler.Scheduler"): ...

    @classmethod
    @abstractmethod
    def load(cls, meta: GGUFMeta, tensors: list[GGUFTensor], config: Config) -> Model: ...
