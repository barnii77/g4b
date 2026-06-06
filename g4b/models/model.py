from abc import ABC, abstractmethod
from g4b.gguf import GGUFMeta, GGUFTensor
from g4b.config import Config
from g4b import scheduler


class Model(ABC):
    @abstractmethod
    def decode(self, sched: "scheduler.Scheduler"): ...  # TODO other params?

    @abstractmethod
    def prefill_chunk(self, sched: "scheduler.Scheduler"): ...  # TODO other params?

    @classmethod
    @abstractmethod
    def load(cls, meta: GGUFMeta, tensors: list[GGUFTensor], config: Config) -> Model: ...
