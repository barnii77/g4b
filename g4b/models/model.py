import functools
from abc import ABC, abstractmethod
from g4b.gguf import GGUFMeta, GGUFTensor
from g4b.config import Config
from g4b import scheduler


class Model(ABC):
    @classmethod
    @abstractmethod
    def load(cls, gguf_meta: GGUFMeta, gguf_tensors: list[GGUFTensor], config: Config): ...

    @abstractmethod
    def prefill_chunk(self, sched: "scheduler.Scheduler"): ...  # TODO other params?

    @abstractmethod
    def decode(self, sched: "scheduler.Scheduler"): ...  # TODO other params?


# TODO this assumes a static schedule for the forward pass where no kernels are launched conditionally...
#  ensure my use-case actually matches this.
def record_static_cuda_graph(step_fn):
    """This function assumes a static schedule for the forward pass where no kernels are launched conditionally."""
    # first call does not record cuda graph because triton kernels need to compile.
    # second call then records cuda graph.
    # subsequent calls use the cuda graph instead of the normal method.

    has_compiled = False
    cuda_graph = None

    @functools.wraps(step_fn)
    def wrapper(self, sched: "scheduler.Scheduler"):
        nonlocal has_compiled, cuda_graph
        if not has_compiled:
            has_compiled = True
            return step_fn(self, sched)
        elif cuda_graph is None:
            # TODO record cuda graph
            return step_fn(self, sched)
        else:
            # TODO launch cuda graph
            ...

    return wrapper
