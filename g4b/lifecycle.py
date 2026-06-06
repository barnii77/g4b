import triton
import triton.knobs
import triton.runtime.autotuner
import functools
from g4b import scheduler

_phases = ["init", "warmup", "record", "deployment"]
_phase = _phases[0]


def complete_phase(expected_phase: str | None = None):
    """
    After device initialization, this function should be called once to transition to the warmup phase.
    The startup sequence then generates and submits artificial user requests and triton compiles and autotunes.
    Ending the warmup phase forbids triton from recompiling or autotuning anything beyond this point.
    In the record phase, it once again submits generated user inputs and records cuda graphs with the compiled kernels.
    After that, you advance one last time to the deployment phase, at which point the engine just launches cuda graphs.
    """
    global _phase
    assert _phase != _phases[-1]
    if expected_phase is not None and _phase != expected_phase:
        raise RuntimeError(f"Expected phase does not match actual phase: expected {expected_phase} vs actual {_phase}")
    _phase = _phases[_phases.index(_phase) + 1]
    if _phase == _phases[-1]:
        triton.knobs.runtime.jit_cache_hook = _forbid_triton_compile
        triton.runtime.autotuner.Autotuner._bench = _forbid_triton_autotune


def record_static_cuda_graph(step_fn):
    """This function assumes a static schedule for the forward pass where no kernels are launched conditionally."""
    cuda_graph = None

    @functools.wraps(step_fn)
    def wrapper(self, sched: "scheduler.Scheduler"):
        nonlocal cuda_graph
        if _phase == "warmup":
            return step_fn(self, sched)
        elif _phase == "record":
            # TODO record cuda graph
            return step_fn(self, sched)
        else:
            assert _phase == "deployment"
            # TODO launch cuda graph
            ...

    return wrapper


def _forbid_triton_compile(*_, **kwargs):
    raise RuntimeError(f"Triton compilation after warmup: {kwargs['repr']}")


def _forbid_triton_autotune(self, *_, **__):
    raise RuntimeError(f"Triton autotune bench after warmup: {self.base_fn.__name__}")
