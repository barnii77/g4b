import math
import triton
from triton import language as tl
from triton.runtime import Autotuner, Heuristics
from triton.experimental.gluon import language as gl
from g4b import tensor


def _is_tensor_like(t):
    return hasattr(t, "shape") and hasattr(t, "stride") and hasattr(t, "dtype") and hasattr(t, "data_ptr")


def _normalize_strides(t) -> list[int]:
    if isinstance(t.stride, (list, tuple)):
        return list(t.stride)
    elif callable(t.stride):
        return [t.stride(i) for i in range(len(t.shape))]
    raise RuntimeError("Failed to normalize tensor stride into list[int]")


class _TritonTensorAdapter:
    def __init__(self, t: "tensor.Tensor"):
        self.tensor = t

    def data_ptr(self) -> int:
        return self.tensor.data_ptr()

    @property
    def dtype(self) -> tl.dtype:
        return self.tensor.dtype.tl_dtype


class _GluonTensorAdapter:
    def __init__(self, t: "tensor.Tensor"):
        self.tensor = t

    def data_ptr(self) -> int:
        return self.tensor.data_ptr()

    @property
    def dtype(self) -> gl.dtype:
        return self.tensor.dtype.gl_dtype


def _unpack_tensors_for_kernel(kernel, kwargs):
    """
    Unpack all tensor arguments {k}={v} into:
    - {k}_ptr={gluon_adapter(v) | triton_adapter(v)} (which itself is auto converted by triton to pointer type),
    - foreach dim N:
        - {k}_shape{N}
        - {k}_stride{N}
    The other arguments are not touched.

    This function makes it much more convenient to call kernels because you don't have to write lots of boilerplate to
    decompose multiple 4D tensors into their individual shape and stride components.
    """

    unpacked = {}
    is_gluon = getattr(kernel, "is_gluon", lambda: False)()

    def add_if_needed(k, v):
        unwrapped_kernel = kernel
        if isinstance(unwrapped_kernel, Autotuner):
            unwrapped_kernel = kernel.fn
        if isinstance(kernel, Heuristics):
            unwrapped_kernel = unwrapped_kernel.fn
        if k in unwrapped_kernel.signature.parameters:
            unpacked[k] = v

    for k, v in kwargs.items():
        # Check _is_tensor_like(v) to allow torch Tensors as well for testing
        if (is_g4b_tensor := isinstance(v, tensor.Tensor)) or _is_tensor_like(v):
            # unpack tensors into {k}_ptr, {k}_stride{N}, and {k}_shape{N}
            v_ptr = _GluonTensorAdapter(v) if is_gluon else _TritonTensorAdapter(v) if is_g4b_tensor else v
            add_if_needed(f"{k}_ptr", v_ptr)
            stride = _normalize_strides(v)
            assert len(v.shape) == len(stride)
            for dim, (shape, stride) in enumerate(zip(v.shape, stride)):
                add_if_needed(f"{k}_shape{dim}", shape)
                add_if_needed(f"{k}_stride{dim}", stride)
        else:
            # pass through other args unchanged
            unpacked[k] = v

    return unpacked


class _ConfiguredLaunch:
    def __init__(self, kernel, grid):
        self.kernel = kernel
        self.grid = grid

    def __call__(self, *args, **kwargs):
        if args != ():
            raise RuntimeError("only kwargs allowed")
        kwargs = _unpack_tensors_for_kernel(self.kernel, kwargs)
        return self.kernel[self.grid](**kwargs)


class _Launch:
    def __getitem__(self, item):
        if not isinstance(item, tuple) or len(item) != 2:
            raise RuntimeError("Usage: launch[kernel, grid](...)")
        return _ConfiguredLaunch(*item)


launch = _Launch()


# Fuck triton autotuner, it needs torch and we don't want torch. So we roll our own, hell yeah.
def default_bencher(fn, quantiles):
    """Benchmark Triton launches on the same cuda.core stream that g4b gives Triton."""
    from cuda.core import EventOptions
    import g4b.device

    warmup_ms = 300
    rep_ms = 500

    def time_n(n: int) -> float:
        start = g4b.device.device.create_event(options=EventOptions(timing_enabled=True))
        end = g4b.device.device.create_event(options=EventOptions(timing_enabled=True))
        g4b.device.stream.record(start)
        for _ in range(n):
            fn()
        g4b.device.stream.record(end)
        end.sync()
        return end - start

    fn()
    g4b.device.stream.sync()

    estimate_ms = max(time_n(5) / 5, 1e-6)
    n_warmup = max(1, int(warmup_ms / estimate_ms))
    n_repeat = max(1, int(rep_ms / estimate_ms))

    for _ in range(n_warmup):
        fn()
    g4b.device.stream.sync()

    n_batches = min(20, n_repeat)
    n_per_batch = max(1, math.ceil(n_repeat / n_batches))
    times = [time_n(n_per_batch) / n_per_batch for _ in range(n_batches)]

    if quantiles is not None:
        return [_quantile(times, q) for q in quantiles]
    return sum(times) / len(times)


def _quantile(values, q: float):
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = q * (len(values) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[lo]
    frac = pos - lo
    return values[lo] * (1 - frac) + values[hi] * frac
