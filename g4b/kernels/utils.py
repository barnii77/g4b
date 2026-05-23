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
