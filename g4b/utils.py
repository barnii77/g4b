import math
from cuda.bindings import runtime as cudart
from typing import Sequence


def runtime_error(msg):
    """A wrapper for `raise RuntimeError` that can be used in an expression."""
    raise RuntimeError(msg)


class CudaError(RuntimeError):
    pass


def cuda_check(err: cudart.cudaError_t | tuple):
    """Anyone who's ever used CUDA knows what this is ;)"""
    if isinstance(err, tuple):
        err = err[0]
    assert isinstance(err, cudart.cudaError_t)
    if err != cudart.cudaError_t.cudaSuccess:
        raise CudaError(cudart.cudaGetErrorString(err))


def contiguous_strides_for_shape(shape: Sequence[int]) -> list[int]:
    strides = [1]
    for s in reversed(shape[1:]):
        strides.append(strides[-1] * s)
    strides.reverse()
    return strides


def canonicalize_shape_for_size(shape: Sequence[int], size: int) -> list[int]:
    if any(s < -1 for s in shape):
        raise RuntimeError(f"invalid shape {shape}: no dims less than -1 allowed")
    if shape.count(-1) > 1:
        raise RuntimeError(f"invalid shape {shape}: too many -1")
    dim_idx = shape.index(-1)
    dim_size = to_int_exact(size / -math.prod(shape))
    return [*shape[:dim_idx], dim_size, *shape[dim_idx + 1:]]


def to_int_exact(x: int | float) -> int:
    if int(x) != x:
        raise RuntimeError("exact conversion not possible")
    return int(x)
