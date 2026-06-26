import math
import logging
import shutil
import subprocess
import ctypes
import tempfile
from pathlib import Path
from functools import cache
from cuda.bindings import runtime as cudart
from typing import Sequence
from g4b.gguf import GGUFTensor


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
    if -1 not in shape:
        if math.prod(shape) != size:
            raise RuntimeError(f"invalid shape {shape}: size mismatch with {size}")
        return list(shape)
    dim_idx = shape.index(-1)
    dim_size = to_int_exact(size / -math.prod(shape))
    return [*shape[:dim_idx], dim_size, *shape[dim_idx + 1:]]


def to_int_exact(x: int | float) -> int:
    if int(x) != x:
        raise RuntimeError("exact conversion not possible")
    return int(x)


def gguf_shape_from_conventional(conventional_shape: Sequence[int]) -> list[int]:
    return list(reversed(conventional_shape))


def conventional_shape_from_gguf(gguf_shape: Sequence[int]) -> list[int]:
    return list(reversed(gguf_shape))


def gguf_tensors_by_name(tensors: list[GGUFTensor]) -> dict[str, GGUFTensor]:
    out = {}
    for tensor in tensors:
        out[tensor.name] = tensor
    return out


def create_file_logger(path: str | Path, level: int = logging.INFO) -> logging.Logger:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("g4b")
    logger.setLevel(level)

    # Avoid duplicate handlers
    if not any(isinstance(h, logging.FileHandler) and Path(h.baseFilename) == path.resolve() for h in logger.handlers):
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s [%(threadName)s] %(levelname)s: %(message)s"))
        logger.addHandler(handler)

    return logger


def shared_prefix_length(a: list, b: list) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def floor_to_multiple_of(x: int, m: int) -> int:
    return x // m * m


def get_cpp_compiler_path():
    return (
        shutil.which("c++")
        or shutil.which("g++")
        or shutil.which("clang++")
        or runtime_error("C++ compiler missing")
    )


def compile_and_load_cpp(src: Path) -> ctypes.CDLL:
    if not src.is_file():
        raise RuntimeError("src must reference file")
    cc = get_cpp_compiler_path()
    dest = get_temp_dir() / src.name
    cmd = [cc, src, "-O3", "-shared", "-fPIC", "-o", dest]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL)
    return ctypes.cdll.LoadLibrary(str(dest))


@cache
def get_temp_dir() -> Path:
    return Path(tempfile.gettempdir())
