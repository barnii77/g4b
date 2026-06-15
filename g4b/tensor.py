import ctypes
import math
from triton import language as tl
from triton.experimental.gluon import language as gl
from cuda.core import Buffer, Event
from cuda.bindings import runtime as cudart
from dataclasses import dataclass
from typing import Sequence
from g4b import device
from g4b.gguf import GGUFType, GGUFTensor
from g4b.utils import (
    cuda_check,
    contiguous_strides_for_shape,
    canonicalize_shape_for_size,
    to_int_exact,
    gguf_shape_from_conventional,
    conventional_shape_from_gguf,
)


@dataclass(frozen=True)
class DType:
    name: str
    storage: str
    gguf_dtype: GGUFType | None
    _bytes_per_elem: float | int | None = None  # fallback for when gguf_dtype does not exist

    @property
    def tl_dtype(self) -> tl.dtype:
        return getattr(tl, self.storage)

    @property
    def gl_dtype(self) -> gl.dtype:
        return getattr(gl, self.storage)

    def sizeof_tensor(self, shape: Sequence[int]) -> int:
        if self.gguf_dtype is not None:
            return self.gguf_dtype.sizeof_tensor(gguf_shape_from_conventional(shape))
        else:
            assert self._bytes_per_elem is not None
            return round(self._bytes_per_elem * math.prod(shape))


int8 = DType("int8", "int8", GGUFType.GGML_TYPE_I8)
uint8 = DType("uint8", "uint8", None, 1)
int16 = DType("int16", "int16", GGUFType.GGML_TYPE_I16)
uint16 = DType("uint16", "uint16", None, 2)
int32 = DType("int32", "int32", GGUFType.GGML_TYPE_I32)
uint32 = DType("uint32", "uint32", None, 4)
float8e5 = DType("fp8e5", "fp8e5", None, 1)
float8e5b16 = DType("fp8e5b16", "fp8e5b16", None, 1)
float8e4nv = DType("fp8e4nv", "fp8e4nv", None, 1)
float8e4b8 = DType("fp8e4b8", "fp8e4b8", None, 1)
float8e4b15 = DType("fp8e4b15", "fp8e4b15", None, 1)
float16 = DType("float16", "float16", GGUFType.GGML_TYPE_F16)
bfloat16 = DType("bfloat16", "bfloat16", GGUFType.GGML_TYPE_BF16)
float32 = DType("float32", "float32", GGUFType.GGML_TYPE_F32)
q4_k = DType("q4_k", "int8", GGUFType.GGML_TYPE_Q4_K)
q5_k = DType("q5_k", "int8", GGUFType.GGML_TYPE_Q5_K)
q6_k = DType("q6_k", "int8", GGUFType.GGML_TYPE_Q6_K)
dtypes = [
    # fmt: off
    int8, uint8, int16, uint16, int32, uint32,
    float8e5, float8e5b16, float8e4nv, float8e4b8, float8e4b15,
    float16, bfloat16, float32,
    q4_k, q5_k, q6_k,
    # fmt: on
]


@dataclass(frozen=True)
class Tensor:
    buffer: Buffer
    dtype: DType
    shape: Sequence[int]
    stride: Sequence[int]
    _base_ptr_byte_offset: int = 0

    def data_ptr(self) -> int:
        return int(self.buffer.handle + self._base_ptr_byte_offset)

    @classmethod
    def from_bytes_sync(cls, data: bytes, dtype: DType, shape: Sequence[int], strides: Sequence[int] | None = None):
        if strides is None:
            strides = contiguous_strides_for_shape(shape)
            if _is_quantized_dtype(dtype):
                strides = _storage_based_strides_from_q_elem_strides(strides, dtype)
        buf = device.alloc(len(data))
        _copy_htod_sync(buf, data)
        return cls(buf, dtype, shape, strides)

    @classmethod
    def from_gguf_tensor(cls, gguf_tensor: GGUFTensor):
        return cls.from_bytes_sync(
            gguf_tensor.data,
            _dtype_from(gguf_tensor.dtype),
            conventional_shape_from_gguf(gguf_tensor.shape),
        )

    @classmethod
    def alloc_empty(cls, dtype: DType, shape: Sequence[int], strides: Sequence[int] | None = None):
        if strides is None:
            strides = contiguous_strides_for_shape(shape)
            if _is_quantized_dtype(dtype):
                strides = _storage_based_strides_from_q_elem_strides(strides, dtype)
        size_in_bytes = dtype.sizeof_tensor(shape)
        buf = device.alloc(size_in_bytes)
        return cls(buf, dtype, shape, strides)

    def to_bytes_sync(self) -> bytes:
        if self._base_ptr_byte_offset != 0:
            raise RuntimeError("unsupported for slice_at tensors")
        return _copy_dtoh_sync(self.buffer)  # TODO use tensor shape and dtype size to transfer only relevant chunk

    def copy_from_bytes_sync(self, data: bytes | bytearray | memoryview):
        if self._base_ptr_byte_offset != 0:
            raise RuntimeError("unsupported for slice_at tensors")
        _copy_htod_sync(self.buffer, data)

    def copy_to(self, dst: Buffer, event: Event):
        # TODO replace with cuda runtime api calls so I can support self._base_ptr_byte_offset != 0
        if self._base_ptr_byte_offset != 0:
            raise RuntimeError("unsupported for slice_at tensors")
        self.buffer.copy_to(dst, stream=device.stream)
        return device.stream.record(event)

    def is_contiguous(self) -> bool:
        expected = contiguous_strides_for_shape(self.shape)
        if _is_quantized_dtype(self.dtype):
            expected = _storage_based_strides_from_q_elem_strides(expected, self.dtype)
        return list(self.stride) == expected

    def reshape(self, shape: Sequence[int]) -> Tensor:
        # TODO validate if this reshape is actually possible given the strides and update strides properly
        assert self.is_contiguous(), "reshape of non-contiguous tensor unsupported"
        shape = canonicalize_shape_for_size(shape, math.prod(self.shape))
        strides = contiguous_strides_for_shape(shape)
        if _is_quantized_dtype(self.dtype):
            strides = _storage_based_strides_from_q_elem_strides(strides, self.dtype)
        return Tensor(self.buffer, self.dtype, shape, strides, self._base_ptr_byte_offset)

    def view(self, dtype: DType) -> Tensor:
        # attempts to resize last dim, fails if shape[-1] or stride[-1] not divisible to cleanly fit new dtype
        size_ratio = self.dtype.tl_dtype.itemsize / dtype.tl_dtype.itemsize
        return Tensor(
            self.buffer,
            dtype,
            [*self.shape[:-1], to_int_exact(self.shape[-1] * size_ratio)],
            [*self.stride[:-1], to_int_exact(self.stride[-1] * size_ratio)],
            self._base_ptr_byte_offset,
        )

    def permute(self, dims: Sequence[int]) -> Tensor:
        dims = list(dims)
        if sorted(dims) != list(range(len(self.shape))):
            raise RuntimeError(f"incomplete dim list {dims}")
        return Tensor(
            self.buffer,
            self.dtype,
            [self.shape[i] for i in dims],
            [self.stride[i] for i in dims],
            self._base_ptr_byte_offset,
        )

    def transpose(self, dim1: int, dim2: int) -> Tensor:
        permute_seq = list(range(len(self.shape)))
        permute_seq[dim1] = dim2
        permute_seq[dim2] = dim1
        return self.permute(permute_seq)

    def slice_until(self, dim: int, end: int):
        assert len(self.shape) > dim
        assert self.shape[dim] >= end
        new_shape = [end if i == dim else s for i, s in enumerate(self.shape)]
        return Tensor(self.buffer, self.dtype, new_shape, self.stride, self._base_ptr_byte_offset)

    def slice_at(self, dim: int, idx: int):
        if dim >= len(self.shape) or idx >= self.shape[dim]:
            raise RuntimeError("illegal dim or idx")
        if _is_quantized_dtype(self.dtype):
            raise RuntimeError("cannot slice_at into quantized tensor")
        return Tensor(
            self.buffer,
            self.dtype,
            list(self.shape[:dim]) + list(self.shape[dim + 1 :]),
            list(self.stride[:dim]) + list(self.stride[dim + 1 :]),
            self._base_ptr_byte_offset + self.stride[dim] * idx * self.dtype.tl_dtype.itemsize,
        )


def _copy_htod_sync(dst, data: bytes | bytearray | memoryview):
    stream = device.stream
    data = bytes(data)

    n = len(data)
    if n > dst.size:
        raise ValueError(f"source too large: {n} bytes > dst.size={dst.size}")

    copy_src = ctypes.create_string_buffer(data)
    cuda_check(
        cudart.cudaMemcpyAsync(
            int(dst.handle),
            ctypes.addressof(copy_src),
            n,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            int(stream.handle),
        )
    )
    stream.sync()


def _copy_dtoh_sync(src) -> bytes:
    stream = device.stream
    copy_dst = ctypes.create_string_buffer(src.size)
    cuda_check(
        cudart.cudaMemcpyAsync(
            ctypes.addressof(copy_dst),
            int(src.handle),
            src.size,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            int(stream.handle),
        )
    )
    stream.sync()
    return bytes(copy_dst)


def _dtype_from(gguf_dtype: GGUFType) -> DType:
    for dtype in dtypes:
        if dtype.gguf_dtype == gguf_dtype:
            return dtype
    raise RuntimeError("dtype not found")


# Tensors with quantized dtypes will get auto-computed strides in terms of their logical type,
#  but my kernels require strides in terms of the storage type (i.e. uint8 for q4_k etc.).
def _storage_based_strides_from_q_elem_strides(strides: Sequence[int], logical_dtype: DType) -> list[int]:
    assert logical_dtype.gguf_dtype is not None
    gguf_type = logical_dtype.gguf_dtype
    assert gguf_type.block_bytes() < gguf_type.block_elements(), "not a quantized dtype"
    scaling_factor = gguf_type.block_bytes() / gguf_type.block_elements()
    return [round(s * scaling_factor) if i != len(strides) - 1 else s for i, s in enumerate(strides)]


def _is_quantized_dtype(dtype: DType):
    return dtype.name != dtype.storage  # Is this property stable?
