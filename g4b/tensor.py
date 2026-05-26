import ctypes
import math
from triton import language as tl
from triton.experimental.gluon import language as gl
from cuda.core import Buffer, Event
from cuda.bindings import runtime as cudart
from dataclasses import dataclass
from typing import Sequence
from g4b import device
from g4b.utils import cuda_check, contiguous_strides_for_shape, canonicalize_shape_for_size, to_int_exact


@dataclass(frozen=True)
class DType:
    name: str
    storage: str

    @property
    def tl_dtype(self) -> tl.dtype:
        return getattr(tl, self.storage)

    @property
    def gl_dtype(self) -> gl.dtype:
        return getattr(gl, self.storage)


int8 = DType("int8", "int8")
uint8 = DType("uint8", "uint8")
int16 = DType("int16", "int16")
uint16 = DType("uint16", "uint16")
int32 = DType("int32", "int32")
uint32 = DType("uint32", "uint32")
float8e5 = DType("fp8e5", "fp8e5")
float8e5b16 = DType("fp8e5b16", "fp8e5b16")
float8e4nv = DType("fp8e4nv", "fp8e4nv")
float8e4b8 = DType("fp8e4b8", "fp8e4b8")
float8e4b15 = DType("fp8e4b15", "fp8e4b15")
float16 = DType("float16", "float16")
bfloat16 = DType("bfloat16", "bfloat16")
float32 = DType("float32", "float32")
q4_k = DType("q4_k", "uint8")
q5_k = DType("q5_k", "uint8")
q6_k = DType("q6_k", "uint8")


# TODO reshape, transpose, permute, view(dtype) methods... but no methods that would require kernel launches
@dataclass(frozen=True)
class Tensor:
    buffer: Buffer
    dtype: DType
    shape: Sequence[int]
    stride: Sequence[int]

    def data_ptr(self) -> int:
        return int(self.buffer.handle)

    @classmethod
    def from_bytes_sync(cls, data: bytes, dtype: DType, shape: Sequence[int], strides: Sequence[int] | None = None):
        if strides is None:
            strides = contiguous_strides_for_shape(shape)
        buf = device.alloc(len(data))
        _copy_htod_sync(buf, data)
        return cls(buf, dtype, shape, strides)

    def to_bytes_sync(self) -> bytes:
        return _copy_dtoh_sync(self.buffer)

    def copy_to(self, dst: Buffer, event: Event):
        self.buffer.copy_to(dst, stream=device.stream)
        return device.stream.record(event)

    def is_contiguous(self) -> bool:
        return self.stride == contiguous_strides_for_shape(self.shape)

    def reshape(self, shape: Sequence[int]) -> Tensor:
        # TODO validate if this reshape is actually possible given the strides and update strides properly
        assert self.is_contiguous(), "reshape of non-contiguous tensor unsupported"
        shape = canonicalize_shape_for_size(shape, math.prod(self.shape))
        return Tensor(
            self.buffer,
            self.dtype,
            shape,
            contiguous_strides_for_shape(shape),
        )

    def view(self, dtype: DType) -> Tensor:
        # attempts to resize last dim, fails if shape[-1] or stride[-1] not divisible to cleanly fit new dtype
        size_ratio = self.dtype.tl_dtype.itemsize / dtype.tl_dtype.itemsize
        return Tensor(
            self.buffer,
            dtype,
            (*self.shape[:-1], to_int_exact(self.shape[-1] * size_ratio)),
            (*self.stride[:-1], to_int_exact(self.stride[-1] * size_ratio)),
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
