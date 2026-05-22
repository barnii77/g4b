import ctypes
from triton import language as tl
from triton.experimental.gluon import language as gl
from cuda.core import Buffer
from cuda.bindings import runtime as cudart
from dataclasses import dataclass
from typing import Sequence
from g4b import device
from g4b.utils import cuda_check


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
int32 = DType("int32", "int32")
uint32 = DType("uint32", "uint32")
bfloat16 = DType("bfloat16", "bfloat16")
float32 = DType("float32", "float32")
q4_k = DType("q4_k", "uint8")
q5_k = DType("q5_k", "uint8")
q6_k = DType("q6_k", "uint8")


@dataclass(frozen=True)
class Tensor:
    buffer: Buffer
    type: DType
    shape: Sequence[int]
    stride: Sequence[int]

    def data_ptr(self) -> int:
        return int(self.buffer.handle)

    @property
    def dtype(self) -> tl.dtype:
        return self.type.tl_dtype

    @classmethod
    def from_bytes(cls, data: bytes, dtype: DType, shape: Sequence[int], strides: Sequence[int] | None = None):
        if strides is None:
            strides = [1]
            for s in reversed(shape[1:]):
                strides.append(strides[-1] * s)
            strides.reverse()
        buf = device.alloc(len(data))
        _copy_htod_sync(buf, data, device.side_stream)
        return cls(buf, dtype, shape, strides)

    def to_bytes(self) -> bytes:
        return _copy_dtoh_sync(self.buffer, device.side_stream)


def _copy_htod_sync(dst, data: bytes | bytearray | memoryview, stream):
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


def _copy_dtoh_sync(src, stream) -> bytes:
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
