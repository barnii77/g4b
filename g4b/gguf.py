# TODO Currently this loader is very dumb and loads the entire model into system RAM before moving it to device.
#      This is acceptable because the models the engine is made for are very small, but it's still dumb.

import struct
import math
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Callable
from enum import IntEnum

__all__ = [
    "load",
    "GGUFType",
    "GGUFMetaType",
    "GGUFMeta",
    "GGUFTensor",
    "GGUFError",
    "GGUF_TYPE_BLOCK_BYTES",
    "GGUF_TYPE_BLOCK_ELEMENTS",
]

type GGUFMetaType = int | float | str | list[GGUFMetaType]
type GGUFMeta = dict[str, GGUFMetaType]


class GGUFError(RuntimeError):
    pass


# https://github.com/ggml-org/ggml/blob/master/docs/gguf.md#file-structure
# https://github.com/ggml-org/ggml/blob/5725fee65f5a9bee66581ac03c87aab078120e62/src/ggml.c#L624
class GGUFType(IntEnum):
    GGML_TYPE_F32 = 0
    GGML_TYPE_F16 = 1
    GGML_TYPE_Q4_0 = 2
    GGML_TYPE_Q4_1 = 3
    # GGML_TYPE_Q4_2 = 4, support has been removed
    # GGML_TYPE_Q4_3 = 5, support has been removed
    GGML_TYPE_Q5_0 = 6
    GGML_TYPE_Q5_1 = 7
    GGML_TYPE_Q8_0 = 8
    GGML_TYPE_Q8_1 = 9
    GGML_TYPE_Q2_K = 10
    GGML_TYPE_Q3_K = 11
    GGML_TYPE_Q4_K = 12
    GGML_TYPE_Q5_K = 13
    GGML_TYPE_Q6_K = 14
    GGML_TYPE_Q8_K = 15
    GGML_TYPE_IQ2_XXS = 16
    GGML_TYPE_IQ2_XS = 17
    GGML_TYPE_IQ3_XXS = 18
    GGML_TYPE_IQ1_S = 19
    GGML_TYPE_IQ4_NL = 20
    GGML_TYPE_IQ3_S = 21
    GGML_TYPE_IQ2_S = 22
    GGML_TYPE_IQ4_XS = 23
    GGML_TYPE_I8 = 24
    GGML_TYPE_I16 = 25
    GGML_TYPE_I32 = 26
    GGML_TYPE_I64 = 27
    GGML_TYPE_F64 = 28
    GGML_TYPE_IQ1_M = 29
    GGML_TYPE_BF16 = 30
    # GGML_TYPE_Q4_0_4_4 = 31, support has been removed from gguf files
    # GGML_TYPE_Q4_0_4_8 = 32
    # GGML_TYPE_Q4_0_8_8 = 33
    GGML_TYPE_TQ1_0 = 34
    GGML_TYPE_TQ2_0 = 35
    # GGML_TYPE_IQ4_NL_4_4 = 36
    # GGML_TYPE_IQ4_NL_4_8 = 37
    # GGML_TYPE_IQ4_NL_8_8 = 38
    GGML_TYPE_MXFP4 = 39  # MXFP4 (1 block)
    GGML_TYPE_NVFP4 = 40
    GGML_TYPE_Q1_0 = 41
    GGML_TYPE_COUNT = 42

    def block_bytes(self) -> int:
        assert self in GGUF_TYPE_BLOCK_BYTES
        return GGUF_TYPE_BLOCK_BYTES[self]

    def block_elements(self) -> int:
        assert self in GGUF_TYPE_BLOCK_ELEMENTS
        return GGUF_TYPE_BLOCK_ELEMENTS[self]

    def sizeof_tensor(self, shape: list[int]) -> int:
        if not shape:
            raise GGUFError("tensor has no dimensions")

        block_elements = self.block_elements()

        if shape[0] % block_elements != 0:
            raise GGUFError(
                f"{self} tensor innermost dimension {shape[0]} "
                f"is not divisible by block size {block_elements}"
            )

        rows = math.prod(shape[1:])
        return rows * (shape[0] // block_elements) * self.block_bytes()

    def __str__(self):
        return self.name.removeprefix("GGML_TYPE_").lower()

    def __repr__(self):
        return str(self)


@dataclass(frozen=True)
class GGUFTensor:
    name: str
    shape: list[int]
    dtype: GGUFType
    data: bytes

    def __repr__(self):
        return str(self)

    def __str__(self):
        preview_size = 12
        data_preview = (
            self.data[: int(preview_size // 2)].hex(" ")
            + " ... "
            + self.data[-int(preview_size // 2) :].hex(" ")
            if len(self.data) > preview_size
            else self.data.hex(" ")
        )
        return f"GGUFTensor<{self.name}, {self.shape}, {self.dtype}, '{data_preview}'>"


def load(path: Path) -> tuple[GGUFMeta, list[GGUFTensor]]:
    with path.open(mode="rb") as f:
        tensor_count, kv_count = load_header(f)
        meta = load_meta(f, kv_count)
        tensor_alignment = meta.get("general.alignment", 32)
        if not isinstance(tensor_alignment, int):
            raise GGUFError("general.alignment must be an int")
        tensors = load_tensors(f, tensor_count, tensor_alignment)
        return meta, tensors


# fmt: off
def load_int8(file: BinaryIO) -> int: return int.from_bytes(file.read(1), byteorder='little', signed=True)
def load_int16(file: BinaryIO) -> int: return int.from_bytes(file.read(2), byteorder='little', signed=True)
def load_int32(file: BinaryIO) -> int: return int.from_bytes(file.read(4), byteorder='little', signed=True)
def load_int64(file: BinaryIO) -> int: return int.from_bytes(file.read(8), byteorder='little', signed=True)
def load_uint8(file: BinaryIO) -> int: return int.from_bytes(file.read(1), byteorder='little')
def load_uint16(file: BinaryIO) -> int: return int.from_bytes(file.read(2), byteorder='little')
def load_uint32(file: BinaryIO) -> int: return int.from_bytes(file.read(4), byteorder='little')
def load_uint64(file: BinaryIO) -> int: return int.from_bytes(file.read(8), byteorder='little')
def load_bool(file: BinaryIO) -> bool: return bool(load_int8(file))
def load_float32(file: BinaryIO) -> float: return struct.unpack('<f', file.read(4))[0]
def load_float64(file: BinaryIO) -> float: return struct.unpack('<d', file.read(8))[0]
# fmt: on


def load_str(file: BinaryIO) -> str:
    length = load_uint64(file)
    return file.read(length).decode()


def load_array(file: BinaryIO) -> list[GGUFMetaType]:
    element_type_id = load_uint32(file)
    loader = get_loader(element_type_id)
    length = load_uint64(file)
    return [loader(file) for _ in range(length)]


def get_loader(type_id: int) -> Callable[[BinaryIO], GGUFMetaType]:
    # fmt: off
    loaders: list[Callable] = [load_uint8, load_int8, load_uint16, load_int16, load_uint32, load_int32, load_float32,
                               load_bool, load_str, load_array, load_uint64, load_int64, load_float64]
    # fmt: on
    if type_id >= len(loaders):
        raise GGUFError(f"unknown type id {type_id}")
    return loaders[type_id]


def load_gguf_type(file: BinaryIO) -> GGUFMetaType:
    type_id = load_uint32(file)
    return get_loader(type_id)(file)


def load_header(file: BinaryIO) -> tuple[int, int]:
    gguf_magic, gguf_version = file.read(4), load_uint32(file)
    if gguf_magic != b"GGUF" or gguf_version not in (2, 3):
        raise GGUFError("invalid header")
    tensor_count, kv_count = load_uint64(file), load_uint64(file)
    return tensor_count, kv_count


def load_meta(file: BinaryIO, kv_count: int) -> GGUFMeta:
    meta: GGUFMeta = {}
    for _ in range(kv_count):
        k = load_str(file)
        v = load_gguf_type(file)
        meta[k] = v
    return meta


def load_tensors(file: BinaryIO, tensor_count: int, tensor_alignment: int) -> list[GGUFTensor]:
    tensors_pre_data_load: list[tuple[str, list[int], GGUFType, int]] = []
    for _ in range(tensor_count):
        name = load_str(file)
        n_dims = load_uint32(file)
        shape = [load_uint64(file) for _ in range(n_dims)]
        dtype = GGUFType(load_uint32(file))
        data_offset = load_uint64(file)
        tensors_pre_data_load.append((name, shape, dtype, data_offset))

    end_of_meta = file.tell()
    tensor_blob_offset = math.ceil(end_of_meta / tensor_alignment) * tensor_alignment

    tensors: list[GGUFTensor] = []
    for name, shape, dtype, data_offset in tensors_pre_data_load:
        file.seek(tensor_blob_offset + data_offset)
        n_bytes = dtype.sizeof_tensor(shape)
        data = file.read(n_bytes)
        tensors.append(GGUFTensor(name, shape, dtype, data))

    return tensors


# fmt: off
GGUF_TYPE_BLOCK_BYTES: dict[GGUFType, int] = {
    GGUFType.GGML_TYPE_F32: 4,      GGUFType.GGML_TYPE_F16: 2,      GGUFType.GGML_TYPE_BF16: 2,     GGUFType.GGML_TYPE_F64: 8,
    GGUFType.GGML_TYPE_I8: 1,       GGUFType.GGML_TYPE_I16: 2,      GGUFType.GGML_TYPE_I32: 4,      GGUFType.GGML_TYPE_I64: 8,

    GGUFType.GGML_TYPE_Q1_0: 18,    GGUFType.GGML_TYPE_Q4_0: 18,    GGUFType.GGML_TYPE_Q4_1: 20,
    GGUFType.GGML_TYPE_Q5_0: 22,    GGUFType.GGML_TYPE_Q5_1: 24,
    GGUFType.GGML_TYPE_Q8_0: 34,    GGUFType.GGML_TYPE_Q8_1: 36,

    GGUFType.GGML_TYPE_Q2_K: 84,    GGUFType.GGML_TYPE_Q3_K: 110,   GGUFType.GGML_TYPE_Q4_K: 144,   GGUFType.GGML_TYPE_Q5_K: 176,
    GGUFType.GGML_TYPE_Q6_K: 210,   GGUFType.GGML_TYPE_Q8_K: 292,

    GGUFType.GGML_TYPE_IQ1_S: 50,   GGUFType.GGML_TYPE_IQ1_M: 56,   GGUFType.GGML_TYPE_IQ2_XXS: 66, GGUFType.GGML_TYPE_IQ2_XS: 74,
    GGUFType.GGML_TYPE_IQ2_S: 82,   GGUFType.GGML_TYPE_IQ3_XXS: 98, GGUFType.GGML_TYPE_IQ3_S: 110,
    GGUFType.GGML_TYPE_IQ4_NL: 18,  GGUFType.GGML_TYPE_IQ4_XS: 136,

    GGUFType.GGML_TYPE_TQ1_0: 54,   GGUFType.GGML_TYPE_TQ2_0: 66,
    GGUFType.GGML_TYPE_MXFP4: 17,   GGUFType.GGML_TYPE_NVFP4: 36,
}
GGUF_TYPE_BLOCK_ELEMENTS: dict[GGUFType, int] = {
    GGUFType.GGML_TYPE_F32: 1,      GGUFType.GGML_TYPE_F16: 1,      GGUFType.GGML_TYPE_BF16: 1,     GGUFType.GGML_TYPE_F64: 1,
    GGUFType.GGML_TYPE_I8: 1,       GGUFType.GGML_TYPE_I16: 1,      GGUFType.GGML_TYPE_I32: 1,      GGUFType.GGML_TYPE_I64: 1,

    GGUFType.GGML_TYPE_Q1_0: 128,   GGUFType.GGML_TYPE_Q4_0: 32,    GGUFType.GGML_TYPE_Q4_1: 32,
    GGUFType.GGML_TYPE_Q5_0: 32,    GGUFType.GGML_TYPE_Q5_1: 32,
    GGUFType.GGML_TYPE_Q8_0: 32,    GGUFType.GGML_TYPE_Q8_1: 32,

    GGUFType.GGML_TYPE_Q2_K: 256,   GGUFType.GGML_TYPE_Q3_K: 256,   GGUFType.GGML_TYPE_Q4_K: 256,   GGUFType.GGML_TYPE_Q5_K: 256,
    GGUFType.GGML_TYPE_Q6_K: 256,   GGUFType.GGML_TYPE_Q8_K: 256,

    GGUFType.GGML_TYPE_IQ1_S: 256,  GGUFType.GGML_TYPE_IQ1_M: 256,  GGUFType.GGML_TYPE_IQ2_XXS: 256, GGUFType.GGML_TYPE_IQ2_XS: 256,
    GGUFType.GGML_TYPE_IQ2_S: 256,  GGUFType.GGML_TYPE_IQ3_XXS: 256, GGUFType.GGML_TYPE_IQ3_S: 256,
    GGUFType.GGML_TYPE_IQ4_NL: 32,  GGUFType.GGML_TYPE_IQ4_XS: 256,

    GGUFType.GGML_TYPE_TQ1_0: 256,  GGUFType.GGML_TYPE_TQ2_0: 256,
    GGUFType.GGML_TYPE_MXFP4: 32,   GGUFType.GGML_TYPE_NVFP4: 64,
}
# fmt: on
