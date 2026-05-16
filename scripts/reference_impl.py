"""
This file provides a pytorch-based reference implementation for the Gemma 4 Model Family.
It is inefficient in many ways, e.g.
    - it dequantizes everything to fp32 at load time.
    - it doesn't try to fuse any kernels.
"""

import torch
from pathlib import Path
from g4b import gguf

torch.inference_mode()


def load_model():
    meta, tensors = gguf.load(Path("/mnt/C/models/gemma-4-E4B-it-UD-Q4_K_XL.gguf"))
    check_dtypes_supported(tensors)

    model_dict: dict[str, torch.Tensor] = {}
    for t in tensors:
        tensor = convert_to_fp32_tensor(t)
        model_dict[t.name] = tensor
    del tensors

    # TODO convert to nn.Module's instead and return
    print("Model dict:")
    for k, v in model_dict.items():
        print(k, ":", None if v is None else v.std() if not v.std().isnan().item() else 0)


# TODO model impl, inference


def check_dtypes_supported(tensors):
    for t in tensors:
        if t.dtype not in (
            gguf.GGUFType.GGML_TYPE_F32,
            gguf.GGUFType.GGML_TYPE_Q4_K,
            gguf.GGUFType.GGML_TYPE_Q5_K,
            gguf.GGUFType.GGML_TYPE_Q6_K,
            gguf.GGUFType.GGML_TYPE_BF16,
        ):
            raise RuntimeError("unsupported quantization dtype", t.dtype)


# Relevant info for dequantization (source: PR introducing K-quants)
# GGML_TYPE_Q2_K - "type-1" 2-bit quantization in super-blocks containing 16 blocks, each block having 16 weight. Block scales and mins are quantized with 4 bits. This ends up effectively using 2.5625 bits per weight (bpw)
# GGML_TYPE_Q3_K - "type-0" 3-bit quantization in super-blocks containing 16 blocks, each block having 16 weights. Scales are quantized with 6 bits. This end up using 3.4375 bpw.
# GGML_TYPE_Q4_K - "type-1" 4-bit quantization in super-blocks containing 8 blocks, each block having 32 weights. Scales and mins are quantized with 6 bits. This ends up using 4.5 bpw.
# GGML_TYPE_Q5_K - "type-1" 5-bit quantization. Same super-block structure as GGML_TYPE_Q4_K resulting in 5.5 bpw
# GGML_TYPE_Q6_K - "type-0" 6-bit quantization. Super-blocks with 16 blocks, each block having 16 weights. Scales are quantized with 8 bits. This ends up using 6.5625 bpw
# GGML_TYPE_Q8_K - "type-0" 8-bit quantization. Only used for quantizing intermediate results. The difference to the existing Q8_0 is that the block size is 256. All 2-6 bit dot products are implemented for this quantization type.
#
# Relevant source lines for dequant logic: https://github.com/ggml-org/llama.cpp/blob/769cc93a43b51bf6013986180c73ee60cf24cede/gguf-py/gguf/quants.py#L479


# https://github.com/ggml-org/llama.cpp/blob/769cc93a43b51bf6013986180c73ee60cf24cede/gguf-py/gguf/quants.py#L482-L494
def dequant_q4k_to_fp32(tensor: gguf.GGUFTensor) -> torch.Tensor:
    block_size_bytes = gguf.GGUF_TYPE_BLOCK_BYTES[tensor.dtype]
    block_size_elems = gguf.GGUF_TYPE_BLOCK_ELEMENTS[tensor.dtype]

    raw = torch.frombuffer(bytearray(tensor.data), dtype=torch.uint8).reshape((-1, block_size_bytes))
    n_blocks = raw.shape[0]

    raw_scales_scale, raw_mins_scale, raw_scales_and_mins, qs = raw.split_with_sizes(
        [2, 2, 12, block_size_bytes - 16], dim=-1
    )

    dd = raw_scales_scale.view(torch.float16).to(torch.float32)
    md = raw_mins_scale.view(torch.float16).to(torch.float32)

    # bitpacked sub-block scales and mins (see llama.cpp source linked above for ascii diagram)
    frags = raw_scales_and_mins.reshape((n_blocks, 3, 4)).split(1, dim=-2)
    frags: tuple[torch.Tensor, ...] = tuple(map(lambda t: t.reshape((n_blocks, 4)), frags))
    d_frags, m_frags, mixed_frags = frags

    # unpack sub-block scales and mins
    sc = torch.cat((d_frags & 0x3F, ((d_frags & 0xC0) >> 2) | (mixed_frags & 0x0F)), dim=-1)
    mins = torch.cat((m_frags & 0x3F, ((m_frags & 0xC0) >> 2) | (mixed_frags >> 4)), dim=-1)

    ds = sc.to(torch.float32) * dd
    ms = mins.to(torch.float32) * md

    # pull high 4 bits and low 4 bits apart into 2 values (order: low then high)
    qs = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 4], dtype=torch.uint8).reshape((1, 1, 2, 1))
    qs = qs & 0x0F
    qs = qs.reshape((n_blocks, 8, 32)).to(torch.float32)

    return (ds.reshape((n_blocks, 8, 1)) * qs - ms.reshape((n_blocks, 8, 1))).reshape((n_blocks, block_size_elems))


# https://github.com/ggml-org/llama.cpp/blob/769cc93a43b51bf6013986180c73ee60cf24cede/gguf-py/gguf/quants.py#L525
def dequant_q5k_to_fp32(tensor: gguf.GGUFTensor) -> torch.Tensor:
    block_size_bytes = gguf.GGUF_TYPE_BLOCK_BYTES[tensor.dtype]
    block_size_elems = gguf.GGUF_TYPE_BLOCK_ELEMENTS[tensor.dtype]

    raw = torch.frombuffer(bytearray(tensor.data), dtype=torch.uint8).reshape((-1, block_size_bytes))
    n_blocks = raw.shape[0]

    raw_scales_scale, raw_mins_scale, raw_scales_and_mins, qh, ql = raw.split_with_sizes(
        [2, 2, 12, block_size_elems // 8, block_size_bytes - 16 - block_size_elems // 8], dim=-1
    )

    dd = raw_scales_scale.view(torch.float16).to(torch.float32)
    md = raw_mins_scale.view(torch.float16).to(torch.float32)

    # bitpacked sub-block scales and mins (see llama.cpp source linked above for ascii diagram)
    frags = raw_scales_and_mins.reshape((n_blocks, 3, 4)).split(1, dim=-2)
    frags: tuple[torch.Tensor, ...] = tuple(map(lambda t: t.reshape((n_blocks, 4)), frags))
    d_frags, m_frags, mixed_frags = frags

    # unpack sub-block scales and mins
    sc = torch.cat((d_frags & 0x3F, ((d_frags & 0xC0) >> 2) | (mixed_frags & 0x0F)), dim=-1)
    mins = torch.cat((m_frags & 0x3F, ((m_frags & 0xC0) >> 2) | (mixed_frags >> 4)), dim=-1)

    ds = sc.to(torch.float32) * dd
    ms = mins.to(torch.float32) * md

    # pull high 4 bits and low 4 bits apart into 2 values (order: low then high)
    ql = ql.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 4], dtype=torch.uint8).reshape((1, 1, 2, 1))
    ql = ql & 0x0F
    ql = ql.reshape((n_blocks, 8, 32))
    qh = qh.reshape((n_blocks, -1, 1, 32)) >> torch.arange(8, dtype=torch.uint8).reshape((1, 1, 8, 1))
    qh = qh & 0x01
    qh = qh.reshape((n_blocks, 8, 32))
    qs = ((qh << 4) | ql).to(torch.float32)

    return (ds.reshape((n_blocks, 8, 1)) * qs - ms.reshape((n_blocks, 8, 1))).reshape((n_blocks, block_size_elems))


# https://github.com/ggml-org/llama.cpp/blob/769cc93a43b51bf6013986180c73ee60cf24cede/gguf-py/gguf/quants.py#L552
def dequant_q6k_to_fp32(tensor: gguf.GGUFTensor) -> torch.Tensor:
    block_size_bytes = gguf.GGUF_TYPE_BLOCK_BYTES[tensor.dtype]
    block_size_elems = gguf.GGUF_TYPE_BLOCK_ELEMENTS[tensor.dtype]

    raw = torch.frombuffer(bytearray(tensor.data), dtype=torch.uint8).reshape((-1, block_size_bytes))
    n_blocks = raw.shape[0]

    ql, qh, raw_scales, raw_scales_scale = raw.split_with_sizes(
        [block_size_elems // 2, block_size_elems // 4, 16, 2], dim=-1
    )

    dd = raw_scales_scale.view(torch.float16).to(torch.float32)
    sc = raw_scales.view(torch.int8).to(torch.float32)
    ds = sc * dd

    # pull high 4 bits and low 4 bits apart into 2 values (order: low then high)
    ql = ql.reshape((n_blocks, -1, 1, 64)) >> torch.tensor([0, 4], dtype=torch.uint8).reshape((1, 1, 2, 1))
    ql = ql & 0x0F
    ql = ql.reshape((n_blocks, -1, 32))
    qh = qh.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 2, 4, 6], dtype=torch.uint8).reshape((1, 1, 4, 1))
    qh = qh & 0x03
    qh = qh.reshape((n_blocks, -1, 32))
    qs = ((qh << 4) | ql).to(torch.int8) - 32
    qs = qs.reshape((n_blocks, 16, 16)).to(torch.float32)

    return (ds.reshape((n_blocks, 16, 1)) * qs).reshape((n_blocks, block_size_elems))



def convert_to_fp32_tensor(tensor: gguf.GGUFTensor) -> torch.Tensor:
    dequant_handlers = {
        gguf.GGUFType.GGML_TYPE_Q4_K: dequant_q4k_to_fp32,
        gguf.GGUFType.GGML_TYPE_Q5_K: dequant_q5k_to_fp32,
        gguf.GGUFType.GGML_TYPE_Q6_K: dequant_q6k_to_fp32,
    }
    if tensor.dtype in dequant_handlers:
        out = dequant_handlers[tensor.dtype](tensor)
    elif tensor.dtype == gguf.GGUFType.GGML_TYPE_BF16:
        out = torch.frombuffer(bytearray(tensor.data), dtype=torch.bfloat16).to(torch.float32)
    else:
        assert tensor.dtype == gguf.GGUFType.GGML_TYPE_F32
        out = torch.frombuffer(bytearray(tensor.data), dtype=torch.float32)
    return out.reshape(tuple(reversed(tensor.shape)))


if __name__ == "__main__":
    load_model()
