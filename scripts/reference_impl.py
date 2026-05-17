"""
This file provides a pytorch-based reference implementation for the Gemma 4 Model Family.
It is inefficient in many ways, e.g.
    - it dequantizes everything to fp32 at load time.
    - it doesn't try to fuse any kernels.
    - it doesn't use any torch.nn prebuilt modules (except nn.Linear)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from dataclasses import dataclass
from g4b import gguf


@dataclass
class Gemma4Config:
    block_count: int
    context_length: int
    embedding_length: int
    feed_forward_length: int
    attention_head_count: int
    attention_head_count_kv: int
    rope_freq_base: float
    rope_freq_base_swa: float
    attention_layer_norm_rms_epsilon: float
    attention_key_length: int
    attention_value_length: int
    final_logit_softcapping: float
    attention_sliding_window: int
    attention_shared_kv_layers: int
    embedding_length_per_layer_input: int
    attention_sliding_window_pattern: list[bool]
    attention_key_length_swa: int
    attention_value_length_swa: int
    rope_dimension_count: int
    rope_dimension_count_swa: int


@dataclass
class SamplingConfig:
    top_k: int
    top_p: float
    temperature: float


@dataclass
class TokenizerConfig:
    model: str
    tokens: list[str]
    scores: list[float]
    token_type: list[int]
    merges: list[str]
    bos_token_id: int
    eos_token_id: int
    unknown_token_id: int
    padding_token_id: int
    mask_token_id: int
    chat_template: str
    add_space_prefix: bool
    add_bos_token: bool


def make_linear(*shape: int, w: torch.Tensor):
    assert shape == w.T.shape, f"expected weight tensor shape {shape}, but got {w.shape}"
    lin = nn.Linear(*shape, bias=False)
    lin.weight = nn.Parameter(w)
    return lin


class RMSNorm(nn.Module):
    def __init__(self, conf: Gemma4Config, w: torch.Tensor | None):
        super().__init__()
        self.w = w
        self.eps = conf.attention_layer_norm_rms_epsilon

    def forward(self, x: torch.Tensor):
        # x / (x.square().mean(dim=-1) + self.eps).sqrt() * self.w
        return F.rms_norm(x, [x.shape[-1]], self.w, self.eps)


class Embeddings(nn.Module):
    def __init__(
        self,
        conf: Gemma4Config,
        token_embeddings: torch.Tensor,
        per_layer_embeddings: torch.Tensor,
        per_layer_token_proj: torch.Tensor,
        per_layer_token_norm: RMSNorm,
    ):
        super().__init__()
        self.embed_dim = conf.embedding_length
        self.ple_dim = conf.embedding_length_per_layer_input
        self.n_layers = conf.block_count
        self.token_embeddings = token_embeddings.reshape((-1, self.embed_dim))
        self.per_layer_embeddings = per_layer_embeddings.reshape((-1, self.n_layers, self.ple_dim))
        self.per_layer_token_proj = make_linear(
            conf.embedding_length, conf.block_count * conf.embedding_length_per_layer_input, w=per_layer_token_proj
        )
        self.per_layer_token_norm = per_layer_token_norm

    def forward(self, input_ids: torch.Tensor):
        embed = self.token_embeddings[input_ids] * self.embed_dim**0.5
        ple_lookup = self.per_layer_embeddings[input_ids] * self.ple_dim**0.5
        ple_lookup = ple_lookup.reshape((*input_ids.shape, self.n_layers, self.ple_dim))

        ple_proj = self.per_layer_token_proj(embed) * self.embed_dim**-0.5
        ple_proj = ple_proj.reshape((*input_ids.shape, self.n_layers, self.ple_dim))
        ple_proj = self.per_layer_token_norm(ple_proj)

        ple = (ple_lookup + ple_proj) * 2**-0.5
        return embed, ple


class PerLayerEmbeddings(nn.Module):
    def __init__(self, conf: Gemma4Config, gate_proj: torch.Tensor, out_proj: torch.Tensor, out_norm: RMSNorm):
        super().__init__()
        self.gate_proj = make_linear(conf.embedding_length, conf.embedding_length_per_layer_input, w=gate_proj)
        self.out_proj = make_linear(conf.embedding_length_per_layer_input, conf.embedding_length, w=out_proj)
        self.out_norm = out_norm

    def forward(self, x: torch.Tensor, ple: torch.Tensor):
        gate = F.gelu(self.gate_proj(x), approximate="tanh")
        return self.out_norm(self.out_proj(ple * gate))


class MLP(nn.Module):
    def __init__(
        self,
        conf: Gemma4Config,
        up_proj: torch.Tensor,
        gate_proj: torch.Tensor,
        down_proj: torch.Tensor,
        input_norm: RMSNorm,
        output_norm: RMSNorm,
    ):
        super().__init__()
        self.up_proj = make_linear(conf.embedding_length, conf.feed_forward_length, w=up_proj)
        self.gate_proj = make_linear(conf.embedding_length, conf.feed_forward_length, w=gate_proj)
        self.down_proj = make_linear(conf.feed_forward_length, conf.embedding_length, w=down_proj)
        self.input_norm = input_norm
        self.output_norm = output_norm

    def forward(self, x: torch.Tensor):
        x = self.input_norm(x)
        y = self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))
        return self.output_norm(y)


# TODO this currently does not implement prefill correctly.
class Attention(nn.Module):
    def __init__(
        self,
        conf: Gemma4Config,
        layer_idx: int,
        input_norm: RMSNorm,
        q_proj: torch.Tensor,
        q_norm: RMSNorm,
        k_proj: torch.Tensor,
        k_norm: RMSNorm,
        v_proj: torch.Tensor,
        v_norm: RMSNorm,
        o_proj: torch.Tensor,
        o_norm: RMSNorm,
        rope_freqs: torch.Tensor,
    ):
        super().__init__()

        self.layer_idx = layer_idx
        self.n_layers = conf.block_count
        self.n_shared_kv_layers = conf.attention_shared_kv_layers
        self.is_swa = conf.attention_sliding_window_pattern[layer_idx]
        self.previous_same_type_layer_idx = None
        for i in reversed(range(layer_idx)):
            if conf.attention_sliding_window_pattern[i] == self.is_swa and self.n_layers - i > self.n_shared_kv_layers:
                self.previous_same_type_layer_idx = i
                break

        self.sliding_window_size = conf.attention_sliding_window
        self.head_count_q = conf.attention_head_count
        self.head_count_kv = conf.attention_head_count_kv
        self.rope = RoPE(conf, None if self.is_swa else rope_freqs)
        self.input_norm = input_norm
        # fmt: off
        if self.is_swa:
            self.q_proj = make_linear(conf.embedding_length, conf.attention_head_count * conf.attention_key_length_swa, w=q_proj)
            self.k_proj = make_linear(conf.embedding_length, conf.attention_head_count_kv * conf.attention_key_length_swa, w=k_proj)
            self.v_proj = make_linear(conf.embedding_length, conf.attention_head_count_kv * conf.attention_value_length_swa, w=v_proj)
            self.o_proj = make_linear(conf.attention_head_count * conf.attention_value_length_swa, conf.embedding_length, w=o_proj)
            self.head_size_qk = conf.attention_key_length_swa
            self.head_size_v = conf.attention_value_length_swa
        else:
            self.q_proj = make_linear(conf.embedding_length, conf.attention_head_count * conf.attention_key_length, w=q_proj)
            self.k_proj = make_linear(conf.embedding_length, conf.attention_head_count_kv * conf.attention_key_length, w=k_proj)
            self.v_proj = make_linear(conf.embedding_length, conf.attention_head_count_kv * conf.attention_value_length, w=v_proj)
            self.o_proj = make_linear(conf.attention_head_count * conf.attention_value_length, conf.embedding_length, w=o_proj)
            self.head_size_qk = conf.attention_key_length
            self.head_size_v = conf.attention_value_length
        # fmt: on
        self.q_norm = q_norm
        self.k_norm = k_norm
        self.v_norm = v_norm
        self.o_norm = o_norm

    def forward(self, x: torch.Tensor, kv_cache: KVCaches):
        x = self.input_norm(x)
        B, T, H_q, H_kv = x.shape[0], x.shape[1], self.head_count_q, self.head_count_kv
        D_qk, D_v = self.head_size_qk, self.head_size_v

        if self.n_layers - self.layer_idx > self.n_shared_kv_layers:
            # used so you know where in time the query actually is (since the query tensor has T=1) and also in case
            # the context window overflows and the kv cache has to truncate, for the newly inserted keys/values.
            time_offset = kv_cache[self.layer_idx].base_time_offset
            # compute KV and add to cache
            k = self.k_norm(self.k_proj(x).reshape((B, T, H_kv, D_qk)).transpose(1, 2))
            v = self.v_norm(self.v_proj(x).reshape((B, T, H_kv, D_v)).transpose(1, 2))
            (k,) = self.rope(k, base_time_offset=time_offset)
            kv_cache[self.layer_idx].add(k, v)
            retrieve_layer_idx = self.layer_idx
        else:
            # otherwise use kv cache from last same-type attn instead of this layer having its own
            assert self.previous_same_type_layer_idx is not None
            retrieve_layer_idx = self.previous_same_type_layer_idx
            # add of previous layer already advanced base_time_offset, so we un-advance
            time_offset = kv_cache[retrieve_layer_idx].base_time_offset - T

        q = self.q_norm(self.q_proj(x).reshape((B, T, H_q, D_qk)).transpose(1, 2))
        (q,) = self.rope(q, base_time_offset=time_offset)
        out = kv_cache[retrieve_layer_idx].retrieve(q)
        out = out.transpose(1, 2).reshape((B, T, H_q * D_v))
        out = self.o_norm(self.o_proj(out))
        return out


class RoPE(nn.Module):
    def __init__(self, conf: Gemma4Config, rope_freqs: torch.Tensor | None):
        super().__init__()
        rope_freq_base = conf.rope_freq_base_swa if rope_freqs is None else conf.rope_freq_base
        head_dim = conf.rope_dimension_count_swa if rope_freqs is None else conf.rope_dimension_count
        self.freqs = 1.0 / (rope_freq_base ** (torch.arange(0, head_dim, 2) / head_dim))
        if rope_freqs is not None:
            self.freqs /= rope_freqs

    @staticmethod
    def rotate_half(x: torch.Tensor):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, *tensors: torch.Tensor, base_time_offset: int = 0) -> tuple[torch.Tensor, ...]:
        T = tensors[0].shape[-2]
        assert all(t.shape[-2] == T for t in tensors)

        theta = torch.arange(base_time_offset, base_time_offset + T).reshape((T, 1)) * self.freqs
        theta = torch.cat([theta, theta], dim=-1)

        sin = theta.sin()  # shape: (seq_len, head_dim)
        cos = theta.cos()  # shape: (seq_len, head_dim)

        return tuple(tensor * cos + RoPE.rotate_half(tensor) * sin for tensor in tensors)


class DecoderLayer(nn.Module):
    def __init__(
        self,
        conf: Gemma4Config,
        layer_idx: int,
        q_proj: torch.Tensor,
        q_norm: RMSNorm,
        k_proj: torch.Tensor,
        k_norm: RMSNorm,
        v_proj: torch.Tensor,
        v_norm: RMSNorm,
        o_proj: torch.Tensor,
        attn_input_norm: RMSNorm,
        attn_output_norm: RMSNorm,
        mlp_up_proj: torch.Tensor,
        mlp_gate_proj: torch.Tensor,
        mlp_down_proj: torch.Tensor,
        mlp_input_norm: RMSNorm,
        mlp_output_norm: RMSNorm,
        ple_gate_proj: torch.Tensor,
        ple_out_proj: torch.Tensor,
        ple_out_norm: RMSNorm,
        rope_freqs: torch.Tensor,
        layer_output_scale: torch.Tensor,
    ):
        super().__init__()
        self.attn = Attention(
            conf,
            layer_idx,
            attn_input_norm,
            q_proj,
            q_norm,
            k_proj,
            k_norm,
            v_proj,
            v_norm,
            o_proj,
            attn_output_norm,
            rope_freqs,
        )
        self.mlp = MLP(conf, mlp_up_proj, mlp_gate_proj, mlp_down_proj, mlp_input_norm, mlp_output_norm)
        self.ple = PerLayerEmbeddings(conf, ple_gate_proj, ple_out_proj, ple_out_norm)
        self.layer_output_scale = layer_output_scale

    def forward(self, x: torch.Tensor, ple: torch.Tensor, kv_cache: KVCaches):
        x += self.attn(x, kv_cache)
        x += self.mlp(x)
        x += self.ple(x, ple)
        return x * self.layer_output_scale

    @classmethod
    def load_from_model_dict(cls, conf: Gemma4Config, m: dict[str, torch.Tensor], layer_idx: int) -> DecoderLayer:
        prefix = f"blk.{layer_idx}"
        by_name = lambda name: m[f"{prefix}.{name}.weight"]
        return cls(
            conf,
            layer_idx,
            q_proj=by_name("attn_q"),
            q_norm=RMSNorm(conf, by_name("attn_q_norm")),
            k_proj=by_name("attn_k"),
            k_norm=RMSNorm(conf, by_name("attn_k_norm")),
            v_proj=by_name("attn_v"),
            v_norm=RMSNorm(conf, None),
            o_proj=by_name("attn_output"),
            attn_input_norm=RMSNorm(conf, by_name("attn_norm")),
            attn_output_norm=RMSNorm(conf, by_name("post_attention_norm")),
            mlp_up_proj=by_name("ffn_up"),
            mlp_gate_proj=by_name("ffn_gate"),
            mlp_down_proj=by_name("ffn_down"),
            mlp_input_norm=RMSNorm(conf, by_name("ffn_norm")),
            mlp_output_norm=RMSNorm(conf, by_name("post_ffw_norm")),
            ple_gate_proj=by_name("inp_gate"),
            ple_out_proj=by_name("proj"),
            ple_out_norm=RMSNorm(conf, by_name("post_norm")),
            rope_freqs=m["rope_freqs.weight"],
            layer_output_scale=by_name("layer_output_scale"),
        )


class ModelHead(nn.Module):
    def __init__(self, conf: Gemma4Config, norm: RMSNorm, token_embeddings: torch.Tensor):
        super().__init__()
        self.norm = norm
        self.token_embeddings = token_embeddings
        self.logit_softcap = conf.final_logit_softcapping

    def forward(self, x: torch.Tensor):
        x = self.norm(x)
        x = x @ self.token_embeddings.T
        x = self.logit_softcap * (x / self.logit_softcap).tanh()
        return x


class Gemma4TextModel(nn.Module):
    def __init__(
        self,
        layers: nn.ModuleList[DecoderLayer],
        lm_head: ModelHead,
        embeddings: Embeddings,
    ):
        super().__init__()
        self.embeddings = embeddings
        self.layers = layers
        self.lm_head = lm_head

    def forward(self, input_ids: torch.Tensor, kv_cache: KVCaches):
        assert input_ids.ndim == 2
        x, ple = self.embeddings(input_ids)
        for i, l in enumerate(self.layers):
            x = l(x, ple[:, :, i, :], kv_cache)
        x = self.lm_head(x)
        return x

    @classmethod
    def load_from_model_dict(cls, conf: Gemma4Config, m: dict[str, torch.Tensor]) -> Gemma4TextModel:
        return cls(
            layers=nn.ModuleList(DecoderLayer.load_from_model_dict(conf, m, i) for i in range(conf.block_count)),
            lm_head=ModelHead(conf, RMSNorm(conf, m["output_norm.weight"]), m["token_embd.weight"]),
            embeddings=Embeddings(
                conf,
                token_embeddings=m["token_embd.weight"],
                per_layer_embeddings=m["per_layer_token_embd.weight"],
                per_layer_token_proj=m["per_layer_model_proj.weight"],
                per_layer_token_norm=RMSNorm(conf, m["per_layer_proj_norm.weight"]),
            ),
        )


class LayerKVCache:
    def __init__(self, conf: Gemma4Config, layer_idx: int):
        self.is_swa = conf.attention_sliding_window_pattern[layer_idx]
        self.max_context_len = conf.attention_sliding_window if self.is_swa else conf.context_length
        self.buffer_t_size = 4 * self.max_context_len
        # batch size fixed to 1 for now
        B, H = 1, conf.attention_head_count_kv
        D_k = conf.attention_key_length_swa if self.is_swa else conf.attention_key_length
        D_v = conf.attention_value_length_swa if self.is_swa else conf.attention_value_length
        self.k_buf = torch.zeros((B, H, self.buffer_t_size, D_k))
        self.v_buf = torch.zeros((B, H, self.buffer_t_size, D_v))
        self.next_t_write_offset = 0
        self.completed_buffer_rotations = 0

    def retrieve(self, q: torch.Tensor) -> torch.Tensor:
        t_end = self.next_t_write_offset
        t_start = max(t_end - self.max_context_len, 0)
        k, v = self.k_buf[:, :, t_start:t_end], self.v_buf[:, :, t_start:t_end]

        k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], 1)
        v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], 1)

        mask = torch.arange(t_end - q.shape[-2], t_end)[:, None] < torch.arange(t_start, t_end)[None, :]
        mask = torch.where(mask, float("-inf"), 0)
        return (q @ k.transpose(-1, -2) + mask).softmax(-1) @ v

    def add(self, k: torch.Tensor, v: torch.Tensor):
        B, H, T_new, D_k = k.shape
        D_v = v.shape[-1]
        assert v.shape == (B, H, T_new, D_v)
        assert self.k_buf.shape == (B, H, self.buffer_t_size, D_k)
        assert self.v_buf.shape == (B, H, self.buffer_t_size, D_v)

        if T_new >= self.max_context_len:
            # cap T_new at max context len
            k = k[:, :, -self.max_context_len :]
            v = v[:, :, -self.max_context_len :]
            T_new = self.max_context_len

        if self.next_t_write_offset + T_new >= self.buffer_t_size:
            # buffer full - rotate to the beginning
            T_old = self.max_context_len - T_new
            if T_new != self.max_context_len:
                self.k_buf[:, :, :T_old] = self.k_buf[:, :, -T_old:]
                self.v_buf[:, :, :T_old] = self.v_buf[:, :, -T_old:]
            self.next_t_write_offset = T_old
            self.completed_buffer_rotations += 1

        # append
        self.k_buf[:, :, self.next_t_write_offset : self.next_t_write_offset + T_new] = k
        self.v_buf[:, :, self.next_t_write_offset : self.next_t_write_offset + T_new] = v
        self.next_t_write_offset += T_new

    @property
    def base_time_offset(self) -> int:
        return self.next_t_write_offset + (self.buffer_t_size - self.max_context_len) * self.completed_buffer_rotations


type KVCaches = list[LayerKVCache]


def make_kv_caches(conf: Gemma4Config) -> KVCaches:
    return [LayerKVCache(conf, i) for i in range(conf.block_count - conf.attention_shared_kv_layers)]


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


@torch.inference_mode()
def load_model() -> tuple[Gemma4TextModel, Gemma4Config, TokenizerConfig, SamplingConfig]:
    meta, tensors = gguf.load(Path("/mnt/C/models/gemma-4-E4B-it-UD-Q4_K_XL.gguf"))
    check_meta(meta)
    check_dtypes_supported(tensors)

    model_dict: dict[str, torch.Tensor] = {}
    for t in tensors:
        tensor = convert_to_fp32_tensor(t)
        model_dict[t.name] = tensor
    del tensors

    conf, tokenizer_conf, sampling_conf = load_gemma4_conf(meta), load_tokenizer_conf(meta), load_sampling_conf(meta)
    return Gemma4TextModel.load_from_model_dict(conf, model_dict), conf, tokenizer_conf, sampling_conf


@torch.inference_mode()
def sample_model(
    model: Gemma4TextModel,
    conf: Gemma4Config,
    tokenizer_conf: TokenizerConfig,
    sampling_conf: SamplingConfig,
    input_ids: list[int],
    seed: int,
):
    torch.random.manual_seed(seed)
    kv_cache = make_kv_caches(conf)
    token_ids = input_ids.copy()
    # TODO handle prefill

    while token_ids[-1] != tokenizer_conf.eos_token_id:
        logits = model(torch.tensor([token_ids[-1]]).reshape((1, -1)), kv_cache)
        logits /= sampling_conf.temperature
        samples = top_k_top_p_filtering(logits.reshape((1, -1)), sampling_conf.top_k, sampling_conf.top_p)
        sample = samples.item()
        assert isinstance(sample, int)
        token_ids.append(sample)
        yield detokenize(tokenizer_conf, [token_ids[-1]])


def check_meta(meta: gguf.GGUFMeta):
    if (x := meta["general.architecture"]) != "gemma4":
        raise RuntimeError(f"unknown general.architecture '{x}'")
    if (x := meta["general.type"]) != "model":
        raise RuntimeError(f"unknown general.type '{x}'")
    if (x := meta["general.quantization_version"]) != 2:
        raise RuntimeError(f"unknown general.quantization_version '{x}'")
    if (x := meta["general.file_type"]) != 15:
        raise RuntimeError(f"unknown general.file_type '{x}'")


def load_gemma4_conf(meta: gguf.GGUFMeta) -> Gemma4Config:
    return Gemma4Config(
        block_count=meta["gemma4.block_count"],
        context_length=meta["gemma4.context_length"],
        embedding_length=meta["gemma4.embedding_length"],
        feed_forward_length=meta["gemma4.feed_forward_length"],
        attention_head_count=meta["gemma4.attention.head_count"],
        attention_head_count_kv=meta["gemma4.attention.head_count_kv"],
        rope_freq_base=meta["gemma4.rope.freq_base"],
        rope_freq_base_swa=meta["gemma4.rope.freq_base_swa"],
        attention_layer_norm_rms_epsilon=meta["gemma4.attention.layer_norm_rms_epsilon"],
        attention_key_length=meta["gemma4.attention.key_length"],
        attention_value_length=meta["gemma4.attention.value_length"],
        final_logit_softcapping=meta["gemma4.final_logit_softcapping"],
        attention_sliding_window=meta["gemma4.attention.sliding_window"],
        attention_shared_kv_layers=meta["gemma4.attention.shared_kv_layers"],
        embedding_length_per_layer_input=meta["gemma4.embedding_length_per_layer_input"],
        attention_sliding_window_pattern=meta["gemma4.attention.sliding_window_pattern"],
        attention_key_length_swa=meta["gemma4.attention.key_length_swa"],
        attention_value_length_swa=meta["gemma4.attention.value_length_swa"],
        rope_dimension_count=meta["gemma4.rope.dimension_count"],
        rope_dimension_count_swa=meta["gemma4.rope.dimension_count_swa"],
    )


def load_sampling_conf(meta: gguf.GGUFMeta) -> SamplingConfig:
    return SamplingConfig(
        top_k=meta["general.sampling.top_k"],
        top_p=meta["general.sampling.top_p"],
        temperature=meta["general.sampling.temp"],
    )


def load_tokenizer_conf(meta: gguf.GGUFMeta) -> TokenizerConfig:
    if (x := meta["tokenizer.ggml.model"]) != "gemma4":
        raise RuntimeError(f"tokenizer.ggml.model '{x}' is unsupported")
    if meta["tokenizer.ggml.add_space_prefix"]:
        raise RuntimeError(f"tokenizer.ggml.add_space_prefix True is unsupported")
    if not meta["tokenizer.ggml.add_bos_token"]:
        raise RuntimeError(f"tokenizer.ggml.add_bos_token False is unsupported")
    return TokenizerConfig(
        model=meta["tokenizer.ggml.model"],
        tokens=meta["tokenizer.ggml.tokens"],
        scores=meta["tokenizer.ggml.scores"],
        token_type=meta["tokenizer.ggml.token_type"],
        merges=meta["tokenizer.ggml.merges"],
        bos_token_id=meta["tokenizer.ggml.bos_token_id"],
        eos_token_id=meta["tokenizer.ggml.eos_token_id"],
        unknown_token_id=meta["tokenizer.ggml.unknown_token_id"],
        padding_token_id=meta["tokenizer.ggml.padding_token_id"],
        mask_token_id=meta["tokenizer.ggml.mask_token_id"],
        chat_template=meta["tokenizer.chat_template"],
        add_space_prefix=meta["tokenizer.ggml.add_space_prefix"],
        add_bos_token=meta["tokenizer.ggml.add_bos_token"],
    )


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


# https://gist.github.com/bsantraigi/5752667525d88d375207f099bd78818b
def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float("Inf")):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (vocabulary size)
        top_k >0: keep only top k tokens with highest probability (top-k filtering).
        top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)

    Basic outline taken from https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 2  # [BATCH_SIZE, VOCAB_SIZE]
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k, dim=1)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)

    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # Replace logits to be removed with -inf in the sorted_logits
    sorted_logits[sorted_indices_to_remove] = filter_value
    # Then reverse the sorting process by mapping back sorted_logits to their original position
    logits = torch.gather(sorted_logits, 1, sorted_indices.argsort(-1))

    pred_token = torch.multinomial(F.softmax(logits, -1), 1)  # [BATCH_SIZE, 1]
    return pred_token


def detokenize(tokenizer_conf: TokenizerConfig, token_ids: list[int]) -> str:
    return "".join(map(lambda i: tokenizer_conf.tokens[i], token_ids))


# TODO primitive slow tokenize function


if __name__ == "__main__":
    model, gemma_config, tokenizer_config, sampling_config = load_model()
    for tok in sample_model(
        model, gemma_config, tokenizer_config, sampling_config, [tokenizer_config.bos_token_id], 43
    ):
        print(tok, end="")
    print()
