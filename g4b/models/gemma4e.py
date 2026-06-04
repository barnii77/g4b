import warnings
from dataclasses import dataclass
from g4b.gguf import GGUFMeta, GGUFTensor, GGUFType
from g4b.models.model import Model, record_static_cuda_graph
from g4b.scheduler import Scheduler
from g4b.tensor import Tensor, float32, bfloat16, int8, int32
from g4b.config import Config
from g4b.utils import gguf_tensors_by_name
from g4b import kernels

# TODO I cannot bake in the ple_proj into ple_lookup because they are both quantized, so I'll need a kernel that
#  computes both at the start of the forward pass.

# TODO I can fuse the sum-of-squares computation of a decoder block's output rmsnorm into the last matmul (with
#  atomic_add). Then, I make a specialized reduction kernel instead of the next layer's input rmsnorm which
#  1) reads the sum-of-squares, divides by D and takes the root and computes the inverse -> inv_rms
#  2) loads residual stream and output of last layer's last matmul which did the sum-of-squares accum
#  3) new_resid = residual.load() + last_matmul_out.load() * inv_rms * last_rmsnorm_w.load(); residual.store(new_resid)
#  4) immediately also computes a reduce-sum over new_resid and computes the new_resid_inv_rms from that
#  5) prologue-fuses new_resid_inv_rms scaling before the next matmul after dequant, using block-scaled mma if avail.
#  The end result of this is that you do 2 RMSNorms with 1 resid_stream_combine kernel and get the rest for ~free.
#  This method can (and also kinda has to) be applied as well to the lm_head, since the lm head will receive not a clean
#  residual stream but rather the second-to-last layer's residual stream plus the last layer's matmul output and sum of
#  squares. Then you can again run this resid_stream_combine kernel instead of the lm head norm, and prologue fuse the scaling
#  into the logits matmul. Then you just write out the logits to a bit tensor and run your sampling kernel.
#  Also, in all this, I must not forget about DecoderLayer.layer_output_scale, so I guess I need a special case after
#  the PLE layer. Remember, the layer output scale scales the entire residual stream, not just the decoder layer delta.
#  I think this optimization needs to be documented somewhere.
# TODO I can fuse the rmsnorm w mul into the epilogue which computes the sum of squares too or, for input rmsnorms,
#  fuse it into the resid_stream_combine kernel. (just make sure to compute the sum of squares from the original values).

# TODO think about RoPE fusion and SwiGLU fusion.
#  SwiGLU fusion works by having a single for loop over K in your kernel that MMAs both the up proj and gate proj, and
#  then you get up_tile and gate_tile and then you immediately do out = gelu(gate_tile) * up_tile; out_ptr.store(out)
#  As for RoPE fusion, it may be tricky because of NeoX RoPE.
#  I think these optimizations need to be documented somewhere as well.

# TODO think about how to handle sliding window attention... do I need a ring buffer KV cache?
#  Hmm, actually I think I need a ring buffer KV cache for global attention too.
#  Also how do I keep track of time (index in T dim) for RoPE?

################
# Tensors names below are suffixed with shape and dtype annotations.
# The dtypes refer to the following:
DTR = float32  # DType for Residual stream
DTA = float32  # DType for Attention
DTH = bfloat16  # DType for (pre-)activations inside mlp
DTKV = bfloat16  # DType for KV cache
DTMM = int8  # DType for MatMul tensor core ops
DTSS = float32  # DType for Sum-of-Squares accumulation for rmsnorm
DTPLE = float32  # DType for computations in the per-layer-embeddings low-rank space
DTSAMP = float32
# The shape names, explained:
#   - B: batch size
#   - T: context window size
#   - t: number of decode tokens (typically 1 for standard autoregressive generation, t=T for prefill)
#       - WARNING: this one differs between chunked prefill and decode
#   - W: window size for sliding window attention
#   - D: residual stream/embedding size
#   - k: query/key size (of a single key)
#       - WARNING: may differ between MHA and SWA
#   - v: value size (of a single value)
#       - WARNING: may differ between MHA and SWA
#   - h: number of query heads
#   - g: number of GQA key/value heads (applies to both MHA and SWA)
#   - U: the MLP hidden size (up-projection size)
#   - P: per-layer embedding size (e.g. 256 for gemma 4 e4b)
#   - V: vocab size
#   - L: number of layers
################


@dataclass(frozen=True)
class Attention:
    # parameters
    q_proj_Dhk_q4: Tensor
    k_proj_Dgk_q6: Tensor
    v_proj_Dgv_q6: Tensor
    o_proj_hvD_q4: Tensor
    q_rmsnorm_w_k_fp32: Tensor
    k_rmsnorm_w_k_fp32: Tensor
    o_rmsnorm_w_D_fp32: Tensor
    input_rmsnorm_w_D_fp32: Tensor
    rope_freqs_k_fp32: Tensor  # TODO I must compute the default rope frequencies and `if not is_swa` I must multiply in the rope_freqs from the gguf file. See ref impl -> class RoPE.
    context_window_size: int  # small for SWA
    owns_kv_cache: bool  # model's late layers share KV cache with earlier ones. If true, must not write to KV cache.
    head_count_q: int
    head_count_kv: int
    head_size_qk: int
    head_size_v: int

    # runtime state
    k_cache_Bg__T_or_W__k__dtkv: Tensor
    v_cache_Bg__T_or_W__v__dtkv: Tensor
    shared_q_scratchpad_Bhtk_dta: Tensor
    shared_k_scratchpad_Bgtk_dta: Tensor
    shared_v_scratchpad_Bgtv_dta: Tensor
    shared_o_proj_input_scratchpad_Bhtv_dta: Tensor
    shared_last_and_this_layer_output_scratchpad_BtD_dtr: Tensor
    shared_last_and_this_layer_output_sum_of_squares_accum_scratchpad_Bt_dtss: Tensor


@dataclass(frozen=True)
class MLP:
    # parameters
    up_proj_DU_q4: Tensor
    gate_proj_DU_q4: Tensor
    down_proj_UD_q6: Tensor
    input_rmsnorm_w_D_fp32: Tensor
    output_rmsnorm_w_D_fp32: Tensor

    # runtime state
    shared_down_proj_input_scratchpad_BtU_dth: Tensor
    shared_last_and_this_layer_output_scratchpad_BtD_dtr: Tensor
    shared_last_and_this_layer_output_sum_of_squares_accum_scratchpad_Bt_dtss: Tensor


@dataclass(frozen=True)
class PerLayerEmbeddings:
    # parameters
    gate_proj_DP_fp32: Tensor
    out_proj_PD_fp32: Tensor
    output_rmsnorm_w_D_fp32: Tensor

    # runtime state
    shared_out_proj_input_scratchpad_BtP_dtple: Tensor
    shared_last_and_this_layer_output_scratchpad_BtD_dtr: Tensor
    shared_last_and_this_layer_output_sum_of_squares_accum_scratchpad_Bt_dtss: Tensor


@dataclass(frozen=True)
class DecoderLayer:
    # parameters
    attn: Attention
    mlp: MLP
    ple: PerLayerEmbeddings
    layer_output_scale: float


@dataclass(frozen=True)
class Embeddings:
    # parameters
    embeddings_VD_q5: Tensor
    ple_table_VLP_q5: Tensor
    ple_proj_DLP_bf16: Tensor
    ple_rmsnorm_w_P_fp32: Tensor

    # runtime state
    ple_LBtP_dtple: Tensor


@dataclass(frozen=True)
class LmHead:
    # parameters
    rmsnorm_w_D_fp32: Tensor
    embeddings_DV_q5: Tensor

    # runtime state
    logits_BtV_dtsamp: Tensor
    logit_softcap: float  # TODO don't forget to epilogue fuse this into the residual->logits matmul


@dataclass(frozen=True)
class SamplingState:
    # runtime state
    out_token_ids_Bt_int32: Tensor
    top_k_logits_scratchpad_Bt__num_splits__top_k__dtsamp: Tensor
    top_k_idx_scratchpad_Bt__num_splits__top_k__int32: Tensor
    top_k: int
    top_p: float
    temperature: float


@dataclass(frozen=True)
class Gemma4E(Model):
    # parameters
    embeddings: Embeddings
    layers: list[DecoderLayer]
    lm_head: LmHead
    sampling_state: SamplingState
    rmsnorm_epsilon: float  # TODO ensure all RMSNorms use the epsilon from the gguf file

    # runtime state
    residual_BtD_dtr: Tensor
    # TODO update this in separate kernel after sampling!
    time_dim_offsets_B_int32: Tensor  # time dim is dynamically sized

    @record_static_cuda_graph
    def decode(self, sched: Scheduler): ...  # TODO

    @record_static_cuda_graph
    def prefill_chunk(self, sched: "scheduler.Scheduler"): ...  # TODO

    # TODO technically for MTP if the MTP model produced keys and values (e.g. self-speculative decoding),
    #  I would need the kv cache to have time dim size (T + t - 1), not T. I must think about how to annotate this
    #  throughout the code and whether any of my code so far assumes size T when it should be (T + t - 1)...
    #  Should I reassign the name T and W to mean (context_len + t - 1)? Probably...
    #  This means the attn kernel needs two separate args: clen and T (though one is derived from the other I guess)
    @classmethod
    def load(cls, meta: GGUFMeta, tensors: list[GGUFTensor], config: Config):
        _check_meta(meta)
        _check_dtypes_supported(tensors)
        if config.context_len > meta["gemma4.context_length"]:
            warnings.warn("configured context length exceeds maximum context length specified in GGUF file")

        # define important sizes
        B = config.batch_size
        t = 1  # no MTP support at the moment
        T = ...  # config.context_len ... + t - 1?
        W = ...  # meta["gemma4.attention.sliding_window"] ... + t - 1?
        D = meta["gemma4.embedding_length"]
        k_gqa = meta["gemma4.attention.key_length"]
        v_gqa = meta["gemma4.attention.value_length"]
        k_swa = meta["gemma4.attention.key_length_swa"]
        v_swa = meta["gemma4.attention.value_length_swa"]
        h = meta["gemma4.attention.head_count"]
        g = meta["gemma4.attention.head_count_kv"]
        U = meta["gemma4.feed_forward_length"]
        P = meta["gemma4.embedding_length_per_layer_input"]
        V = len(meta["tokenizer.ggml.tokens"])
        L = meta["gemma4.block_count"]
        assert all(isinstance(x, int) for x in [B, t, T, W, D, k_gqa, v_gqa, k_swa, v_swa, h, g, U, P, V, L])

        ts = gguf_tensors_by_name(tensors)

        # sampling state tensors and config
        top_k = meta["general.sampling.top_k"]
        assert isinstance(top_k, int)
        top_p = meta["general.sampling.top_p"]
        assert isinstance(top_p, float)
        temperature = meta["general.sampling.temp"]
        assert isinstance(temperature, float)
        out_token_ids = Tensor.alloc_empty(int32, [B, t])
        num_sampling_splits = kernels.sample_logits.get_recommended_num_v_splits(V)
        top_k_logits_scratchpad = Tensor.alloc_empty(DTSAMP, [B, t, num_sampling_splits, top_k])
        top_k_idx_scratchpad = Tensor.alloc_empty(int32, [B, t, num_sampling_splits, top_k])
        sampling_state = SamplingState(
            out_token_ids_Bt_int32=out_token_ids,
            top_k_logits_scratchpad_Bt__num_splits__top_k__dtsamp=top_k_logits_scratchpad,
            top_k_idx_scratchpad_Bt__num_splits__top_k__int32=top_k_idx_scratchpad,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )

        # embeddings
        token_embeddings = Tensor.from_gguf_tensor(ts["token_embd.weight"])
        ple_table = Tensor.from_gguf_tensor(ts["per_layer_token_embd.weight"])
        ple_proj = Tensor.from_gguf_tensor(ts["per_layer_model_proj.weight"])
        ple_norm = Tensor.from_gguf_tensor(ts["per_layer_proj_norm.weight"])
        ple_buf = Tensor.alloc_empty(DTPLE, [L, B, t, P])
        embeddings = Embeddings(
            embeddings_VD_q5=token_embeddings,
            ple_table_VLP_q5=ple_table,
            ple_proj_DLP_bf16=ple_proj,
            ple_rmsnorm_w_P_fp32=ple_norm,
            ple_LBtP_dtple=ple_buf,
        )

        # lm head
        head_rmsnorm_w = Tensor.from_gguf_tensor(ts["output_norm.weight"])
        embeddings_DV = token_embeddings.transpose(0, 1)
        logits_buf = Tensor.alloc_empty(DTSAMP, [B, t, V])
        logit_softcap = meta["gemma4.final_logit_softcapping"]
        assert isinstance(logit_softcap, float)
        lm_head = LmHead(
            rmsnorm_w_D_fp32=head_rmsnorm_w,
            embeddings_DV_q5=embeddings_DV,
            logits_BtV_dtsamp=logits_buf,
            logit_softcap=logit_softcap,
        )

        # create global state tensors
        residual = Tensor.alloc_empty(DTR, [B, t, D])
        time_dim_offsets = Tensor.alloc_empty(int32, [B])
        rmsnorm_epsilon = meta["gemma4.attention.layer_norm_rms_epsilon"]
        assert isinstance(rmsnorm_epsilon, float)

        ####### layers
        layers: list[DecoderLayer] = []
        # shared state
        layer_output_scratchpad = Tensor.alloc_empty(DTR, [B, t, D])
        layer_output_sos_scratchpad = Tensor.alloc_empty(DTSS, [B, t])
        # ple state
        ple_out_proj_input_scratchpad = Tensor.alloc_empty(DTPLE, [B, t, P])
        # mlp state
        mlp_down_proj_input_scratchpad = Tensor.alloc_empty(DTH, [B, t, U])
        # shared attn state
        q_gqa_scratchpad = Tensor.alloc_empty(DTA, [B, h, t, k_gqa])
        k_gqa_scratchpad = Tensor.alloc_empty(DTA, [B, g, t, k_gqa])
        v_gqa_scratchpad = Tensor.alloc_empty(DTA, [B, g, t, v_gqa])
        o_gqa_proj_input_scratchpad = Tensor.alloc_empty(DTA, [B, h, t, v_gqa])
        q_swa_scratchpad = Tensor.alloc_empty(DTA, [B, h, t, k_swa])
        k_swa_scratchpad = Tensor.alloc_empty(DTA, [B, g, t, k_swa])
        v_swa_scratchpad = Tensor.alloc_empty(DTA, [B, g, t, v_swa])
        o_swa_proj_input_scratchpad = Tensor.alloc_empty(DTA, [B, h, t, v_swa])
        rope_freqs_gqa = _compute_rope_freqs(
            ts[f"rope_freqs.weight"],
            meta["gemma4.rope.freq_base"],
            meta["gemma4.rope.dimension_count"],
        )
        rope_freqs_swa = _compute_rope_freqs(
            None,
            meta["gemma4.rope.freq_base_swa"],
            meta["gemma4.rope.dimension_count_swa"],
        )
        kv_cache_by_layer: dict[int, tuple[Tensor, Tensor]] = {}

        for i in range(L):
            by_name = lambda name: Tensor.from_gguf_tensor(ts[f"blk.{i}.{name}.weight"])

            # attention
            q_proj = by_name("attn_q")
            k_proj = by_name("attn_k")
            v_proj = by_name("attn_v")
            o_proj = by_name("attn_output")
            q_rmsnorm_w = by_name("attn_q_norm")
            k_rmsnorm_w = by_name("attn_k_norm")
            o_rmsnorm_w = by_name("post_attention_norm")
            input_rmsnorm_w = by_name("attn_norm")
            swa_pattern = meta["gemma4.attention.sliding_window_pattern"]
            is_swa = swa_pattern[i]
            context_window_size = min(W, config.context_len) if is_swa else config.context_len
            assert isinstance(context_window_size, int)

            owns_kv_cache = L - i >= meta["gemma4.attention.shared_kv_layers"]
            if owns_kv_cache:
                if is_swa:
                    k_cache = Tensor.alloc_empty(DTKV, [B, g, W, k_swa])
                    v_cache = Tensor.alloc_empty(DTKV, [B, g, W, v_swa])
                else:
                    k_cache = Tensor.alloc_empty(DTKV, [B, g, T, k_gqa])
                    v_cache = Tensor.alloc_empty(DTKV, [B, g, T, v_gqa])
                kv_cache_by_layer[i] = k_cache, v_cache
            else:
                layers_with_same_type = filter(lambda j: swa_pattern[j] == is_swa, kv_cache_by_layer.keys())
                layer_to_share_with = max(layers_with_same_type)
                k_cache, v_cache = kv_cache_by_layer[layer_to_share_with]

            if is_swa:
                attn = Attention(
                    q_proj_Dhk_q4=q_proj,
                    k_proj_Dgk_q6=k_proj,
                    v_proj_Dgv_q6=v_proj,
                    o_proj_hvD_q4=o_proj,
                    q_rmsnorm_w_k_fp32=q_rmsnorm_w,
                    k_rmsnorm_w_k_fp32=k_rmsnorm_w,
                    o_rmsnorm_w_D_fp32=o_rmsnorm_w,
                    input_rmsnorm_w_D_fp32=input_rmsnorm_w,
                    rope_freqs_k_fp32=rope_freqs_swa,
                    context_window_size=context_window_size,
                    owns_kv_cache=owns_kv_cache,
                    head_count_q=h,
                    head_count_kv=g,
                    head_size_qk=k_swa,
                    head_size_v=v_swa,
                    k_cache_Bg__T_or_W__k__dtkv=k_cache,
                    v_cache_Bg__T_or_W__v__dtkv=v_cache,
                    shared_q_scratchpad_Bhtk_dta=q_swa_scratchpad,
                    shared_k_scratchpad_Bgtk_dta=k_swa_scratchpad,
                    shared_v_scratchpad_Bgtv_dta=v_swa_scratchpad,
                    shared_o_proj_input_scratchpad_Bhtv_dta=o_swa_proj_input_scratchpad,
                    shared_last_and_this_layer_output_scratchpad_BtD_dtr=layer_output_scratchpad,
                    shared_last_and_this_layer_output_sum_of_squares_accum_scratchpad_Bt_dtss=layer_output_sos_scratchpad,
                )
            else:
                attn = Attention(
                    q_proj_Dhk_q4=q_proj,
                    k_proj_Dgk_q6=k_proj,
                    v_proj_Dgv_q6=v_proj,
                    o_proj_hvD_q4=o_proj,
                    q_rmsnorm_w_k_fp32=q_rmsnorm_w,
                    k_rmsnorm_w_k_fp32=k_rmsnorm_w,
                    o_rmsnorm_w_D_fp32=o_rmsnorm_w,
                    input_rmsnorm_w_D_fp32=input_rmsnorm_w,
                    rope_freqs_k_fp32=rope_freqs_gqa,
                    context_window_size=context_window_size,
                    owns_kv_cache=owns_kv_cache,
                    head_count_q=h,
                    head_count_kv=g,
                    head_size_qk=k_gqa,
                    head_size_v=v_gqa,
                    k_cache_Bg__T_or_W__k__dtkv=k_cache,
                    v_cache_Bg__T_or_W__v__dtkv=v_cache,
                    shared_q_scratchpad_Bhtk_dta=q_gqa_scratchpad,
                    shared_k_scratchpad_Bgtk_dta=k_gqa_scratchpad,
                    shared_v_scratchpad_Bgtv_dta=v_gqa_scratchpad,
                    shared_o_proj_input_scratchpad_Bhtv_dta=o_gqa_proj_input_scratchpad,
                    shared_last_and_this_layer_output_scratchpad_BtD_dtr=layer_output_scratchpad,
                    shared_last_and_this_layer_output_sum_of_squares_accum_scratchpad_Bt_dtss=layer_output_sos_scratchpad,
                )

            # mlp
            up_proj = by_name("ffn_up")
            gate_proj = by_name("ffn_gate")
            down_proj = by_name("ffn_down")
            mlp_input_norm = by_name("ffn_norm")
            mlp_output_norm = by_name("post_ffw_norm")
            mlp = MLP(
                up_proj_DU_q4=up_proj,
                gate_proj_DU_q4=gate_proj,
                down_proj_UD_q6=down_proj,
                input_rmsnorm_w_D_fp32=mlp_input_norm,
                output_rmsnorm_w_D_fp32=mlp_output_norm,
                shared_down_proj_input_scratchpad_BtU_dth=mlp_down_proj_input_scratchpad,
                shared_last_and_this_layer_output_scratchpad_BtD_dtr=layer_output_scratchpad,
                shared_last_and_this_layer_output_sum_of_squares_accum_scratchpad_Bt_dtss=layer_output_sos_scratchpad,
            )

            # per-layer embeddings
            ple_gate_proj = by_name("inp_gate")
            ple_out_proj = by_name("proj")
            ple_out_norm = by_name("post_norm")
            ple = PerLayerEmbeddings(
                gate_proj_DP_fp32=ple_gate_proj,
                out_proj_PD_fp32=ple_out_proj,
                output_rmsnorm_w_D_fp32=ple_out_norm,
                shared_out_proj_input_scratchpad_BtP_dtple=ple_out_proj_input_scratchpad,
                shared_last_and_this_layer_output_scratchpad_BtD_dtr=layer_output_scratchpad,
                shared_last_and_this_layer_output_sum_of_squares_accum_scratchpad_Bt_dtss=layer_output_sos_scratchpad,
            )

            # construct layer
            layer_output_scale = by_name("layer_output_scale")
            layer = DecoderLayer(attn=attn, mlp=mlp, ple=ple, layer_output_scale=layer_output_scale)
            layers.append(layer)

        return Gemma4E(
            embeddings=embeddings,
            layers=layers,
            lm_head=lm_head,
            sampling_state=sampling_state,
            rmsnorm_epsilon=rmsnorm_epsilon,
            residual_BtD_dtr=residual,
            time_dim_offsets_B_int32=time_dim_offsets,
        )


def _compute_rope_freqs(rope_freqs: GGUFTensor | None, freq_base: float, rope_dim_count: int) -> Tensor:
    freqs = Tensor.from_gguf_tensor(rope_freqs) if rope_freqs is not None else None
    out = Tensor.alloc_empty(float32, [rope_dim_count])
    kernels.rope.populate_rope_frequencies(out, freqs, freq_base)
    return out


def _check_meta(meta: GGUFMeta):
    if (x := meta["general.architecture"]) != "gemma4":
        raise RuntimeError(f"unknown general.architecture '{x}'")
    if (x := meta["general.type"]) != "model":
        raise RuntimeError(f"unknown general.type '{x}'")
    if (x := meta["general.quantization_version"]) != 2:
        raise RuntimeError(f"unknown general.quantization_version '{x}'")
    if (x := meta["general.file_type"]) != 15:
        raise RuntimeError(f"unknown general.file_type '{x}'")


def _check_dtypes_supported(tensors):
    for t in tensors:
        if t.dtype not in (
            GGUFType.GGML_TYPE_F32,
            GGUFType.GGML_TYPE_Q4_K,
            GGUFType.GGML_TYPE_Q5_K,
            GGUFType.GGML_TYPE_Q6_K,
            GGUFType.GGML_TYPE_BF16,
        ):
            raise RuntimeError("unsupported quantization dtype", t.dtype)
