from dataclasses import dataclass
from g4b.gguf import GGUFMeta, GGUFTensor
from g4b.models.model import Model, record_static_cuda_graph
from g4b.scheduler import Scheduler
from g4b.tensor import Tensor, float32, int8
from g4b.config import Config

# TODO dtype suffix in addition to shape suffix
# TODO I can specialize the kernels for literally the exact tensor shapes, i.e. make all shapes and strides constexpr

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
DTMM = int8  # DType for MatMul
# TODO more dtypes?
# The shape names, explained:
#   - B: batch size
#   - T: context window size
#   - t: number of decode tokens (typically 1 for standard autoregressive generation, t=T for prefill)
#   - W: window size for sliding window attention
#   - D: residual stream/embedding size
#   - k: query/key size (of a single key)
#   - v: value size (of a single value)
#   - h: number of query heads (WARNING: may differ between MHA and SWA)
#   - g: number of GQA key/value heads (WARNING: may differ between MHA and SWA)
#   - U: the MLP hidden size (up-projection size)
#   - P: per-layer embedding size (e.g. 256 for gemma 4 e4b)
#   - V: vocab size
#   - L: number of layers
################


@dataclass(frozen=True)
class Attention:
    # parameters
    q_proj_Dhk: Tensor
    k_proj_Dgk: Tensor
    v_proj_Dgv: Tensor
    o_proj_hvD: Tensor
    q_rmsnorm_w_k: Tensor
    k_rmsnorm_w_k: Tensor
    v_rmsnorm_w_v: Tensor
    o_rmsnorm_w_D: Tensor
    input_rmsnorm_w_D: Tensor
    rope_freqs_k: Tensor  # TODO I must compute the default rope frequencies and `if not is_swa` I must multiply in the rope_freqs from the gguf file. See ref impl -> class RoPE.
    rope_freq_base: float  # TODO make sure I assign this correctly with `conf.rope_freq_base_swa if is_swa else conf.rope_freq_base`. See ref impl -> class RoPE.
    sliding_window_size: int | None  # global attention if None
    owns_kv_cache: bool  # model's late layers share KV cache with earlier ones. If true, must not write to KV cache.
    head_count_q: int
    head_count_kv: int
    head_size_qk: int
    head_size_v: int

    # runtime state
    k_cache_Bg__T_or_W__k: Tensor
    v_cache_Bg__T_or_W__v: Tensor
    shared_q_scratchpad_Bhtk: Tensor
    shared_k_scratchpad_Bgtk: Tensor
    shared_v_scratchpad_Bgtk: Tensor
    shared_o_proj_input_scratchpad_Bhtv: Tensor
    shared_last_and_this_layer_output_scratchpad_BtD: Tensor
    shared_last_and_this_layer_output_sum_of_squares_accum_scratchpad_Bt: Tensor


@dataclass(frozen=True)
class MLP:
    # parameters
    up_proj_DU: Tensor
    gate_proj_DU: Tensor
    down_proj_UD: Tensor
    input_rmsnorm_w_D: Tensor
    output_rmsnorm_w_D: Tensor

    # runtime state
    shared_down_proj_input_scratchpad_BtU: Tensor
    shared_last_and_this_layer_output_scratchpad_BtD: Tensor
    shared_last_and_this_layer_output_sum_of_squares_accum_scratchpad_Bt: Tensor


@dataclass(frozen=True)
class PerLayerEmbeddings:
    # parameters
    gate_proj_DP: Tensor
    out_proj_PD: Tensor
    output_rmsnorm_w_D: Tensor

    # runtime state
    shared_out_proj_input_scratchpad_BtP: Tensor
    shared_last_and_this_layer_output_scratchpad_BtD: Tensor
    shared_last_and_this_layer_output_sum_of_squares_accum_scratchpad_Bt: Tensor


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
    embeddings_VD: Tensor
    ple_table_VLP: Tensor
    ple_proj_DLP: Tensor

    # runtime state
    ple_LBtP: Tensor


@dataclass(frozen=True)
class LmHead:
    # parameters
    rmsnorm_w_D: Tensor
    embeddings_DV: Tensor

    # runtime state
    logits_BtV: Tensor
    logits_token_ids_BtV: Tensor
    logit_softcap: float  # TODO don't forget to epilogue fuse this into the residual->logits matmul
    temperature: float


@dataclass(frozen=True)
class Gemma4E(Model):
    # parameters
    embeddings: Embeddings
    layers: list[DecoderLayer]
    lm_head: LmHead

    # runtime state
    residual_BtD_dtr: Tensor
    context_window_sizes_B_int32: Tensor  # time dim is dynamically sized
    out_token_ids_Bt_int32: Tensor
    out_top_k_logits_scratchpad_Bt__num_splits__top_k__fp32: Tensor
    out_top_k_idx_scratchpad_Bt__num_splits__top_k__int32: Tensor

    @classmethod
    def load(cls, gguf_meta: GGUFMeta, gguf_tensors: list[GGUFTensor], config: Config): ...  # TODO

    @record_static_cuda_graph
    def prefill_chunk(self, sched: "scheduler.Scheduler"): ...  # TODO

    @record_static_cuda_graph
    def decode(self, sched: Scheduler): ...  # TODO
