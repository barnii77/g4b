import ctypes
import math
import os
import struct
import warnings
from dataclasses import dataclass
from cuda.core import Buffer, Event
from g4b.gguf import GGUFMeta, GGUFTensor, GGUFType
from g4b.models.model import Model
from g4b.scheduler import Scheduler
from g4b.tensor import Tensor, float32, float16, int8, int32, uint8
from g4b.config import Config
from g4b.lifecycle import record_static_cuda_graph
from g4b.utils import gguf_tensors_by_name
from g4b import kernels, device
from g4b.kernels.memset import memset_contiguous
from g4b.kernels.matmul import matmul_a3d_b2d
from g4b.kernels.fa2 import PHASE_PREFILL, PHASE_DECODE

# TODO if I wanted to support MoE models:
#  I could allocate a tensor for every expert host-side, then build a cuda graph where the router kernel produces
#  expert IDs and then there's cuda graph switch nodes over the produced token IDs which issue a memcpy of the relevant
#  experts to device mem from pinned host mem. This should side-step python and linux latency, though it restricts how
#  advanced the device-side expert caching and prefetch logic can be.

################
# Tensors names below are suffixed with shape and dtype annotations.
# The dtypes refer to the following:
DTR = float32  # DType for Residual stream
DTA = float32  # DType for Attention
DTH = float32 if os.environ.get("G4B_DTH_FP32") else float16  # DType for (pre-)activations inside mlp
DTKV = float32 if os.environ.get("G4B_DTKV_FP32") else float16  # DType for KV cache
DTMM_ACCUM_NORMAL = float32  # if os.environ.get("G4B_DTACCUM_FP32") else float16  # tl.dot accum dtype
DTFA_ACCUM_NORMAL = float32  # if os.environ.get("G4B_DTACCUM_FP32") else float16  # tl.dot accum dtype
DTMM_ACCUM_SENSITIVE = float32
DTMM = int8  # DType for MatMul tensor core ops
DTSS = float32  # DType for Sum-of-Squares accumulation for rmsnorm
DTPLE = float32  # DType for computations in the per-layer-embeddings low-rank space
DTSAMP = float32
FA_MAX_KV_SPLITS = 16
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
# Additional notes on shapes:
#   - A shape like hkD may mean either h x k x D or (h * k) x D, this is not notationally differentiated.
#   - A lot of (all?) weights have the transpose of the valid shape you would expect for the matmul they are used in.
#     This is because gguf stores weights column-major by actually storing their transpose in normal row major format.
#     Transposing ahead of time would force dequantization or at the very least code duplication to handle on-the-fly
#     dequant along the other axis of B. Therefore, I pass transpose_b_before_mma=True to the matmul kernel in these
#     cases, causing the matmul kernel to transpose on the fly inside the kernel.
################


@dataclass(frozen=True)
class Attention:
    # parameters
    q_proj_hkD_q4: Tensor
    k_proj_gkD_q6: Tensor
    v_proj_gvD_q6: Tensor
    o_proj_Dhv_q4: Tensor
    q_rmsnorm_w_k_fp32: Tensor
    k_rmsnorm_w_k_fp32: Tensor
    o_rmsnorm_w_D_fp32: Tensor
    input_rmsnorm_w_D_fp32: Tensor
    rope_freqs_k_fp32: Tensor
    context_window_size: int  # small for SWA
    owns_kv_cache: bool  # model's late layers share KV cache with earlier ones. If false, must not write to KV cache.
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
    shared_q_rsos_Bth_dtss: Tensor
    shared_k_rsos_Btg_dtss: Tensor
    shared_v_rsos_Btg_dtss: Tensor
    shared_o_proj_input_scratchpad_Bhtv_dta: Tensor
    shared_fa_partial_o_SBHtv_f16: Tensor
    shared_fa_partial_l_SBHt_fp32: Tensor
    shared_fa_partial_m_SBHt_fp32: Tensor
    shared_last_and_this_layer_output_scratchpad_BtD_dtr: Tensor
    shared_last_and_this_layer_output_sum_of_squares_accum_scratchpad_Bt_dtss: Tensor


@dataclass(frozen=True)
class MLP:
    # parameters
    up_proj_UD_q4: Tensor
    gate_proj_UD_q4: Tensor
    down_proj_DU_q6: Tensor
    input_rmsnorm_w_D_fp32: Tensor
    output_rmsnorm_w_D_fp32: Tensor

    # runtime state
    shared_down_proj_input_scratchpad_BtU_dth: Tensor
    shared_last_and_this_layer_output_scratchpad_BtD_dtr: Tensor
    shared_last_and_this_layer_output_sum_of_squares_accum_scratchpad_Bt_dtss: Tensor


@dataclass(frozen=True)
class PerLayerEmbeddings:
    # parameters
    gate_proj_PD_fp32: Tensor
    out_proj_DP_fp32: Tensor
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
    layer_output_scale_1_fp32: Tensor


@dataclass(frozen=True)
class Embeddings:
    # parameters
    embeddings_VD_q5: Tensor
    ple_table_VLP_q5: Tensor
    ple_proj_LPD_bf16: Tensor
    ple_rmsnorm_w_P_fp32: Tensor

    # runtime state
    ple_LBtP_dtple: Tensor
    ple_proj_BtLP_dtple: Tensor
    ple_lookup_BtLP_dtple: Tensor
    ple_proj_rsos_BtL_dtss: Tensor


@dataclass(frozen=True)
class LmHead:
    # parameters
    rmsnorm_w_D_fp32: Tensor

    # runtime state
    input_B1D_dtr: Tensor
    input_rsos_B1_dtss: Tensor
    logits_B1V_dtsamp: Tensor
    logit_softcap: float  # TODO don't forget to epilogue fuse this into the residual->logits matmul


@dataclass(frozen=True)
class SamplingState:
    # runtime state
    out_token_ids_B1_int32: Tensor
    top_k_logits_scratchpad_B1__num_splits__top_k__dtsamp: Tensor
    top_k_idx_scratchpad_B1__num_splits__top_k__int32: Tensor
    seed_2_int32: Tensor
    top_k: int
    top_p: float
    temperature: float


@dataclass(frozen=True)
class OutputCopySlot:
    host_buf: Buffer
    event: Event


@dataclass(frozen=True)
class Gemma4E(Model):
    # parameters
    embeddings: Embeddings
    layers: list[DecoderLayer]
    lm_head: LmHead
    sampling_state: SamplingState
    rmsnorm_epsilon: float  # TODO ensure all RMSNorms use the epsilon from the gguf file
    identity_rmsnorm_w_D_fp32: Tensor

    # runtime state
    residual_BtD_dtr: Tensor
    # rolling sum-of-squares of the residual stream. MUST be a buffer distinct from the per-sublayer
    # output-sos scratchpads: update_residual_stream reads the sublayer's output sos (act_rsos arg) and
    # writes the new residual's sos here (out_rsos arg); if they aliased, the wrapper's memset(out_rsos,0)
    # would zero the sublayer sos before it's read -> inv_rms=rsqrt(eps)~1000 -> residual explosion.
    residual_rsos_Bt_dtss: Tensor
    debug_layer_residuals_LB1D_dtr: Tensor | None
    debug_layer_rsos_LB1_dtss: Tensor | None
    input_token_ids_tB_int32: Tensor
    sample_positions_B_int32: Tensor
    cache_offsets_B_int32: Tensor
    time_dim_sizes_B_int32: Tensor  # time dim is dynamically sized
    user_phase_B_uint8: Tensor  # per slot: 0 -> unallocated, 1 -> prefill, 2 -> decode
    output_copy_ring: tuple[OutputCopySlot, ...]
    output_copy_ring_idx: list[int]
    batch_size: int
    prefill_chunk_size: int
    eos_token_id: int

    def max_batch_size(self) -> int:
        return self.batch_size

    def max_prefill_chunk_size(self) -> int:
        return self.prefill_chunk_size

    def stop_token_id(self) -> int:
        return self.eos_token_id

    def prepare_prefill_inputs(
        self,
        token_cols: list[list[int]],
        cache_offsets: list[int],
        time_sizes_after: list[int],
        sample_positions: list[int],
    ):
        self._copy_tokens_t_by_b(token_cols)
        self._copy_i32(self.sample_positions_B_int32, sample_positions)
        self._copy_i32(self.cache_offsets_B_int32, cache_offsets)
        self._copy_i32(self.time_dim_sizes_B_int32, time_sizes_after)
        # TODO scheduler rewrite: set per-slot (unallocated slots -> PHASE_UNALLOCATED) instead of a uniform
        #  vector, and drive transitions via transition_slot() or something.
        self._copy_u8(self.user_phase_B_uint8, [PHASE_PREFILL] * self.batch_size)

    def prepare_decode_inputs(
        self,
        token_cols: list[list[int]],
        cache_offsets: list[int],
        time_sizes_after: list[int],
        sample_positions: list[int],
    ):
        self._copy_tokens_t_by_b(token_cols)
        self._copy_i32(self.sample_positions_B_int32, sample_positions)
        self._copy_i32(self.cache_offsets_B_int32, cache_offsets)
        self._copy_i32(self.time_dim_sizes_B_int32, time_sizes_after)
        self._copy_u8(self.user_phase_B_uint8, [PHASE_DECODE] * self.batch_size)

    def collect_output_token_ids(self) -> list[int]:
        n = self.batch_size
        nbytes = 4 * n
        slot_idx = self.output_copy_ring_idx[0]
        self.output_copy_ring_idx[0] = (slot_idx + 1) % len(self.output_copy_ring)
        slot = self.output_copy_ring[slot_idx]
        self.sampling_state.out_token_ids_B1_int32.copy_to(slot.host_buf, slot.event)
        slot.event.sync()
        raw = ctypes.string_at(int(slot.host_buf.handle), nbytes)
        return list(struct.unpack(f"<{n}i", raw[: 4 * n]))

    def _copy_tokens_t_by_b(self, token_cols: list[list[int]]):
        vals = []
        for tt in range(self.prefill_chunk_size):
            for b in range(self.batch_size):
                vals.append(token_cols[b][tt])
        self._copy_i32(self.input_token_ids_tB_int32, vals)

    @staticmethod
    def _copy_i32(tensor: Tensor, vals: list[int]):
        tensor.copy_from_bytes_sync(struct.pack(f"<{len(vals)}i", *vals))

    @staticmethod
    def _copy_u8(tensor: Tensor, vals: list[int]):
        tensor.copy_from_bytes_sync(bytes(vals))

    @record_static_cuda_graph
    def decode(self, sched: Scheduler):
        self._forward(t_now=1, phase="decode")

    @record_static_cuda_graph
    def prefill_chunk(self, sched: Scheduler):
        self._forward(t_now=self.prefill_chunk_size, phase="prefill")

    def _dbg(self, label: str, t: Tensor):
        if not os.environ.get("G4B_DBG"):
            return
        raw = t.to_bytes_sync()
        n = math.prod(t.shape)
        vals = struct.unpack(f"<{n}f", raw[: 4 * n])
        nan = sum(1 for v in vals if v != v)
        inf = sum(1 for v in vals if v == float("inf") or v == float("-inf"))
        finite = [v for v in vals if v == v and abs(v) != float("inf")]
        mn = min(finite) if finite else 0.0
        mx = max(finite) if finite else 0.0
        print(f"  [dbg] {label:32s} shape={list(t.shape)} nan={nan} inf={inf} min={mn:.4g} max={mx:.4g}")

    def _forward(self, t_now: int, phase: str):
        eps = self.rmsnorm_epsilon
        phase_id = PHASE_DECODE if phase == "decode" else PHASE_PREFILL
        residual = _slice_t(self.residual_BtD_dtr, t_now)
        act = _slice_t(self.layers[0].attn.shared_last_and_this_layer_output_scratchpad_BtD_dtr, t_now)
        act_rsos = _slice_t(self.residual_rsos_Bt_dtss, t_now)
        residual_rsos = act_rsos
        token_ids = self.input_token_ids_tB_int32.slice_until(0, t_now)

        kernels.embeddings.gather_token_embeddings(
            residual,
            residual_rsos,
            self.embeddings.embeddings_VD_q5,
            token_ids,
            math.sqrt(residual.shape[-1]),
            user_phase=self.user_phase_B_uint8,
            phase=phase_id,
        )
        self._dbg("after gather embed (residual)", residual)
        self._dbg("after gather embed (rsos)", residual_rsos)
        self._prepare_ple(token_ids, residual, t_now, phase_id)
        self._dbg("after prepare_ple (residual)", residual)
        # act = residual * input_weight (UNnormalized). The q/k/v matmul normalizes by rms(residual) via
        # input_rmsnorm_sum_of_squares=act_rsos, matching how later layers' act buffers are produced by
        # update_residual_stream. residual_rsos already holds sos(residual) (set by gather_token_embeddings).
        kernels.rmsnorm.apply_weight_out(
            residual,
            act,
            self.layers[0].attn.input_rmsnorm_w_D_fp32,
        )

        self._dbg("after first rmsnorm (act)", act)
        for i, layer in enumerate(self.layers):
            self._attention(layer.attn, act, residual, act_rsos, t_now, phase, phase_id)
            self._dbg(
                f"L{i} after attention (attn out)",
                _slice_t(layer.attn.shared_last_and_this_layer_output_scratchpad_BtD_dtr, t_now),
            )
            next_w = layer.mlp.input_rmsnorm_w_D_fp32
            kernels.update_residual_stream.update_residual_stream(
                residual,
                _slice_t(layer.attn.shared_last_and_this_layer_output_scratchpad_BtD_dtr, t_now),
                _slice_t(layer.attn.shared_last_and_this_layer_output_sum_of_squares_accum_scratchpad_Bt_dtss, t_now),
                act_rsos,
                layer.attn.o_rmsnorm_w_D_fp32,
                next_w,
                eps,
            )

            self._dbg(f"L{i} after attn resid (residual)", residual)
            self._mlp(layer.mlp, act, residual, act_rsos, t_now, phase_id)
            self._dbg(
                f"L{i} after mlp (mlp out)",
                _slice_t(layer.mlp.shared_last_and_this_layer_output_scratchpad_BtD_dtr, t_now),
            )
            kernels.update_residual_stream.update_residual_stream(
                residual,
                _slice_t(layer.mlp.shared_last_and_this_layer_output_scratchpad_BtD_dtr, t_now),
                _slice_t(layer.mlp.shared_last_and_this_layer_output_sum_of_squares_accum_scratchpad_Bt_dtss, t_now),
                act_rsos,
                layer.mlp.output_rmsnorm_w_D_fp32,
                self.identity_rmsnorm_w_D_fp32,
                eps,
            )

            self._dbg(f"L{i} after mlp resid (residual)", residual)
            self._ple(layer.ple, residual, act_rsos, i, t_now, phase_id)
            self._dbg(
                f"L{i} after ple (ple out)",
                _slice_t(layer.ple.shared_last_and_this_layer_output_scratchpad_BtD_dtr, t_now),
            )
            next_layer_w = (
                self.layers[i + 1].attn.input_rmsnorm_w_D_fp32
                if i + 1 < len(self.layers)
                else self.lm_head.rmsnorm_w_D_fp32
            )
            kernels.update_residual_stream.update_residual_stream(
                residual,
                _slice_t(layer.ple.shared_last_and_this_layer_output_scratchpad_BtD_dtr, t_now),
                _slice_t(layer.ple.shared_last_and_this_layer_output_sum_of_squares_accum_scratchpad_Bt_dtss, t_now),
                act_rsos,
                layer.ple.output_rmsnorm_w_D_fp32,
                next_layer_w,
                eps,
                layer.layer_output_scale_1_fp32,
            )
            self._dbg(f"L{i} after ple resid (residual)", residual)
            self._dbg(f"L{i} act_rsos", act_rsos)
            if _dbg_capture_layer_residuals():
                kernels.select_t.select_t(
                    residual, self.sample_positions_B_int32, self.debug_layer_residuals_LB1D_dtr.slice_at(0, i)
                )
                kernels.select_t.select_t(
                    act_rsos.reshape((act_rsos.shape[0], act_rsos.shape[1], 1)),
                    self.sample_positions_B_int32,
                    self.debug_layer_rsos_LB1_dtss.slice_at(0, i).reshape(
                        (self.debug_layer_rsos_LB1_dtss.shape[1], 1, 1)
                    ),
                )

        self._dbg("final act_rsos (for logits)", act_rsos)
        kernels.select_t.select_t(act, self.sample_positions_B_int32, self.lm_head.input_B1D_dtr)
        kernels.select_t.select_t(
            act_rsos.reshape((act_rsos.shape[0], act_rsos.shape[1], 1)),
            self.sample_positions_B_int32,
            self.lm_head.input_rsos_B1_dtss.reshape((self.lm_head.input_rsos_B1_dtss.shape[0], 1, 1)),
        )
        matmul_a3d_b2d(
            self.lm_head.logits_B1V_dtsamp,
            None,
            self.lm_head.input_B1D_dtr,
            self.embeddings.embeddings_VD_q5,
            transpose_b_before_mma=True,
            input_rmsnorm_sum_of_squares=self.lm_head.input_rsos_B1_dtss,
            accum_dtype=DTMM_ACCUM_SENSITIVE.tl_dtype,
            user_phase=self.user_phase_B_uint8,
            phase=phase_id,
            rmsnorm_eps=eps,
        )
        self._dbg("logits", self.lm_head.logits_B1V_dtsamp)
        kernels.sample_logits.sample_logits(
            self.lm_head.logits_B1V_dtsamp,
            self.sampling_state.out_token_ids_B1_int32,
            self.sampling_state.seed_2_int32,
            self.sampling_state.top_k_logits_scratchpad_B1__num_splits__top_k__dtsamp,
            self.sampling_state.top_k_idx_scratchpad_B1__num_splits__top_k__int32,
            self.sampling_state.temperature,
            self.sampling_state.top_k,
            self.sampling_state.top_p,
            self.sampling_state.top_k_logits_scratchpad_B1__num_splits__top_k__dtsamp.shape[2],
            logit_softcap=self.lm_head.logit_softcap,
            user_phase=self.user_phase_B_uint8,
            phase=phase_id,
        )
        if phase == "decode":
            kernels.advance_decode_state.advance_decode_state(
                self.input_token_ids_tB_int32,
                self.sampling_state.out_token_ids_B1_int32,
                self.cache_offsets_B_int32,
                self.time_dim_sizes_B_int32,
                self.user_phase_B_uint8,
            )

    def _prepare_ple(self, token_ids: Tensor, residual: Tensor, t_now: int, phase_id: int):
        B, _, D = residual.shape
        L = len(self.layers)
        P = self.embeddings.ple_rmsnorm_w_P_fp32.shape[0]
        # NOTE: ple_lookup and ple_proj MUST be separate buffers; they previously aliased the same buffer so the
        # matmul overwrote the gathered lookup before the add (computing proj+norm(proj) instead of lookup+norm(proj)).
        ple_lookup_flat = _slice_t(self.embeddings.ple_lookup_BtLP_dtple, t_now)
        ple_proj = _slice_t(self.embeddings.ple_proj_BtLP_dtple, t_now)
        ple_table = self.embeddings.ple_table_VLP_q5.reshape((self.embeddings.ple_table_VLP_q5.shape[0], L * P))
        # ple_proj_DLP_bf16 is stored conventionally as [L*P, D]; matmul computes residual @ W^T.
        ple_proj_w = self.embeddings.ple_proj_LPD_bf16
        # reference: ple_lookup = per_layer_embeddings[ids] * sqrt(ple_dim)
        kernels.embeddings.gather_token_embeddings(
            ple_lookup_flat,
            None,
            ple_table,
            token_ids,
            math.sqrt(P),
            user_phase=self.user_phase_B_uint8,
            phase=phase_id,
        )
        matmul_a3d_b2d(
            ple_proj,
            None,
            residual,
            ple_proj_w,
            transpose_b_before_mma=True,
            accum_dtype=DTMM_ACCUM_SENSITIVE.tl_dtype,
            user_phase=self.user_phase_B_uint8,
            phase=phase_id,
            rmsnorm_eps=self.rmsnorm_epsilon,
        )
        ple_proj_4d = ple_proj.reshape((B, t_now, L, P))
        ple_lookup_4d = ple_lookup_flat.reshape((B, t_now, L, P))
        ple_rsos = _slice_t(self.embeddings.ple_proj_rsos_BtL_dtss, t_now)
        # rmsnorm is scale-invariant, so the reference's D^-0.5 on ple_proj cancels: just rmsnorm the raw projection.
        kernels.rsos.compute_rsos(ple_proj_4d, ple_rsos, scale=1.0)
        for layer_idx in range(L):
            kernels.elementwise_ops.add(
                ple_lookup_4d.slice_at(2, layer_idx),
                ple_proj_4d.slice_at(2, layer_idx),
                _slice_t(self.embeddings.ple_LBtP_dtple.slice_at(0, layer_idx), t_now),
                output_scale_factor=2**-0.5,
                rmsnorm_eps=self.rmsnorm_epsilon,
                b_rsos=ple_rsos.slice_at(2, layer_idx),
                b_rmsnorm_w=self.embeddings.ple_rmsnorm_w_P_fp32,
            )

    def _attention(
        self, attn: Attention, act: Tensor, residual: Tensor, act_rsos: Tensor, t_now: int, phase: str, phase_id: int
    ):
        q_flat = _flat_head_out(attn.shared_q_scratchpad_Bhtk_dta, t_now)
        k_flat = _flat_head_out(attn.shared_k_scratchpad_Bgtk_dta, t_now)
        v_flat = _flat_head_out(attn.shared_v_scratchpad_Bgtv_dta, t_now)
        o_flat = _flat_head_out(attn.shared_o_proj_input_scratchpad_Bhtv_dta, t_now)
        # rsos buffers are physically [B, t, n_heads]; compute_rsos needs a contiguous output whose shape
        # matches the (contiguous, physical [B, t, H, D]) scratchpad's leading dims. We then hand a permuted
        # [B, n_heads, t] *view* to rope / add_kv (those kernels read rsos purely via strides).
        q_rsos_phys = _slice_t(attn.shared_q_rsos_Bth_dtss, t_now)
        k_rsos_phys = _slice_t(attn.shared_k_rsos_Btg_dtss, t_now)
        v_rsos_phys = _slice_t(attn.shared_v_rsos_Btg_dtss, t_now)
        q_rsos = q_rsos_phys.permute((0, 2, 1))
        k_rsos = k_rsos_phys.permute((0, 2, 1))
        v_rsos = v_rsos_phys.permute((0, 2, 1))
        out_rsos = _slice_t(attn.shared_last_and_this_layer_output_sum_of_squares_accum_scratchpad_Bt_dtss, t_now)
        out = _slice_t(attn.shared_last_and_this_layer_output_scratchpad_BtD_dtr, t_now)

        # q/k/v projections fuse, in the matmul epilogue storer: (1) the input rmsnorm (using resid sos act_rsos)
        # and (2) the PER-HEAD output sum-of-squares written into the [B,t,n_heads] rsos buffers (rsos_head_dim is
        # the head size). This replaces 3 separate compute_rsos launches per layer.
        matmul_a3d_b2d(
            q_flat,
            q_rsos_phys,
            act,
            attn.q_proj_hkD_q4,
            storer_fn=kernels.matmul_epilogue.qkv_input_rmsnorm_per_head_rsos_storer_jfn,
            transpose_b_before_mma=True,
            input_rmsnorm_sum_of_squares=act_rsos,
            rsos_head_dim=attn.head_size_qk,
            accum_dtype=DTMM_ACCUM_NORMAL.tl_dtype,
            user_phase=self.user_phase_B_uint8,
            phase=phase_id,
            rmsnorm_eps=self.rmsnorm_epsilon,
        )
        matmul_a3d_b2d(
            k_flat,
            k_rsos_phys,
            act,
            attn.k_proj_gkD_q6,
            storer_fn=kernels.matmul_epilogue.qkv_input_rmsnorm_per_head_rsos_storer_jfn,
            transpose_b_before_mma=True,
            input_rmsnorm_sum_of_squares=act_rsos,
            rsos_head_dim=attn.head_size_qk,
            accum_dtype=DTMM_ACCUM_NORMAL.tl_dtype,
            user_phase=self.user_phase_B_uint8,
            phase=phase_id,
            rmsnorm_eps=self.rmsnorm_epsilon,
        )
        matmul_a3d_b2d(
            v_flat,
            v_rsos_phys,
            act,
            attn.v_proj_gvD_q6,
            storer_fn=kernels.matmul_epilogue.qkv_input_rmsnorm_per_head_rsos_storer_jfn,
            transpose_b_before_mma=True,
            input_rmsnorm_sum_of_squares=act_rsos,
            rsos_head_dim=attn.head_size_v,
            accum_dtype=DTMM_ACCUM_NORMAL.tl_dtype,
            user_phase=self.user_phase_B_uint8,
            phase=phase_id,
            rmsnorm_eps=self.rmsnorm_epsilon,
        )
        q = _attn_view(attn.shared_q_scratchpad_Bhtk_dta, t_now)
        k = _attn_view(attn.shared_k_scratchpad_Bgtk_dta, t_now)
        v = _attn_view(attn.shared_v_scratchpad_Bgtv_dta, t_now)
        o = _attn_view(attn.shared_o_proj_input_scratchpad_Bhtv_dta, t_now)
        partial_o = attn.shared_fa_partial_o_SBHtv_f16.slice_until(3, t_now) if phase == "decode" else None
        partial_l = attn.shared_fa_partial_l_SBHt_fp32.slice_until(3, t_now) if phase == "decode" else None
        partial_m = attn.shared_fa_partial_m_SBHt_fp32.slice_until(3, t_now) if phase == "decode" else None
        kernels.rope.apply_rope(
            q,
            k,
            attn.rope_freqs_k_fp32,
            self.cache_offsets_B_int32,
            q_rsos,
            k_rsos,
            attn.q_rmsnorm_w_k_fp32,
            attn.k_rmsnorm_w_k_fp32,
            self.rmsnorm_epsilon,
        )
        if attn.owns_kv_cache:
            kernels.add_kv_to_cache.add_kv_to_cache(
                k,
                attn.k_cache_Bg__T_or_W__k__dtkv,
                self.cache_offsets_B_int32,
                self.rmsnorm_epsilon,
                user_phase=self.user_phase_B_uint8,
                phase=phase_id,
            )
            kernels.add_kv_to_cache.add_kv_to_cache(
                v,
                attn.v_cache_Bg__T_or_W__v__dtkv,
                self.cache_offsets_B_int32,
                self.rmsnorm_epsilon,
                v_rsos,
                user_phase=self.user_phase_B_uint8,
                phase=phase_id,
            )
        kernels.fa2.flash_attention(
            q,
            attn.k_cache_Bg__T_or_W__k__dtkv,
            attn.v_cache_Bg__T_or_W__v__dtkv,
            o,
            self.time_dim_sizes_B_int32,
            self.user_phase_B_uint8,
            attn.context_window_size,
            phase_id,
            partial_o,
            partial_l,
            partial_m,
            use_grouped_query_tile=False,
            use_fp32_dot=DTFA_ACCUM_NORMAL == float32 or bool(os.environ.get("G4B_FA_FP32_DOT")),
        )
        matmul_a3d_b2d(
            out,
            out_rsos,
            o_flat,
            attn.o_proj_Dhv_q4,
            transpose_b_before_mma=True,
            accum_dtype=DTMM_ACCUM_NORMAL.tl_dtype,
            user_phase=self.user_phase_B_uint8,
            phase=phase_id,
            rmsnorm_eps=self.rmsnorm_epsilon,
        )

    def _mlp(self, mlp: MLP, act: Tensor, residual: Tensor, act_rsos: Tensor, t_now: int, phase_id: int):
        h = _slice_t(mlp.shared_down_proj_input_scratchpad_BtU_dth, t_now)
        out = _slice_t(mlp.shared_last_and_this_layer_output_scratchpad_BtD_dtr, t_now)
        out_rsos = _slice_t(mlp.shared_last_and_this_layer_output_sum_of_squares_accum_scratchpad_Bt_dtss, t_now)
        matmul_a3d_b2d(
            h,
            None,
            act,
            mlp.up_proj_UD_q4,
            mlp.gate_proj_UD_q4,
            c_c2_merge_tiles_fn=kernels.matmul_epilogue.geglu_fusion_matmul_merge_tiles_mixin_jfn,
            transpose_b_before_mma=True,
            input_rmsnorm_sum_of_squares=act_rsos,
            accum_dtype=DTMM_ACCUM_NORMAL.tl_dtype,
            user_phase=self.user_phase_B_uint8,
            phase=phase_id,
            rmsnorm_eps=self.rmsnorm_epsilon,
        )
        matmul_a3d_b2d(
            out,
            out_rsos,
            h,
            mlp.down_proj_DU_q6,
            transpose_b_before_mma=True,
            accum_dtype=DTMM_ACCUM_NORMAL.tl_dtype,
            user_phase=self.user_phase_B_uint8,
            phase=phase_id,
            rmsnorm_eps=self.rmsnorm_epsilon,
        )

    def _ple(
        self,
        ple: PerLayerEmbeddings,
        residual: Tensor,
        act_rsos: Tensor,
        layer_idx: int,
        t_now: int,
        phase_id: int,
    ):
        ple_vals = _slice_t(self.embeddings.ple_LBtP_dtple.slice_at(0, layer_idx), t_now)
        h = _slice_t(ple.shared_out_proj_input_scratchpad_BtP_dtple, t_now)
        out = _slice_t(ple.shared_last_and_this_layer_output_scratchpad_BtD_dtr, t_now)
        out_rsos = _slice_t(ple.shared_last_and_this_layer_output_sum_of_squares_accum_scratchpad_Bt_dtss, t_now)
        # reference PLE applies gate_proj to the RAW residual (no input rmsnorm): gate = gelu(gate_proj(x)).
        matmul_a3d_b2d(
            h,
            None,
            residual,
            ple.gate_proj_PD_fp32,
            storer_extra=ple_vals,
            storer_fn=kernels.matmul_epilogue.ple_gate_storer_jfn,
            transpose_b_before_mma=True,
            accum_dtype=DTMM_ACCUM_SENSITIVE.tl_dtype,
            user_phase=self.user_phase_B_uint8,
            phase=phase_id,
            rmsnorm_eps=self.rmsnorm_epsilon,
        )
        matmul_a3d_b2d(
            out,
            out_rsos,
            h,
            ple.out_proj_DP_fp32,
            transpose_b_before_mma=True,
            accum_dtype=DTMM_ACCUM_SENSITIVE.tl_dtype,
            user_phase=self.user_phase_B_uint8,
            phase=phase_id,
            rmsnorm_eps=self.rmsnorm_epsilon,
        )

    @classmethod
    def load(cls, meta: GGUFMeta, tensors: list[GGUFTensor], config: Config):
        _check_meta(meta)
        _check_dtypes_supported(tensors)
        if config.context_len > meta["gemma4.context_length"]:
            warnings.warn("configured context length exceeds maximum context length specified in GGUF file")

        # define important sizes
        B = config.batch_size
        t = config.prefill_chunk_size
        T = config.context_len + t - 1
        W = meta["gemma4.attention.sliding_window"] + t - 1
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
        out_token_ids = Tensor.alloc_empty(int32, [B, 1])
        num_sampling_splits = kernels.sample_logits.get_recommended_num_v_splits(V)
        top_k_logits_scratchpad = Tensor.alloc_empty(DTSAMP, [B, 1, num_sampling_splits, top_k])
        top_k_idx_scratchpad = Tensor.alloc_empty(int32, [B, 1, num_sampling_splits, top_k])
        seed = Tensor.alloc_empty(int32, [2])
        seed.copy_from_bytes_sync(struct.pack("<2i", 12345, 0))
        sampling_state = SamplingState(
            out_token_ids_B1_int32=out_token_ids,
            top_k_logits_scratchpad_B1__num_splits__top_k__dtsamp=top_k_logits_scratchpad,
            top_k_idx_scratchpad_B1__num_splits__top_k__int32=top_k_idx_scratchpad,
            seed_2_int32=seed,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )
        output_copy_nbytes = int32.sizeof_tensor(out_token_ids.shape)
        output_copy_ring = tuple(
            OutputCopySlot(device.alloc_pinned_host(output_copy_nbytes), device.event()) for _ in range(4)
        )

        # embeddings
        token_embeddings = Tensor.from_gguf_tensor(ts["token_embd.weight"])
        ple_table = Tensor.from_gguf_tensor(ts["per_layer_token_embd.weight"])
        ple_proj = Tensor.from_gguf_tensor(ts["per_layer_model_proj.weight"])
        ple_norm = Tensor.from_gguf_tensor(ts["per_layer_proj_norm.weight"])
        ple_buf = Tensor.alloc_empty(DTPLE, [L, B, t, P])
        ple_proj_buf = Tensor.alloc_empty(DTPLE, [B, t, L * P])
        ple_lookup_buf = Tensor.alloc_empty(DTPLE, [B, t, L * P])
        ple_proj_rsos = Tensor.alloc_empty(DTSS, [B, t, L])
        embeddings = Embeddings(
            embeddings_VD_q5=token_embeddings,
            ple_table_VLP_q5=ple_table,
            ple_proj_LPD_bf16=ple_proj,
            ple_rmsnorm_w_P_fp32=ple_norm,
            ple_LBtP_dtple=ple_buf,
            ple_proj_BtLP_dtple=ple_proj_buf,
            ple_lookup_BtLP_dtple=ple_lookup_buf,
            ple_proj_rsos_BtL_dtss=ple_proj_rsos,
        )

        # lm head
        head_rmsnorm_w = Tensor.from_gguf_tensor(ts["output_norm.weight"])
        lm_head_input = Tensor.alloc_empty(DTR, [B, 1, D])
        lm_head_input_rsos = Tensor.alloc_empty(DTSS, [B, 1])
        logits_buf = Tensor.alloc_empty(DTSAMP, [B, 1, V])
        logit_softcap = meta["gemma4.final_logit_softcapping"]
        assert isinstance(logit_softcap, float)
        lm_head = LmHead(
            rmsnorm_w_D_fp32=head_rmsnorm_w,
            input_B1D_dtr=lm_head_input,
            input_rsos_B1_dtss=lm_head_input_rsos,
            logits_B1V_dtsamp=logits_buf,
            logit_softcap=logit_softcap,
        )

        # create global state tensors
        residual = Tensor.alloc_empty(DTR, [B, t, D])
        residual_rsos = Tensor.alloc_empty(DTSS, [B, t])
        debug_layer_residuals = Tensor.alloc_empty(DTR, [L, B, 1, D]) if _dbg_capture_layer_residuals() else None
        debug_layer_rsos = Tensor.alloc_empty(DTSS, [L, B, 1]) if _dbg_capture_layer_residuals() else None
        input_token_ids = Tensor.alloc_empty(int32, [t, B])
        sample_positions = Tensor.alloc_empty(int32, [B])
        cache_offsets = Tensor.alloc_empty(int32, [B])
        time_dim_sizes = Tensor.alloc_empty(int32, [B])
        memset_contiguous(input_token_ids, 0)
        memset_contiguous(sample_positions, 0)
        memset_contiguous(cache_offsets, 0)
        memset_contiguous(time_dim_sizes, 0)
        user_phase = Tensor.alloc_empty(uint8, [B])
        memset_contiguous(user_phase, 0)
        rmsnorm_epsilon = meta["gemma4.attention.layer_norm_rms_epsilon"]
        assert isinstance(rmsnorm_epsilon, float)
        identity_rmsnorm_w = Tensor.from_bytes_sync(struct.pack(f"<{D}f", *([1.0] * D)), float32, [D])

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
        q_gqa_scratchpad = Tensor.alloc_empty(DTA, [B, t, h, k_gqa])
        k_gqa_scratchpad = Tensor.alloc_empty(DTA, [B, t, g, k_gqa])
        v_gqa_scratchpad = Tensor.alloc_empty(DTA, [B, t, g, v_gqa])
        o_gqa_proj_input_scratchpad = Tensor.alloc_empty(DTA, [B, t, h, v_gqa])
        q_gqa_rsos = Tensor.alloc_empty(DTSS, [B, t, h])
        k_gqa_rsos = Tensor.alloc_empty(DTSS, [B, t, g])
        v_gqa_rsos = Tensor.alloc_empty(DTSS, [B, t, g])
        fa_gqa_partial_o = Tensor.alloc_empty(float16, [FA_MAX_KV_SPLITS, B, h, t, v_gqa])
        fa_gqa_partial_l = Tensor.alloc_empty(float32, [FA_MAX_KV_SPLITS, B, h, t])
        fa_gqa_partial_m = Tensor.alloc_empty(float32, [FA_MAX_KV_SPLITS, B, h, t])
        q_swa_scratchpad = Tensor.alloc_empty(DTA, [B, t, h, k_swa])
        k_swa_scratchpad = Tensor.alloc_empty(DTA, [B, t, g, k_swa])
        v_swa_scratchpad = Tensor.alloc_empty(DTA, [B, t, g, v_swa])
        o_swa_proj_input_scratchpad = Tensor.alloc_empty(DTA, [B, t, h, v_swa])
        q_swa_rsos = Tensor.alloc_empty(DTSS, [B, t, h])
        k_swa_rsos = Tensor.alloc_empty(DTSS, [B, t, g])
        v_swa_rsos = Tensor.alloc_empty(DTSS, [B, t, g])
        fa_swa_partial_o = Tensor.alloc_empty(float16, [FA_MAX_KV_SPLITS, B, h, t, v_swa])
        fa_swa_partial_l = Tensor.alloc_empty(float32, [FA_MAX_KV_SPLITS, B, h, t])
        fa_swa_partial_m = Tensor.alloc_empty(float32, [FA_MAX_KV_SPLITS, B, h, t])
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
            context_window_size = (
                min(meta["gemma4.attention.sliding_window"], config.context_len) if is_swa else config.context_len
            )
            assert isinstance(context_window_size, int)

            owns_kv_cache = L - i > meta["gemma4.attention.shared_kv_layers"]
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
                    q_proj_hkD_q4=q_proj,
                    k_proj_gkD_q6=k_proj,
                    v_proj_gvD_q6=v_proj,
                    o_proj_Dhv_q4=o_proj,
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
                    shared_q_rsos_Bth_dtss=q_swa_rsos,
                    shared_k_rsos_Btg_dtss=k_swa_rsos,
                    shared_v_rsos_Btg_dtss=v_swa_rsos,
                    shared_o_proj_input_scratchpad_Bhtv_dta=o_swa_proj_input_scratchpad,
                    shared_fa_partial_o_SBHtv_f16=fa_swa_partial_o,
                    shared_fa_partial_l_SBHt_fp32=fa_swa_partial_l,
                    shared_fa_partial_m_SBHt_fp32=fa_swa_partial_m,
                    shared_last_and_this_layer_output_scratchpad_BtD_dtr=layer_output_scratchpad,
                    shared_last_and_this_layer_output_sum_of_squares_accum_scratchpad_Bt_dtss=layer_output_sos_scratchpad,
                )
            else:
                attn = Attention(
                    q_proj_hkD_q4=q_proj,
                    k_proj_gkD_q6=k_proj,
                    v_proj_gvD_q6=v_proj,
                    o_proj_Dhv_q4=o_proj,
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
                    shared_q_rsos_Bth_dtss=q_gqa_rsos,
                    shared_k_rsos_Btg_dtss=k_gqa_rsos,
                    shared_v_rsos_Btg_dtss=v_gqa_rsos,
                    shared_o_proj_input_scratchpad_Bhtv_dta=o_gqa_proj_input_scratchpad,
                    shared_fa_partial_o_SBHtv_f16=fa_gqa_partial_o,
                    shared_fa_partial_l_SBHt_fp32=fa_gqa_partial_l,
                    shared_fa_partial_m_SBHt_fp32=fa_gqa_partial_m,
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
                up_proj_UD_q4=up_proj,
                gate_proj_UD_q4=gate_proj,
                down_proj_DU_q6=down_proj,
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
                gate_proj_PD_fp32=ple_gate_proj,
                out_proj_DP_fp32=ple_out_proj,
                output_rmsnorm_w_D_fp32=ple_out_norm,
                shared_out_proj_input_scratchpad_BtP_dtple=ple_out_proj_input_scratchpad,
                shared_last_and_this_layer_output_scratchpad_BtD_dtr=layer_output_scratchpad,
                shared_last_and_this_layer_output_sum_of_squares_accum_scratchpad_Bt_dtss=layer_output_sos_scratchpad,
            )

            # construct layer
            layer_output_scale = by_name("layer_output_scale")
            layer = DecoderLayer(attn=attn, mlp=mlp, ple=ple, layer_output_scale_1_fp32=layer_output_scale)
            layers.append(layer)

        model = Gemma4E(
            embeddings=embeddings,
            layers=layers,
            lm_head=lm_head,
            sampling_state=sampling_state,
            rmsnorm_epsilon=rmsnorm_epsilon,
            identity_rmsnorm_w_D_fp32=identity_rmsnorm_w,
            residual_BtD_dtr=residual,
            residual_rsos_Bt_dtss=residual_rsos,
            debug_layer_residuals_LB1D_dtr=debug_layer_residuals,
            debug_layer_rsos_LB1_dtss=debug_layer_rsos,
            input_token_ids_tB_int32=input_token_ids,
            sample_positions_B_int32=sample_positions,
            cache_offsets_B_int32=cache_offsets,
            time_dim_sizes_B_int32=time_dim_sizes,
            user_phase_B_uint8=user_phase,
            output_copy_ring=output_copy_ring,
            output_copy_ring_idx=[0],
            batch_size=B,
            prefill_chunk_size=t,
            eos_token_id=meta["tokenizer.ggml.eos_token_id"],
        )
        device.sync_all_streams()
        return model


def _compute_rope_freqs(rope_freqs: GGUFTensor | None, freq_base: float, rope_dim_count: int) -> Tensor:
    assert rope_dim_count % 2 == 0
    assert rope_freqs is None or rope_freqs.shape[-1] == rope_dim_count // 2
    freqs = Tensor.from_gguf_tensor(rope_freqs) if rope_freqs is not None else None
    out = Tensor.alloc_empty(float32, [rope_dim_count // 2])
    kernels.rope.populate_rope_frequencies(out, freqs, freq_base)
    return out


def _slice_t(x: Tensor, t_now: int) -> Tensor:
    return x.slice_until(1, t_now)


def _slice_heads_t(x: Tensor, t_now: int) -> Tensor:
    return x.slice_until(2, t_now)


def _flat_head_out(x: Tensor, t_now: int) -> Tensor:
    # x is physically [B, t, H, D]; matmul wants [B, t, H*D].
    x = _slice_t(x, t_now)
    return x.reshape((x.shape[0], x.shape[1], x.shape[2] * x.shape[3]))


def _attn_view(x: Tensor, t_now: int) -> Tensor:
    # FA/add_kv kernels want [B, H, t, D].
    return _slice_t(x, t_now).permute((0, 2, 1, 3))


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


def _dbg_capture_layer_residuals() -> bool:
    return bool(os.environ.get("G4B_CAPTURE_LAYER_RESIDUALS"))
