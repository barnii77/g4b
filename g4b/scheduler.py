import asyncio
import os
import time
from queue import SimpleQueue
from typing import TYPE_CHECKING
from g4b.utils import shared_prefix_length, floor_to_multiple_of

if TYPE_CHECKING:
    import g4b.models
    import g4b.tokenizer


class Request:
    def __init__(
        self,
        input_tokens: list[int],
        change_cv: asyncio.Condition | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
        initial_context_len: int = 0,
        max_context_len: int | None = None,
    ):
        self.input_tokens = input_tokens
        self._output_tokens: list[int] = []
        self._prev_retrieve_last_token_idx = 0
        self._change_cv = change_cv
        self._loop = loop
        self._prefill_pos = 0
        self._context_len = initial_context_len
        self._max_context_len = max_context_len
        self._context_window_exceeded = False
        self._last_sample_t_idx = 0
        self._decode_state_prepared = False
        self._done = False

    def get_new_tokens(self) -> list[int]:
        out = self._output_tokens[self._prev_retrieve_last_token_idx :]
        self._prev_retrieve_last_token_idx = len(self._output_tokens)
        return out


class Scheduler:
    _STATS_INTERVAL = 16

    def __init__(self, model: "g4b.models.Model", tokenizer: "g4b.tokenizer.Tokenizer"):
        self._model = model
        self._tokenizer = tokenizer
        self._queue: SimpleQueue[Request] = SimpleQueue()
        self._active: list[Request | None] = [None] * model.max_batch_size()
        self._prev_processed_tokens_by_slot = [[] for _ in range(model.max_batch_size())]
        self._phase_active: list[bool] = [False] * model.max_batch_size()
        self._last_phase: str | None = None

        self._print_stats = bool(os.environ.get("G4B_PRINT_STATS"))
        self._total_prefill_tokens = 0
        self._total_decode_tokens = 0
        self._window_prefill_tokens = 0
        self._window_decode_tokens = 0
        self._tokens_since_log = 0
        self._last_log_time = time.perf_counter()

    def step(self):
        self._fill_free_slots()
        if not any(rq is not None and not rq._done for rq in self._active):
            return

        phase = "decode" if any(self._needs_decode(rq) for rq in self._active if rq is not None) else "prefill"
        prefill_tokens = 0
        decode_tokens = 0
        if phase == "prefill":
            prefill_tokens = self._prepare_prefill_inputs()
            self._model.prefill_chunk(self)
        else:
            decode_tokens = self._prepare_decode_inputs()
            self._model.decode(self)
        self._last_phase = phase
        self._collect_outputs(phase)
        self._drop_done_slots()
        if self._print_stats:
            self._update_stats(prefill_tokens, decode_tokens)

    def submit(self, request: Request):
        self._queue.put(request)

    def reset(self):
        self._queue = SimpleQueue()
        self._active = [None] * self._model.max_batch_size()
        self._phase_active = [False] * self._model.max_batch_size()
        self._prev_processed_tokens_by_slot = [[] for _ in range(self._model.max_batch_size())]

    def abort(self, request: Request):
        request._done = True
        for i, active in enumerate(self._active):
            if active is request:
                self._active[i] = None

    def _fill_free_slots(self):
        if any(rq is not None and self._needs_decode(rq) for rq in self._active):
            return

        slot_is_free = lambda rq: rq is None or rq._done

        n_free_slots = sum(1 for rq in self._active if slot_is_free(rq))
        for _ in range(n_free_slots):
            if self._queue.empty():
                break
            new_rq = self._queue.get()
            assert new_rq._prefill_pos == 0

            # schedule new request in the slot which has the longest shared prefix with the new request
            shared_prefix_len_by_slot = [
                shared_prefix_length(prev_tokens, new_rq.input_tokens) if slot_is_free(rq) else -1
                for prev_tokens, rq in zip(self._prev_processed_tokens_by_slot, self._active)
            ]
            ideal_slot = max(range(len(shared_prefix_len_by_slot)), key=lambda i: shared_prefix_len_by_slot[i])
            shared_prefix_len = shared_prefix_len_by_slot[ideal_slot]
            assert shared_prefix_len >= 0

            # skip prefilling fully shared chunks except the last one because it samples a token and it must resample
            # TODO that is inefficient though... I should still share the last chunk and just retrigger sampling
            n = max(0, floor_to_multiple_of(shared_prefix_len - 1, self._model.max_prefill_chunk_size()))
            new_rq._prefill_pos = n
            new_rq._context_len += n

            self._active[ideal_slot] = new_rq
            self._prev_processed_tokens_by_slot[ideal_slot] = new_rq.input_tokens.copy()

    def _needs_prefill(self, rq: Request) -> bool:
        return not rq._done and rq._prefill_pos < len(rq.input_tokens)

    def _needs_decode(self, rq: Request) -> bool:
        return not rq._done and not self._needs_prefill(rq)

    def _prepare_prefill_inputs(self) -> int:
        t = self._model.max_prefill_chunk_size()
        token_cols: list[list[int]] = []
        cache_offsets: list[int] = []
        time_sizes_after: list[int] = []
        sample_positions: list[int] = []
        phase_active: list[bool] = []
        total_real = 0
        for rq in self._active:
            if rq is None or not self._needs_prefill(rq):
                token_cols.append([0] * t)
                cache_offsets.append(0)
                time_sizes_after.append(0)
                sample_positions.append(0)
                phase_active.append(False)
                continue

            phase_active.append(True)
            start = rq._prefill_pos
            end = min(start + t, len(rq.input_tokens))
            real_n = max(1, end - start)
            total_real += real_n
            toks = rq.input_tokens[start:end]
            if not toks:
                toks = [rq.input_tokens[max(0, len(rq.input_tokens) - 1)]]
            toks = toks + [toks[-1]] * (t - len(toks))
            token_cols.append(toks)
            start_offset = rq._context_len
            cache_offsets.append(start_offset)
            rq._prefill_pos = end
            rq._context_len += real_n
            rq._last_sample_t_idx = real_n - 1
            sample_positions.append(rq._last_sample_t_idx)
            rq._decode_state_prepared = False
            # The model always processes a full t-wide chunk of query positions and writes t KV slots
            # starting at start_offset, so the attention window must span all t of them (start_offset + t),
            # NOT just the real_n valid tokens. FA derives q_t_base = window_size - t; using real_n here
            # would make q_t_base negative for a short/padded prompt, so every query attends to zero keys
            # and produces 0/0 = NaN. _context_len still advances by real_n for the next step's offset.
            time_sizes_after.append(start_offset + t)

        self._phase_active = phase_active
        self._model.prepare_prefill_inputs(token_cols, cache_offsets, time_sizes_after, sample_positions)
        return total_real

    def _prepare_decode_inputs(self) -> int:
        t = self._model.max_prefill_chunk_size()
        token_cols: list[list[int]] = []
        cache_offsets: list[int] = []
        time_sizes_after: list[int] = []
        sample_positions: list[int] = []
        phase_active: list[bool] = []
        needs_upload = False
        total_decode = 0
        for rq in self._active:
            if rq is None or not self._needs_decode(rq):
                token_cols.append([0] * t)
                cache_offsets.append(0)
                time_sizes_after.append(0)
                sample_positions.append(0)
                phase_active.append(False)
                continue
            phase_active.append(True)
            total_decode += 1
            tok = rq.input_tokens[-1] if not rq._output_tokens else rq._output_tokens[-1]
            token_cols.append([tok] * t)
            cache_offsets.append(rq._context_len)
            needs_upload = needs_upload or not rq._decode_state_prepared
            rq._context_len += 1
            rq._last_sample_t_idx = 0
            time_sizes_after.append(rq._context_len)
            sample_positions.append(0)

        self._phase_active = phase_active
        if needs_upload:
            self._model.prepare_decode_inputs(token_cols, cache_offsets, time_sizes_after, sample_positions)
            for rq in self._active:
                if rq is not None and not rq._done and not self._needs_prefill(rq):
                    rq._decode_state_prepared = True
        return total_decode

    def _update_stats(self, prefill_tokens: int, decode_tokens: int):
        self._total_prefill_tokens += prefill_tokens
        self._total_decode_tokens += decode_tokens
        self._window_prefill_tokens += prefill_tokens
        self._window_decode_tokens += decode_tokens
        self._tokens_since_log += prefill_tokens + decode_tokens
        if self._tokens_since_log >= self._STATS_INTERVAL:
            self._log_stats()
            self._tokens_since_log %= self._STATS_INTERVAL

    def _log_stats(self):
        now = time.perf_counter()
        elapsed = now - self._last_log_time
        prefill_tps = self._window_prefill_tokens / elapsed if elapsed > 0 else 0.0
        decode_tps = self._window_decode_tokens / elapsed if elapsed > 0 else 0.0
        self._window_prefill_tokens = 0
        self._window_decode_tokens = 0
        self._last_log_time = now
        active = sum(1 for rq in self._active if rq is not None and not rq._done)
        active_reqs = [rq._context_len for rq in self._active if rq is not None]
        context_len = max(active_reqs) if active_reqs else -1
        print(
            f"stats tokens={self._total_prefill_tokens + self._total_decode_tokens} "
            f"tps_prefill={prefill_tps:.2f} tps_decode={decode_tps:.2f} "
            f"active={active} context_len={context_len}",
            flush=True,
        )

    def _collect_outputs(self, phase: str):
        vals = self._model.collect_output_token_ids()
        for b, rq in enumerate(self._active):
            if rq is None or rq._done or not self._phase_active[b]:
                continue
            if phase == "prefill" and self._needs_prefill(rq):
                continue
            tok = vals[b]
            rq._output_tokens.append(tok)
            self._prev_processed_tokens_by_slot[b].append(tok)
            if tok in self._tokenizer.gen_ending_tokens():
                rq._done = True
            elif rq._max_context_len is not None and rq._context_len >= rq._max_context_len:
                # The sampled token is valid, but processing it on the next
                # decode step would roll the global KV window and discard
                # context. Terminate before that quality-degrading step.
                rq._context_window_exceeded = True
                rq._done = True
            self._notify(rq)

    def _drop_done_slots(self):
        for i, rq in enumerate(self._active):
            if rq is not None and rq._done:
                self._active[i] = None
                # preserve list of tokens in this slot's KV caches so it can be (partially) reused

    @staticmethod
    def _notify(rq: Request):
        # The websocket condition lives on the uvicorn loop, on a different thread
        # than the scheduler. If notification fails, timeout-based draining still
        # makes progress.
        if rq._change_cv is None or rq._loop is None:
            return
        try:
            rq._loop.call_soon_threadsafe(lambda: asyncio.create_task(_notify_cv(rq._change_cv)))
        except Exception:
            pass


async def _notify_cv(cv: asyncio.Condition):
    async with cv:
        cv.notify_all()
