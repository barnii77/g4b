import asyncio
from queue import SimpleQueue
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import g4b.models


class Request:
    def __init__(
        self,
        input_tokens: list[int],
        change_cv: asyncio.Condition | None = None,
        initial_context_len: int = 0,
    ):
        self.input_tokens = input_tokens
        self._output_tokens: list[int] = []
        self._prev_retrieve_last_token_idx = 0
        self._change_cv = change_cv
        self._prefill_pos = 0
        self._context_len = initial_context_len
        self._last_sample_t_idx = 0
        self._done = False

    def get_new_tokens(self) -> list[int]:
        out = self._output_tokens[self._prev_retrieve_last_token_idx :]
        self._prev_retrieve_last_token_idx = len(self._output_tokens)
        return out


class Scheduler:
    def __init__(self, model: "g4b.models.Model"):
        self._model = model
        self._queue: SimpleQueue[Request] = SimpleQueue()
        self._active: list[Request | None] = [None] * model.max_batch_size()
        self._phase_active: list[bool] = [False] * model.max_batch_size()
        self._last_phase: str | None = None

    def step(self):
        self._fill_free_slots()
        if not any(rq is not None and not rq._done for rq in self._active):
            return

        phase = "decode" if any(self._needs_decode(rq) for rq in self._active if rq is not None) else "prefill"
        if phase == "prefill":
            self._prepare_prefill_inputs()
            self._model.prefill_chunk(self)
        else:
            self._prepare_decode_inputs()
            self._model.decode(self)
        self._last_phase = phase
        self._collect_outputs(phase)
        self._drop_done_slots()

    def submit(self, request: Request):
        self._queue.put(request)

    def reset(self):
        self._queue = SimpleQueue()
        self._active = [None] * self._model.max_batch_size()
        self._phase_active = [False] * self._model.max_batch_size()

    def abort(self, request: Request):
        request._done = True
        for i, active in enumerate(self._active):
            if active is request:
                self._active[i] = None

    def _fill_free_slots(self):
        for i, rq in enumerate(self._active):
            if rq is not None and not rq._done:
                continue
            if self._queue.empty():
                continue
            self._active[i] = self._queue.get()

    def _needs_prefill(self, rq: Request) -> bool:
        return rq._prefill_pos < len(rq.input_tokens)

    def _needs_decode(self, rq: Request) -> bool:
        return not rq._done and not self._needs_prefill(rq)

    def _prepare_prefill_inputs(self):
        t = self._model.max_prefill_chunk_size()
        token_cols: list[list[int]] = []
        cache_offsets: list[int] = []
        time_sizes_after: list[int] = []
        phase_active: list[bool] = []
        for rq in self._active:
            if rq is None or rq._done or not self._needs_prefill(rq):
                token_cols.append([0] * t)
                cache_offsets.append(0)
                time_sizes_after.append(0)
                phase_active.append(False)
                continue

            phase_active.append(True)
            start = rq._prefill_pos
            end = min(start + t, len(rq.input_tokens))
            real_n = max(1, end - start)
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
            # The model always processes a full t-wide chunk of query positions and writes t KV slots
            # starting at start_offset, so the attention window must span all t of them (start_offset + t),
            # NOT just the real_n valid tokens. FA derives q_t_base = window_size - t; using real_n here
            # would make q_t_base negative for a short/padded prompt, so every query attends to zero keys
            # and produces 0/0 = NaN. _context_len still advances by real_n for the next step's offset.
            time_sizes_after.append(start_offset + t)

        self._phase_active = phase_active
        self._model.prepare_prefill_inputs(token_cols, cache_offsets, time_sizes_after)

    def _prepare_decode_inputs(self):
        t = self._model.max_prefill_chunk_size()
        token_cols: list[list[int]] = []
        cache_offsets: list[int] = []
        time_sizes_after: list[int] = []
        phase_active: list[bool] = []
        for rq in self._active:
            if rq is None or rq._done or self._needs_prefill(rq):
                token_cols.append([0] * t)
                cache_offsets.append(0)
                time_sizes_after.append(0)
                phase_active.append(False)
                continue
            phase_active.append(True)
            tok = rq.input_tokens[-1] if not rq._output_tokens else rq._output_tokens[-1]
            token_cols.append([tok] * t)
            cache_offsets.append(rq._context_len)
            rq._context_len += 1
            rq._last_sample_t_idx = 0
            time_sizes_after.append(rq._context_len)

        self._phase_active = phase_active
        self._model.prepare_decode_inputs(token_cols, cache_offsets, time_sizes_after)

    def _collect_outputs(self, phase: str):
        t = self._model.max_prefill_chunk_size()
        vals = self._model.collect_output_token_ids()
        for b, rq in enumerate(self._active):
            if rq is None or rq._done or not self._phase_active[b]:
                continue
            if phase == "prefill" and self._needs_prefill(rq):
                continue
            t_idx = rq._last_sample_t_idx
            tok = vals[b * t + t_idx]
            rq._output_tokens.append(tok)
            if tok == self._model.stop_token_id():
                rq._done = True
            self._notify(rq)

    def _drop_done_slots(self):
        for i, rq in enumerate(self._active):
            if rq is not None and rq._done:
                self._active[i] = None

    @staticmethod
    def _notify(rq: Request):
        # The websocket condition lives on the uvicorn loop. If notification fails,
        # timeout-based draining still makes progress.
        if rq._change_cv is None:
            return
        try:
            loop = rq._change_cv._loop
            if loop is not None:
                loop.call_soon_threadsafe(lambda: asyncio.create_task(_notify_cv(rq._change_cv)))
        except Exception:
            pass


async def _notify_cv(cv: asyncio.Condition):
    async with cv:
        cv.notify_all()
