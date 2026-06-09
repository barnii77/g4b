import g4b
import asyncio
from queue import SimpleQueue


class Request:
    def __init__(self, input_tokens: list[int], change_cv: asyncio.Condition | None = None):
        self.input_tokens = input_tokens
        self._output_tokens: list[int] = []
        self._prev_retrieve_last_token_idx = 0
        self._change_cv = asyncio.Condition() if change_cv else change_cv

    def get_new_tokens(self) -> list[int]:
        out = self._output_tokens[self._prev_retrieve_last_token_idx :]
        self._prev_retrieve_last_token_idx = len(self._output_tokens)
        return out


class Scheduler:
    def __init__(self, model: "g4b.models.Model"):
        self._model = model
        self._queue: SimpleQueue[Request] = SimpleQueue()

    def step(self):
        # TODO chunked prefill
        # TODO decode
        # TODO notify _change_cv
        ...

    def submit(self, request: Request):
        self._queue.put(request)

    def abort(self, request: Request):
        raise RuntimeError("not implemented")  # TODO
