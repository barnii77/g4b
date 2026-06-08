import g4b
import asyncio
from queue import SimpleQueue


class Request:
    def __init__(self, input_tokens: list[int]):
        self.input_tokens = input_tokens
        self._output_tokens: list[int] = []
        self._change_cv = asyncio.Condition()


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
