"""
API SPEC:
- /api/v1/chat:
  - opens a text websocket. websocket json events below.
  - client -> server:
    - {"type": "user.init", "chat": [<event>, <event>, ...]}
    - {"type": "user.prompt", "content": "my prompt", "obfuscation": <obfuscation>}
    - {"type": "user.abort", "obfuscation": <obfuscation>}
    - {"type": "tool.output", "output": "... (may be serialized json) ...", "obfuscation": <obfuscation>}
    - {"type": "ping"}
  - server -> client:
    - {"type": "tool.call", "call": {"name": "mytool", args: {"arg1": <anything>, "arg2": <anything>}}, "obfuscation": <obfuscation>}
    - {"type": "assistant.response.chunk", content: "...", "obfuscation": <obfuscation>}
    - {"type": "assistant.response.done", "obfuscation": <obfuscation>}
    - {"type": "pong"}
    - {"type": "error", "code": <uint32>, "detail": "..."}
- <obfuscation>: An optional but recommended crypto-random base64-encoded sequence of X bytes where X is the absolute value of a sample from a uniform distribution U[L/2, 3L/2] and L is the length of the payload without obfuscation. The purpose of this is to prevent certain side channel attacks. Google "openai api obfuscation strings" for more information.
- <event>: any context-management-related json event documented above except user.init and user.abort
"""

import uvicorn
import threading
import asyncio
import time
import secrets
import random
import json
import base64
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from g4b import scheduler as scheduler_mod
from g4b import tokenizer as tokenizer_mod

app = FastAPI()
_scheduler: "scheduler_mod.Scheduler"
_tokenizer: "tokenizer_mod.Tokenizer"
_chat_template: "tokenizer_mod.ChatTemplate"
RESPONSE_CHUNK_TIMEOUT = 5
RESPONSE_CHUNK_BUFFER_SIZE = 48  # tokens

GENERIC_ERROR = 1
MAX_CONTEXT_WINDOW_EXCEEDED_ERROR = 2  # TODO emit this when it happens so the client can truncate conversation


def register_scheduler(sched: "scheduler_mod.Scheduler"):
    global _scheduler
    _scheduler = sched


def register_tokenizer(tok: "tokenizer_mod.Tokenizer"):
    global _tokenizer
    _tokenizer = tok


def register_chat_template(ct: "tokenizer_mod.ChatTemplate"):
    global _chat_template
    _chat_template = ct


type _ModelOutput = tokenizer_mod.ResponseFragment | tokenizer_mod.ToolCall
type _ModelInput = tokenizer_mod.PromptFragment | tokenizer_mod.ToolOutput


class _UserChatManagerEvent:
    def __init__(self, frags: list[_ModelOutput], is_terminal: bool):
        self.frags = frags
        self.is_terminal = is_terminal


class _UserChatManager:
    def __init__(self):
        self.chat_fragments: list["tokenizer_mod.ChatFragment"] = []
        self.active_request: "scheduler_mod.Request | None" = None
        self.change_cv = asyncio.Condition()
        self.token_buf: list[int] = []

    def _drain_token_buffer_as_far_as_possible(self) -> list[_ModelOutput]:
        return []  # TODO if syntactically incomplete tool call, do not drain that partial call yet

    async def continue_processing(self):
        while True:
            done, _ = await asyncio.wait(
                [self.change_cv.wait()], timeout=RESPONSE_CHUNK_TIMEOUT, return_when=asyncio.FIRST_COMPLETED
            )
            if self.active_request and done:
                break
            else:
                # timeout: must drain at least every RESPONSE_CHUNK_TIMEOUT seconds to ensure smooth UX
                assert _tokenizer.eos not in self.token_buf
                return _UserChatManagerEvent(self._drain_token_buffer_as_far_as_possible(), False)

        new_toks = self.active_request.get_new_tokens()
        self.token_buf.extend(new_toks)

        if (is_terminal := _tokenizer.eos in new_toks) or len(self.token_buf) > RESPONSE_CHUNK_BUFFER_SIZE:
            return _UserChatManagerEvent(self._drain_token_buffer_as_far_as_possible(), is_terminal)
        return _UserChatManagerEvent([], False)

    def submit(self, frag: _ModelInput):
        self.chat_fragments.append(frag)
        if self.active_request:
            return
        inp = _chat_template.apply(self.chat_fragments)
        toks = _tokenizer.tokenize(inp)
        rq = scheduler_mod.Request(toks, self.change_cv)
        _scheduler.submit(rq)
        self.active_request = rq

    def abort(self):
        if self.active_request:
            _scheduler.abort(self.active_request)

    def init(self, chat_fragments: list["tokenizer_mod.ChatFragment"]):
        self.chat_fragments = chat_fragments


@app.websocket("/api/v1/chat")
async def chat(ws: WebSocket):
    await ws.accept()

    is_init = False
    state = _UserChatManager()
    try:
        continue_processing_coro = state.continue_processing()
        recv_json_coro = ws.receive_json()

        while True:
            done, pending = await asyncio.wait(
                [recv_json_coro, continue_processing_coro], return_when=asyncio.FIRST_COMPLETED
            )

            if recv_json_coro in done:
                event = recv_json_coro.result()
                recv_json_coro = ws.receive_json()
            else:
                event = None
            if continue_processing_coro in done:
                advancements: _UserChatManagerEvent = continue_processing_coro.result()
                continue_processing_coro = state.continue_processing()
            else:
                advancements: _UserChatManagerEvent = _UserChatManagerEvent([], False)

            # Handle server -> client event(s)
            if advancements.frags:
                for frag in advancements.frags:
                    if isinstance(frag, tokenizer_mod.ResponseFragment):
                        await ws.send_json(
                            _add_obfuscation(
                                {
                                    "type": "assistant.response.chunk",
                                    "content": frag.content,
                                }
                            )
                        )
                    else:
                        assert isinstance(frag, tokenizer_mod.ToolCall)
                        await ws.send_json(
                            _add_obfuscation(
                                {
                                    "type": "tool.call",
                                    "call": frag.call,
                                }
                            )
                        )
            if advancements.is_terminal:
                await ws.send_json(_add_obfuscation({"type": "assistant.response.done"}))

            # Handle client -> server event
            if event:
                ty = await _assert_string(event.get("type"), "`type` must be string", ws)
                match ty:
                    case "user.init":
                        if is_init:
                            await _chat_err(ws, "already initialized")

                        is_init = True
                        prev_events = event.get("chat")
                        if not isinstance(prev_events, list):
                            await _chat_err(ws, "`chat` must be list")

                        parsed_events: list["tokenizer_mod.ChatFragment"] = []
                        for ev in prev_events:
                            if not isinstance(ev, dict):
                                await _chat_err(ws, "`event` must be json object")

                            ty = ev.get("type")
                            if not isinstance(ty, str):
                                await _chat_err(ws, "`type` must be string")

                            parsed_ev: "tokenizer_mod.ChatFragment"
                            match ty:
                                case "user.prompt":
                                    content = await _assert_string(ev.get("content"), "`content` must be string", ws)
                                    parsed_ev = tokenizer_mod.PromptFragment(content)
                                case "tool.output":
                                    output = await _assert_string(ev.get("output"), "`output` must be string", ws)
                                    parsed_ev = tokenizer_mod.ToolOutput(output)
                                case "tool.call":
                                    call = await _assert_dict(ev.get("call"), "`call` must be json object", ws)
                                    parsed_ev = tokenizer_mod.ToolCall(call)  # TODO validate according to some schema?
                                case "assistant.response.chunk":
                                    content = await _assert_string(ev.get("content"), "`content` must be string", ws)
                                    parsed_ev = tokenizer_mod.ResponseFragment(content)
                                case _:
                                    await _chat_err(ws, f"event type `{ty}` disallowed in `events` for init")
                            parsed_events.append(parsed_ev)

                        state.init(parsed_events)

                    case "user.prompt":
                        if not is_init:
                            await _chat_err(ws, "not initialized")
                        content = await _assert_string(event.get("content"), "`content` must be string", ws)
                        parsed = tokenizer_mod.PromptFragment(content)
                        state.submit(parsed)

                    case "tool.output":
                        if not is_init:
                            await _chat_err(ws, "not initialized")
                        output = await _assert_string(event.get("output"), "`output` must be string", ws)
                        parsed = tokenizer_mod.ToolOutput(output)
                        state.submit(parsed)

                    case "user.abort":
                        if not is_init:
                            await _chat_err(ws, "not initialized")
                        state.abort()

                    case "ping":
                        await ws.send_json({"type": "pong"})

                    case _:
                        await _chat_err(ws, "unrecognized event type")

    except WebSocketDisconnect:
        pass


class Uvicorn:
    def __init__(self, thread: threading.Thread, server: uvicorn.Server):
        self.thread = thread
        self.server = server

    @classmethod
    def start(cls, host: str, port: int) -> Uvicorn:
        config = uvicorn.Config(app, host, port, log_level="info")
        server = uvicorn.Server(config)
        thread = threading.Thread(target=lambda: asyncio.run(server.serve()), name="uvicorn", daemon=True)
        thread.start()
        while not server.started:
            if not thread.is_alive():
                raise RuntimeError("the uvicorn died a painful death")
            time.sleep(0.01)
        return Uvicorn(thread, server)

    def stop(self):
        self.server.should_exit = True
        self.thread.join()


async def _chat_err(ws: WebSocket, error: str, code: int = GENERIC_ERROR):
    await ws.send_json({"type": "error", "detail": error})
    await ws.close(code=4000, reason=error)
    raise RuntimeError(error)


async def _assert_string(x, err: str, ws) -> str:
    if not isinstance(x, str):
        await _chat_err(ws, err)
    return x


async def _assert_dict(x, err: str, ws) -> dict:
    if not isinstance(x, dict):
        await _chat_err(ws, err)
    return x


def _add_obfuscation(resp: dict) -> dict:
    l = len(json.dumps(resp))
    n = round(random.uniform(l // 2, 3 * l // 2))
    obfuscation = base64.b64encode(secrets.token_bytes(n))
    resp["obfuscation"] = obfuscation
    return resp
