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
    - {"type": "tool.call", "call": {"name": "mytool", "args": {"arg1": <anything>, "arg2": <anything>}}, "obfuscation": <obfuscation>}
    - {"type": "assistant.response.chunk", "content": "...", "obfuscation": <obfuscation>}
    - {"type": "assistant.response.done", "obfuscation": <obfuscation>}
    - {"type": "pong"}
    - {"type": "error", "code": <uint32>, "detail": "..."}
- <obfuscation>: An optional but recommended crypto-random base64-encoded sequence of X bytes
  where X is the absolute value of a sample from a uniform distribution U[L/2, 3L/2]
  and L is the length of the payload without obfuscation.
- <event>: any context-management-related json event documented above except user.init and user.abort
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import random
import secrets
import threading
import time
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from g4b import scheduler as scheduler_mod
from g4b import tokenizer as tokenizer_mod


app = FastAPI()

_scheduler: "scheduler_mod.Scheduler"
_tokenizer: "tokenizer_mod.Tokenizer"
_chat_template: "tokenizer_mod.ChatTemplate"
_max_ctx_len: int

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


def register_max_ctx_len(clen: int):
    global _max_ctx_len
    _max_ctx_len = clen


type _ModelOutput = tokenizer_mod.ResponseFragment | tokenizer_mod.ToolCall
type _ModelInput = tokenizer_mod.PromptFragment | tokenizer_mod.ToolOutput


class _ChatClosed(Exception):
    pass


class _UserChatManagerEvent:
    def __init__(self, frags: list[_ModelOutput], is_terminal: bool):
        self.frags = frags
        self.is_terminal = is_terminal


async def _notify_condition(cv: asyncio.Condition):
    async with cv:
        cv.notify_all()


class _UserChatManager:
    def __init__(self, max_context_len: int):
        self.max_context_len = max_context_len
        self.chat_fragments: list["tokenizer_mod.ChatFragment"] = []
        self.active_request: "scheduler_mod.Request | None" = None

        self.loop = asyncio.get_running_loop()
        self.change_cv = asyncio.Condition()

        # Compatibility with current scheduler._notify(), which reads cv._loop.
        # The manager itself does not rely on this private field.
        with contextlib.suppress(Exception):
            self.change_cv._loop = self.loop  # type: ignore[attr-defined]

        self.token_buf: list[int] = []

        # Streaming chunks are sent to the client as they arrive, but committed
        # to chat history as one assistant turn when terminal.
        self._current_response_parts: list[str] = []
        self._terminal_pending = False

    def init(self, chat_fragments: list["tokenizer_mod.ChatFragment"]) -> int:
        """
        Initialize history. Returns MAX_CONTEXT_WINDOW_EXCEEDED_ERROR if the
        serialized history already exceeds max_context_len, otherwise 0.
        """
        self.chat_fragments = chat_fragments
        self.active_request = None
        self.token_buf.clear()
        self._current_response_parts.clear()
        self._terminal_pending = False
        if self._tokenize_history_len() > self.max_context_len:
            return MAX_CONTEXT_WINDOW_EXCEEDED_ERROR
        return 0

    def _tokenize_history_len(self) -> int:
        if not self.chat_fragments:
            return 0
        inp = _chat_template.apply(self.chat_fragments)
        return len(_tokenizer.tokenize(inp))

    def submit(self, frag: _ModelInput) -> tuple[bool, int | None]:
        """
        Submit a user/tool fragment.

        Returns (ok, error_code).  ok is False if a request is already active
        or if the resulting token sequence would exceed max_context_len.
        Crucially, on failure the fragment is not appended to chat history,
        avoiding silent history corruption.
        """
        self._poll_active_request()

        if self.active_request is not None or self._terminal_pending:
            return False, GENERIC_ERROR

        self.chat_fragments.append(frag)
        inp = _chat_template.apply(self.chat_fragments)
        toks = _tokenizer.tokenize(inp)

        if len(toks) > self.max_context_len:
            # Roll back the history append because we cannot honor this request.
            self.chat_fragments.pop()
            return False, MAX_CONTEXT_WINDOW_EXCEEDED_ERROR

        rq = scheduler_mod.Request(toks, self.change_cv)
        _scheduler.submit(rq)
        self.active_request = rq

        self.token_buf.clear()
        self._current_response_parts.clear()
        self._terminal_pending = False
        return True, None

    def abort(self) -> bool:
        """
        Abort current request.

        Returns True if anything active/partial existed. The websocket handler
        emits assistant.response.done immediately so the frontend can stop its
        spinner deterministically.
        """
        had_active = (
            self.active_request is not None
            or bool(self.token_buf)
            or bool(self._current_response_parts)
            or self._terminal_pending
        )

        if self.active_request is not None:
            _scheduler.abort(self.active_request)

        self.active_request = None
        self.token_buf.clear()
        self._current_response_parts.clear()
        self._terminal_pending = False

        self._notify_from_any_thread()
        return had_active

    def _notify_from_any_thread(self):
        try:
            self.loop.call_soon_threadsafe(lambda: asyncio.create_task(_notify_condition(self.change_cv)))
        except Exception:
            pass

    def _poll_active_request(self):
        """
        Pull tokens from the scheduler Request into the local token buffer.

        This is deliberately called on wakeup and on timeout, so correctness does
        not depend on every condition notification being delivered.
        """
        rq = self.active_request
        if rq is None:
            return

        new_toks = rq.get_new_tokens()
        if new_toks:
            self.token_buf.extend(new_toks)

        if rq._done or _tokenizer.eos in new_toks:
            self.active_request = None
            self._terminal_pending = True

    def _drain_token_buffer_as_far_as_possible(self) -> list[_ModelOutput]:
        if not self.token_buf:
            return []

        toks = self.token_buf
        self.token_buf = []

        if _tokenizer.eos in toks:
            toks = toks[: toks.index(_tokenizer.eos)]

        text = _tokenizer.detokenize(toks)  # TODO technically this may be incorrect unless I buffer a few tokens

        # <start_of_turn>/<end_of_turn> are chat-template structural tokens that
        # the tokenizer detokenizes back into literal text. They must not leak to
        # the client; they are added by the chat template at the next turn.
        text = text.replace("<start_of_turn>", "").replace("<end_of_turn>", "")

        if not text:
            return []

        self._current_response_parts.append(text)
        return [tokenizer_mod.ResponseFragment(text)]

    def _commit_current_response_to_history(self):
        if not self._current_response_parts:
            return

        full_text = "".join(self._current_response_parts)
        self._current_response_parts.clear()

        if full_text:
            self.chat_fragments.append(tokenizer_mod.ResponseFragment(full_text))

    def _make_terminal_event(self) -> _UserChatManagerEvent:
        frags = self._drain_token_buffer_as_far_as_possible()
        self._commit_current_response_to_history()
        self._terminal_pending = False
        return _UserChatManagerEvent(frags, True)

    async def _wait_for_change(self):
        async with self.change_cv:
            await self.change_cv.wait()

    async def continue_processing(self) -> _UserChatManagerEvent:
        """
        Wait for model progress or timeout and return websocket-visible fragments.

        Timeout still polls the Request, so a lost notify only causes latency,
        not a permanently stuck stream.
        """
        self._poll_active_request()

        if self._terminal_pending:
            return self._make_terminal_event()

        if len(self.token_buf) > RESPONSE_CHUNK_BUFFER_SIZE:
            return _UserChatManagerEvent(self._drain_token_buffer_as_far_as_possible(), False)

        wait_task = asyncio.create_task(self._wait_for_change())
        try:
            done, pending = await asyncio.wait(
                {wait_task},
                timeout=RESPONSE_CHUNK_TIMEOUT,
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in pending:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        except asyncio.CancelledError:
            wait_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await wait_task
            raise

        signaled = wait_task in done

        self._poll_active_request()

        if self._terminal_pending:
            return self._make_terminal_event()

        if not signaled:
            # Timeout: drain whatever is available so UX remains smooth.
            return _UserChatManagerEvent(self._drain_token_buffer_as_far_as_possible(), False)

        if len(self.token_buf) > RESPONSE_CHUNK_BUFFER_SIZE:
            return _UserChatManagerEvent(self._drain_token_buffer_as_far_as_possible(), False)

        return _UserChatManagerEvent([], False)


@app.websocket("/api/v1/chat")
async def chat(ws: WebSocket):
    await ws.accept()

    is_init = False
    state = _UserChatManager(_max_ctx_len)

    recv_task = asyncio.create_task(ws.receive_json())
    continue_task = asyncio.create_task(state.continue_processing())

    try:
        while True:
            done, _pending = await asyncio.wait(
                {recv_task, continue_task},
                return_when=asyncio.FIRST_COMPLETED,
            )

            event: dict[str, Any] | None = None
            advancements = _UserChatManagerEvent([], False)

            if recv_task in done:
                event = recv_task.result()
                recv_task = asyncio.create_task(ws.receive_json())

            if continue_task in done:
                advancements = continue_task.result()
                continue_task = asyncio.create_task(state.continue_processing())

            # Handle server -> client event(s)
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
            if event is not None:
                if not isinstance(event, dict):
                    await _chat_err(ws, "`event` must be json object")

                ty = await _assert_string(event.get("type"), "`type` must be string", ws)

                match ty:
                    case "ping":
                        await ws.send_json({"type": "pong"})

                    case "user.init":
                        if is_init:
                            await _chat_err(ws, "already initialized")

                        prev_events = event.get("chat")
                        if not isinstance(prev_events, list):
                            await _chat_err(ws, "`chat` must be list")

                        parsed_events: list["tokenizer_mod.ChatFragment"] = []
                        for ev in prev_events:
                            if not isinstance(ev, dict):
                                await _chat_err(ws, "`event` must be json object")

                            ev_ty = ev.get("type")
                            if not isinstance(ev_ty, str):
                                await _chat_err(ws, "`type` must be string")

                            parsed_ev: "tokenizer_mod.ChatFragment"
                            match ev_ty:
                                case "user.prompt":
                                    content = await _assert_string(ev.get("content"), "`content` must be string", ws)
                                    parsed_ev = tokenizer_mod.PromptFragment(content)

                                case "tool.output":
                                    output = await _assert_string(ev.get("output"), "`output` must be string", ws)
                                    parsed_ev = tokenizer_mod.ToolOutput(output)

                                case "tool.call":
                                    call = await _assert_dict(ev.get("call"), "`call` must be json object", ws)
                                    parsed_ev = tokenizer_mod.ToolCall(call)  # TODO validate schema

                                case "assistant.response.chunk":
                                    content = await _assert_string(ev.get("content"), "`content` must be string", ws)
                                    parsed_ev = tokenizer_mod.ResponseFragment(content)

                                case "assistant.response.done":
                                    # Terminal marker is harmless in persisted logs but carries no content.
                                    continue

                                case _:
                                    await _chat_err(ws, f"event type `{ev_ty}` disallowed in `events` for init")
                                    raise AssertionError("unreachable")

                            parsed_events.append(parsed_ev)

                        err_code = state.init(parsed_events)
                        if err_code != 0:
                            await _chat_err(
                                ws,
                                f"provided chat history exceeds max context length ({state.max_context_len} tokens)",
                                code=err_code,
                            )
                        is_init = True

                    case "user.prompt":
                        if not is_init:
                            await _chat_err(ws, "not initialized")

                        content = await _assert_string(event.get("content"), "`content` must be string", ws)
                        parsed = tokenizer_mod.PromptFragment(content)

                        ok, err_code = state.submit(parsed)
                        if not ok:
                            detail = (
                                "request already active; send user.abort before submitting a new prompt"
                                if err_code != MAX_CONTEXT_WINDOW_EXCEEDED_ERROR
                                else f"prompt exceeds max context length ({state.max_context_len} tokens); truncate the conversation"
                            )
                            await ws.send_json(
                                {
                                    "type": "error",
                                    "code": err_code,
                                    "detail": detail,
                                }
                            )

                    case "tool.output":
                        if not is_init:
                            await _chat_err(ws, "not initialized")

                        output = await _assert_string(event.get("output"), "`output` must be string", ws)
                        parsed = tokenizer_mod.ToolOutput(output)

                        ok, err_code = state.submit(parsed)
                        if not ok:
                            detail = (
                                "request already active; send user.abort before submitting tool output"
                                if err_code != MAX_CONTEXT_WINDOW_EXCEEDED_ERROR
                                else f"tool output exceeds max context length ({state.max_context_len} tokens); truncate the conversation"
                            )
                            await ws.send_json(
                                {
                                    "type": "error",
                                    "code": err_code,
                                    "detail": detail,
                                }
                            )

                    case "user.abort":
                        if not is_init:
                            await _chat_err(ws, "not initialized")
                        state.abort()
                        await ws.send_json(_add_obfuscation({"type": "assistant.response.done"}))

                    case _:
                        await _chat_err(ws, "unrecognized event type")

    except (WebSocketDisconnect, _ChatClosed):
        pass
    finally:
        with contextlib.suppress(Exception):
            state.abort()
        await _cancel_task(recv_task)
        await _cancel_task(continue_task)


async def _cancel_task(task: asyncio.Task):
    if task.done():
        with contextlib.suppress(Exception):
            task.result()
        return

    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


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
    with contextlib.suppress(Exception):
        await ws.send_json({"type": "error", "code": code, "detail": error})
    with contextlib.suppress(Exception):
        await ws.close(code=4000, reason=error[:120])
    raise _ChatClosed(error)


async def _assert_string(x, err: str, ws) -> str:
    if not isinstance(x, str):
        await _chat_err(ws, err)
    return x


async def _assert_dict(x, err: str, ws) -> dict:
    if not isinstance(x, dict):
        await _chat_err(ws, err)
    return x


def _add_obfuscation(resp: dict) -> dict:
    l = len(json.dumps(resp, ensure_ascii=False, separators=(",", ":")))
    n = round(random.uniform(l // 2, 3 * l // 2))
    obfuscation = base64.b64encode(secrets.token_bytes(n)).decode("ascii")
    resp["obfuscation"] = obfuscation
    return resp