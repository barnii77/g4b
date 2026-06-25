from __future__ import annotations

import asyncio
import contextlib
import json
import os
import secrets
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Literal

import uvicorn
from fastapi import FastAPI, Request as FastAPIRequest
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from g4b import lifecycle
from g4b import protocol
from g4b import scheduler as scheduler_mod
from g4b import tokenizer as tokenizer_mod
from g4b.config import Config
from g4b.utils import create_file_logger

_log_all_prompts_path = os.environ.get("G4B_LOG_ALL_PROMPTS_PATH")
_prompts_logger = create_file_logger(_log_all_prompts_path) if _log_all_prompts_path else None

app = FastAPI()

_CHAT_APP_HTML_PATH = Path(__file__).parent / "chat.html"

_scheduler: scheduler_mod.Scheduler
_tokenizer: tokenizer_mod.Tokenizer
_chat_template: tokenizer_mod.ChatTemplate
_config: Config

RESPONSE_CHUNK_TIMEOUT = 5
RESPONSE_CHUNK_BUFFER_SIZE = 48

MAX_CONTEXT_WINDOW_EXCEEDED_ERROR = "context_length_exceeded"

_NAME_NONE = 0
_NAME_TURN = 1
_NAME_CHANNEL = 2


def register_scheduler(sched: scheduler_mod.Scheduler):
    global _scheduler
    _scheduler = sched


def register_tokenizer(tok: tokenizer_mod.Tokenizer):
    global _tokenizer
    _tokenizer = tok


def register_chat_template(ct: tokenizer_mod.ChatTemplate):
    global _chat_template
    _chat_template = ct


def register_config(config: Config):
    global _config
    _config = config


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(_CHAT_APP_HTML_PATH.read_text(encoding="utf-8"))


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_request: FastAPIRequest, exc: RequestValidationError) -> JSONResponse:
    first = exc.errors()[0] if exc.errors() else {}
    loc = first.get("loc") or []
    param = ".".join(str(x) for x in loc[1:]) or None
    message = first.get("msg", "invalid request")
    return _openai_error(message, param=param, status_code=400)


# TODO with this API, because we retokenize model outputs every prompt, we nuke the last response's prefix caching
#  in case the response retokenizes differently than the model produced it (common). This is a regression from the
#  previous websocket-based API which did not suffer from this, and it increases TTFT by O(seconds).
@app.post("/v1/chat/completions")
async def chat_completions(req: protocol.ChatCompletionRequest):
    session_or_error = _GenerationSession.from_request(req)
    if isinstance(session_or_error, JSONResponse):
        return session_or_error

    session = session_or_error
    if req.stream:
        return StreamingResponse(_stream_chat_completion(req, session), media_type="text/event-stream")

    try:
        message, finish_reason = await session.run_to_completion()
    finally:
        session.abort()

    completion = protocol.ChatCompletion(
        id=session.response_id,
        created=session.created,
        model=req.model,
        choices=[
            protocol.ChatCompletionChoice(
                message=message,
                finish_reason=finish_reason,
            )
        ],
    )
    return completion.model_dump(exclude_none=True)


@dataclass
class _GeneratedDelta:
    content: str | None = None
    reasoning: str | None = None
    tool_call: protocol.ToolCall | None = None


@dataclass
class _GenerationEvent:
    deltas: list[_GeneratedDelta]
    terminal: bool
    finish_reason: Literal["stop", "length", "tool_calls"] | None = None


# TODO should I really handle parsing tool calls and channels in the same file as the HTTP server?
class _GenerationSession:
    def __init__(self, prompt_tokens: list[int], max_context_len: int | None):
        self.response_id = f"chatcmpl-{secrets.token_hex(12)}"
        self.created = int(time.time())
        self.loop = asyncio.get_running_loop()
        self.change_cv = asyncio.Condition()
        self.request = scheduler_mod.Request(
            prompt_tokens,
            self.change_cv,
            self.loop,
            max_context_len=max_context_len,
        )
        self.token_buf: list[int] = []
        self._terminal_pending = False
        self._context_window_exceeded = False
        self._done = False

        self._held_toks: list[int] = []
        self._turn_is_model = True
        self._channel: str | None = None
        self._name_mode = _NAME_NONE
        self._name_chars: list[str] = []
        self._in_tool_call = False
        self._tool_call_tokens: list[int] = []
        self._tool_calls: list[protocol.ToolCall] = []
        self._content_parts: list[str] = []
        self._reasoning_parts: list[str] = []

        _scheduler.submit(self.request)

    @classmethod
    def from_request(cls, req: protocol.ChatCompletionRequest) -> "_GenerationSession | JSONResponse":
        tools = [] if req.tool_choice == "none" else req.tools
        prompt_tokens = _chat_template.apply(req.messages, tools)
        if len(prompt_tokens) > _config.context_len:
            return _context_length_error("prompt exceeds max context length; truncate the conversation")

        if _prompts_logger and lifecycle.is_deployment():
            _prompts_logger.info(_tokenizer.detokenize(prompt_tokens))

        max_context_len = None if _config.allow_sliding_global_context else _config.context_len
        return cls(prompt_tokens, max_context_len=max_context_len)

    def abort(self):
        if not self._done:
            _scheduler.abort(self.request)
            self._done = True

    async def run_to_completion(self) -> tuple[protocol.ChatMessage, Literal["stop", "length", "tool_calls"]]:
        finish_reason: Literal["stop", "length", "tool_calls"] | None = None
        while finish_reason is None:
            event = await self.next_event()
            finish_reason = event.finish_reason

        message = protocol.ChatMessage(
            role="assistant",
            content="".join(self._content_parts) if not self._tool_calls else ("".join(self._content_parts) or None),
            reasoning="".join(self._reasoning_parts) or None,
            tool_calls=self._tool_calls or None,
        )
        return message, finish_reason

    async def next_event(self) -> _GenerationEvent:
        self._poll_request()

        if self._terminal_pending:
            return self._make_terminal_event()

        if len(self.token_buf) > RESPONSE_CHUNK_BUFFER_SIZE:
            return _GenerationEvent(self._drain_token_buffer_as_far_as_possible(), False)

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
        self._poll_request()

        if self._terminal_pending:
            return self._make_terminal_event()

        if not signaled or len(self.token_buf) > RESPONSE_CHUNK_BUFFER_SIZE:
            return _GenerationEvent(self._drain_token_buffer_as_far_as_possible(), False)

        return _GenerationEvent([], False)

    async def _wait_for_change(self):
        async with self.change_cv:
            await self.change_cv.wait()

    def _poll_request(self):
        if self._done:
            return

        new_toks = self.request.get_new_tokens()
        if new_toks:
            self.token_buf.extend(new_toks)

        if self.request.done:
            self._terminal_pending = True
            self._context_window_exceeded = self.request._context_window_exceeded

    def _make_terminal_event(self) -> _GenerationEvent:
        deltas = self._drain_token_buffer_as_far_as_possible(final=True)
        self._terminal_pending = False
        self._done = True
        if self._context_window_exceeded:
            finish_reason: Literal["stop", "length", "tool_calls"] = "length"
        elif self._tool_calls:
            finish_reason = "tool_calls"
        else:
            finish_reason = "stop"
        return _GenerationEvent(deltas, True, finish_reason)

    def _finish_name(self):
        name = "".join(self._name_chars).strip()
        if self._name_mode == _NAME_TURN:
            self._turn_is_model = name == "model"
            self._channel = None
        else:
            self._channel = name or None
        self._name_mode = _NAME_NONE
        self._name_chars = []

    def _drain_token_buffer_as_far_as_possible(self, final: bool = False) -> list[_GeneratedDelta]:
        raw = self.token_buf
        self.token_buf = []

        gen_ending_toks = _tokenizer.gen_ending_tokens()
        cut = min((raw.index(t) for t in gen_ending_toks if t in raw), default=len(raw))
        toks = self._held_toks + raw[:cut]
        self._held_toks = []

        if not toks:
            return []

        out: list[_GeneratedDelta] = []
        run: list[int] = []

        def flush(force: bool):
            nonlocal run
            if run and self._turn_is_model and not self._in_tool_call:
                text, held = _tokenizer.detokenize_streaming(run, flush=force)
                if held:
                    self._held_toks = held
                if text:
                    out.append(self._record_text_delta(text))
            run = []

        for t in toks:
            if self._in_tool_call:
                if t == _tokenizer.end_of_tool_call:
                    out.append(self._parse_tool_call_tokens(self._tool_call_tokens))
                    self._tool_call_tokens = []
                    self._in_tool_call = False
                else:
                    self._tool_call_tokens.append(t)
                continue

            if self._name_mode != _NAME_NONE:
                piece = _tokenizer.detokenize([t])
                if "\n" in piece:
                    before, *_ = piece.partition("\n")
                    self._name_chars.append(before)
                    self._finish_name()
                else:
                    self._name_chars.append(piece)
                continue

            if t == _tokenizer.start_of_turn:
                flush(force=True)
                self._name_mode = _NAME_TURN
                self._name_chars = []
            elif t == _tokenizer.end_of_turn:
                flush(force=True)
                self._turn_is_model = False
                self._channel = None
            elif t == _tokenizer.start_of_channel:
                flush(force=True)
                self._name_mode = _NAME_CHANNEL
                self._name_chars = []
            elif t == _tokenizer.end_of_channel:
                flush(force=True)
                self._channel = None
            elif t == _tokenizer.start_of_tool_call:
                flush(force=True)
                self._in_tool_call = True
                self._tool_call_tokens = []
            elif self._turn_is_model:
                run.append(t)

        if final and self._in_tool_call and self._tool_call_tokens:
            # Malformed/incomplete tool-call syntax: surface it as content rather than crashing the request.
            run.extend(_tokenizer.tokenize("<|tool_call>"))
            run.extend(self._tool_call_tokens)
            self._in_tool_call = False
            self._tool_call_tokens = []
        flush(force=final)
        return out

    def _record_text_delta(self, text: str) -> _GeneratedDelta:
        if self._channel == "thought":
            self._reasoning_parts.append(text)
            return _GeneratedDelta(reasoning=text)
        self._content_parts.append(text)
        return _GeneratedDelta(content=text)

    def _parse_tool_call_tokens(self, tokens: list[int]) -> _GeneratedDelta:
        raw = _tokenizer.detokenize(tokens).strip()
        if not raw.startswith("call:"):
            self._content_parts.append(raw)
            return _GeneratedDelta(content=raw)
        body = raw[len("call:") :]
        name, sep, arguments = body.partition("{")
        if not sep or not arguments.endswith("}"):
            self._content_parts.append(raw)
            return _GeneratedDelta(content=raw)
        tool_call = protocol.ToolCall(
            id=f"call_{secrets.token_hex(8)}",
            function=protocol.FunctionCall(
                name=name.strip(),
                arguments=arguments[:-1],
            ),
        )
        self._tool_calls.append(tool_call)
        return _GeneratedDelta(tool_call=tool_call)


async def _stream_chat_completion(
    req: protocol.ChatCompletionRequest, session: _GenerationSession
) -> AsyncIterator[str]:
    try:
        yield _sse_chunk(
            protocol.ChatCompletionChunk(
                id=session.response_id,
                created=session.created,
                model=req.model,
                choices=[protocol.ChatCompletionChunkChoice(delta=protocol.ChatMessageDelta(role="assistant"))],
            )
        )

        finish_reason: Literal["stop", "length", "tool_calls"] | None = None
        while finish_reason is None:
            event = await session.next_event()
            for delta in event.deltas:
                if delta.content is not None:
                    yield _sse_delta(req, session, protocol.ChatMessageDelta(content=delta.content))
                if delta.reasoning is not None:
                    yield _sse_delta(req, session, protocol.ChatMessageDelta(reasoning=delta.reasoning))
                if delta.tool_call is not None:
                    yield _sse_delta(req, session, _tool_call_delta(delta.tool_call, len(session._tool_calls) - 1))
            finish_reason = event.finish_reason

        yield _sse_chunk(
            protocol.ChatCompletionChunk(
                id=session.response_id,
                created=session.created,
                model=req.model,
                choices=[
                    protocol.ChatCompletionChunkChoice(
                        delta=protocol.ChatMessageDelta(),
                        finish_reason=finish_reason,
                    )
                ],
            )
        )
        if req.stream_options and req.stream_options.include_usage:
            yield _sse_chunk(
                protocol.ChatCompletionChunk(
                    id=session.response_id,
                    created=session.created,
                    model=req.model,
                    choices=[],
                    usage=protocol.Usage(),
                )
            )
        yield "data: [DONE]\n\n"
    finally:
        session.abort()


def _sse_delta(
    req: protocol.ChatCompletionRequest, session: _GenerationSession, delta: protocol.ChatMessageDelta
) -> str:
    return _sse_chunk(
        protocol.ChatCompletionChunk(
            id=session.response_id,
            created=session.created,
            model=req.model,
            choices=[protocol.ChatCompletionChunkChoice(delta=delta)],
        )
    )


def _tool_call_delta(tool_call: protocol.ToolCall, index: int) -> protocol.ChatMessageDelta:
    return protocol.ChatMessageDelta(
        tool_calls=[
            protocol.ToolCallDelta(
                index=index,
                id=tool_call.id,
                type="function",
                function=protocol.FunctionCallDelta(
                    name=tool_call.function.name,
                    arguments=_function_arguments_as_string(tool_call.function.arguments),
                ),
            )
        ]
    )


def _sse_chunk(chunk: protocol.ChatCompletionChunk) -> str:
    return f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"


def _function_arguments_as_string(arguments: str | dict[str, object]) -> str:
    if isinstance(arguments, str):
        return arguments
    return json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))


def _context_length_error(message: str) -> JSONResponse:
    return _openai_error(
        message,
        code=MAX_CONTEXT_WINDOW_EXCEEDED_ERROR,
        status_code=400,
    )


def _openai_error(
    message: str,
    *,
    param: str | None = None,
    code: str | int | None = None,
    status_code: int = 400,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=protocol.ErrorResponse(
            error=protocol.Error(
                message=message,
                param=param,
                code=code,
            )
        ).model_dump(exclude_none=True),
    )


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
