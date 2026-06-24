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
    - {"type": "assistant.response.chunk", "channel": "thought" | null, "content": "...", "obfuscation": <obfuscation>}
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
import os
import random
import secrets
import threading
import time
import uvicorn
from pathlib import Path
from typing import Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from g4b import lifecycle
from g4b import scheduler as scheduler_mod
from g4b import tokenizer as tokenizer_mod
from g4b.config import Config
from g4b.utils import create_file_logger

_log_all_prompts_path = os.environ.get("G4B_LOG_ALL_PROMPTS_PATH")
_prompts_logger = create_file_logger(_log_all_prompts_path) if _log_all_prompts_path else None

app = FastAPI()


_CHAT_APP_HTML_PATH = Path(__file__).parent / "chat.html"


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(_CHAT_APP_HTML_PATH.read_text(encoding="utf-8"))


_scheduler: "scheduler_mod.Scheduler"
_tokenizer: "tokenizer_mod.Tokenizer"
_chat_template: "tokenizer_mod.ChatTemplate"
_config: Config

RESPONSE_CHUNK_TIMEOUT = 5
RESPONSE_CHUNK_BUFFER_SIZE = 48  # tokens

GENERIC_ERROR = 1
MAX_CONTEXT_WINDOW_EXCEEDED_ERROR = 2

# Sub-states for the streaming token parser when it is reading the *name* that
# follows a structural token, e.g. the "model" in `<|turn>model\n` or the
# "thought" in `<|channel>thought\n`. The name runs until the next newline.
_NAME_NONE = 0
_NAME_TURN = 1
_NAME_CHANNEL = 2


def register_scheduler(sched: "scheduler_mod.Scheduler"):
    global _scheduler
    _scheduler = sched


def register_tokenizer(tok: "tokenizer_mod.Tokenizer"):
    global _tokenizer
    _tokenizer = tok


def register_chat_template(ct: "tokenizer_mod.ChatTemplate"):
    global _chat_template
    _chat_template = ct


def register_config(config: Config):
    global _config
    _config = config


type _ModelOutput = tokenizer_mod.ResponseFragment | tokenizer_mod.ToolCall
type _ModelInput = tokenizer_mod.PromptFragment | tokenizer_mod.ToolOutput


class _ChatClosed(Exception):
    pass


class _UserChatManagerEvent:
    def __init__(self, frags: list[_ModelOutput], is_terminal: bool, error_code: int | None = None):
        self.frags = frags
        self.is_terminal = is_terminal
        self.error_code = error_code


async def _notify_condition(cv: asyncio.Condition):
    async with cv:
        cv.notify_all()


def _strip_channel(tokens: list[int], drop: str) -> list[int]:
    """Splice out every `<|channel>{drop}\n ... <channel|>` span (markers included)
    from verbatim tokens, leaving other channels untouched. An unterminated span is
    closed implicitly at the next channel/turn boundary, matching the display parser."""
    out: list[int] = []
    i, n = 0, len(tokens)
    while i < n:
        if tokens[i] != _tokenizer.start_of_channel:
            out.append(tokens[i])
            i += 1
            continue
        # Read the channel name: tokens after the marker up to the first newline.
        j, name = i + 1, ""
        while j < n and "\n" not in name:
            name += _tokenizer.detokenize([tokens[j]])
            j += 1
        if name.split("\n", 1)[0].strip() != drop:
            out.extend(tokens[i:j])  # keep this channel's marker + name verbatim
            i = j
            continue
        # Drop this channel's body up to (and including) its closing marker.
        stops = (
            _tokenizer.end_of_channel,
            _tokenizer.start_of_channel,
            _tokenizer.start_of_turn,
            _tokenizer.end_of_turn,
        )
        while j < n and tokens[j] not in stops:
            j += 1
        i = j + 1 if j < n and tokens[j] == _tokenizer.end_of_channel else j
    return out


class _UserChatManager:
    def __init__(self):
        # The running token prompt: BOS + system prefix + every turn so far. While
        # a request is in flight it ends with the `<|turn>model\n` opener the model
        # is completing; the model's generated tokens are committed onto it.
        self.history_tokens: list[int] = []
        self.active_request: "scheduler_mod.Request | None" = None

        self.loop = asyncio.get_running_loop()
        self.change_cv = asyncio.Condition()

        self.token_buf: list[int] = []
        # Verbatim raw tokens of the in-progress model turn, accumulated across
        # polls and committed to history (wrapped in turn markers) on termination.
        self._gen_tokens: list[int] = []

        self._terminal_pending = False
        self._terminal_error_code: int | None = None
        self._reset_stream_parser()

    def _reset_stream_parser(self):
        """
        State for the token-level streaming parser in
        _drain_token_buffer_as_far_as_possible, carried across its calls.

        Generation begins *inside* the model turn opened by the prompt's trailing
        `<|turn>model\n`, so we start already in a model turn, outside any channel.
        """
        self._held_toks: list[int] = []  # trailing byte-tokens of a split UTF-8 char
        self._turn_is_model = True  # False while inside a non-model turn (dropped)
        self._channel: str | None = None  # active channel within the model turn
        self._name_mode = _NAME_NONE  # reading a turn-role / channel-name run
        self._name_chars: list[str] = []

    def init(self, chat_fragments: list["tokenizer_mod.ChatFragment"]) -> int:
        """
        Initialize history from prior chat fragments (text/json from the client).

        These are tokenized once here via the chat template; only in-session
        model generations are stored verbatim. Returns
        MAX_CONTEXT_WINDOW_EXCEEDED_ERROR if the serialized history already
        exceeds max_context_len, otherwise 0.
        """
        self.history_tokens = _tokenizer.tokenize(_chat_template.apply(chat_fragments, include_open_turn_to_complete=False))
        self.active_request = None
        self.token_buf.clear()
        self._gen_tokens.clear()
        self._terminal_pending = False
        self._terminal_error_code = None
        self._reset_stream_parser()
        if len(self.history_tokens) > _config.context_len:
            return MAX_CONTEXT_WINDOW_EXCEEDED_ERROR
        return 0

    def submit(self, frag: _ModelInput) -> tuple[bool, int | None]:
        """
        Submit a user/tool fragment.

        Returns (ok, error_code).  ok is False if a request is already active
        or if the resulting token sequence would exceed max_context_len.
        Crucially, on failure the fragment is not appended to history, avoiding
        silent history corruption (e.g. model produced `t` then `p` but token `tp` exists).
        """
        self._poll_active_request()

        if self.active_request is not None or self._terminal_pending:
            return False, GENERIC_ERROR

        # The chat template synthesises the user/tool turn followed by the
        # `<|turn>model\n` generation prompt; both become part of the running
        # prompt. The model then generates its response and its own closing
        # `<turn|>` onto it (see _commit_model_turn).
        turn = _tokenizer.tokenize(_chat_template.apply([frag], include_conversation_init=False), add_bos=False)

        if len(self.history_tokens) + len(turn) > _config.context_len:
            # Do not mutate history: the turn is never appended.
            return False, MAX_CONTEXT_WINDOW_EXCEEDED_ERROR

        self.history_tokens.extend(turn)
        max_context_len = None if _config.allow_sliding_global_context else _config.context_len
        rq = scheduler_mod.Request(self.history_tokens, self.change_cv, self.loop, max_context_len=max_context_len)
        _scheduler.submit(rq)
        self.active_request = rq

        self.token_buf.clear()
        self._gen_tokens.clear()
        self._terminal_pending = False
        self._terminal_error_code = None
        self._reset_stream_parser()
        if _prompts_logger and lifecycle.is_deployment():
            _prompts_logger.info(_tokenizer.detokenize(self.history_tokens))
        return True, None

    def abort(self) -> bool:
        """
        Abort current request.

        Returns True if anything active/partial existed. The websocket handler
        emits assistant.response.done immediately so the frontend can stop its
        spinner deterministically.
        """
        self._poll_active_request()  # capture any last-moment generated tokens

        had_active = (
            self.active_request is not None
            or bool(self.token_buf)
            or bool(self._held_toks)
            or bool(self._gen_tokens)
            or self._terminal_pending
        )

        if self.active_request is not None:
            _scheduler.abort(self.active_request)

        self.active_request = None
        # Persist whatever the model produced so far so history matches what the
        # client was streamed before the abort.
        self._commit_model_turn()
        self.token_buf.clear()
        self._terminal_pending = False
        self._terminal_error_code = None
        self._reset_stream_parser()

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
            # Accumulate the verbatim model turn independently of the display
            # drain, which consumes token_buf and holds back partial UTF-8.
            self._gen_tokens.extend(new_toks)

        if rq.done:
            self.active_request = None
            self._terminal_pending = True
            if rq._context_window_exceeded:
                self._terminal_error_code = MAX_CONTEXT_WINDOW_EXCEEDED_ERROR

    def _finish_name(self):
        name = "".join(self._name_chars).strip()
        if self._name_mode == _NAME_TURN:
            # Only model turns are surfaced; user/tool/etc. turns are dropped.
            self._turn_is_model = name == "model"
            self._channel = None
        else:  # _NAME_CHANNEL
            self._channel = name or None
        self._name_mode = _NAME_NONE
        self._name_chars = []

    def _drain_token_buffer_as_far_as_possible(self, final: bool = False) -> list[_ModelOutput]:
        """
        Streaming stateful parser over newly generated tokens.

        Splits the token stream on the structural tokens `<|turn>`/`<turn|>` and
        `<|channel>`/`<channel|>`, keeping only model-turn content, and further
        splits model content into channel subsequences (None for the visible
        answer, e.g. "thought" for reasoning). Each contiguous channel run is
        detokenized and emitted as one ResponseFragment carrying its channel.

        State (current turn/channel, partial channel-name, and any held-back
        UTF-8 tail) persists across calls via the instance, so chunked drains
        reconstruct the same result as a single pass.

        This is purely the *display* path: emitted fragments are streamed to the
        client. History is maintained separately and verbatim from the raw
        generated tokens (see _commit_model_turn), so nothing here writes to it.
        """
        # TODO this parser only ever emits text ResponseFragments; it has no path to recognize a tool call in the
        #  model's output and emit a ToolCall, so tool calls the model generates are not surfaced as ToolCall events.
        #  Their tokens are still retained verbatim in history.
        raw = self.token_buf
        self.token_buf = []

        # Cut at the first generation-ending token; nothing after it is output.
        gen_ending_toks = _tokenizer.gen_ending_tokens()
        cut = min((raw.index(t) for t in gen_ending_toks if t in raw), default=len(raw))
        toks = self._held_toks + raw[:cut]
        self._held_toks = []

        if not toks:
            return []

        out: list[_ModelOutput] = []
        run: list[int] = []

        def flush(force: bool):
            nonlocal run
            if run and self._turn_is_model:
                text, held = _tokenizer.detokenize_streaming(run, flush=force)
                if held:
                    self._held_toks = held
                if text:
                    out.append(tokenizer_mod.ResponseFragment(text, self._channel))
            run = []

        for t in toks:
            if self._name_mode != _NAME_NONE:
                # Reading a turn-role / channel-name run, which ends at a newline.
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
                flush(force=True)  # implicit close of any open channel (no nesting)
                self._name_mode = _NAME_CHANNEL
                self._name_chars = []
            elif t == _tokenizer.end_of_channel:
                flush(force=True)
                self._channel = None
            elif self._turn_is_model:
                run.append(t)

        flush(force=final)
        return out

    def _make_terminal_event(self) -> _UserChatManagerEvent:
        frags = self._drain_token_buffer_as_far_as_possible(final=True)
        self._commit_model_turn()
        self._terminal_pending = False
        error_code = self._terminal_error_code
        self._terminal_error_code = None
        return _UserChatManagerEvent(frags, True, error_code)

    def _commit_model_turn(self):
        """
        Commit the just-generated model turn onto history as verbatim tokens.

        The opener `<|turn>model\n` is already in history (appended on submit) and
        the model emits its own closing `<turn|>`, so the raw tokens are kept
        exactly as generated — only the trailing newline that separates turns is
        missing, since generation stops at `<turn|>` before it. With
        drop_thoughts_from_history set, thought channels are spliced out.
        Idempotent: clears the accumulator so repeat calls no-op.
        """
        body, self._gen_tokens = self._gen_tokens, []
        if _config.drop_thoughts_from_history:
            body = _strip_channel(body, "thought")
        if body:
            self.history_tokens.extend(body)
            self.history_tokens.extend(_tokenizer.tokenize("\n", add_bos=False))

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
    state = _UserChatManager()

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
                                "channel": frag.channel,
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
                if advancements.error_code == MAX_CONTEXT_WINDOW_EXCEEDED_ERROR:
                    await ws.send_json(
                        {
                            "type": "error",
                            "code": MAX_CONTEXT_WINDOW_EXCEEDED_ERROR,
                            "detail": (
                                f"generation reached max context length ({_config.context_len} tokens); "
                                "truncate the conversation"
                            ),
                        }
                    )
                else:
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
                                    channel = ev.get("channel")
                                    if not isinstance(channel, str) and channel is not None:
                                        await _chat_err(ws, "`channel` must be string or null")
                                    parsed_ev = tokenizer_mod.ResponseFragment(content, channel)

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
                                f"provided chat history exceeds max context length ({_config.context_len} tokens)",
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
                                else f"prompt exceeds max context length ({_config.context_len} tokens); truncate the conversation"
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
                                else f"tool output exceeds max context length ({_config.context_len} tokens); truncate the conversation"
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
