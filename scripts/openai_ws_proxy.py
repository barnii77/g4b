"""
Single-file proxy for the g4b websocket chat interface, backed by OpenAI's
Responses API.

Install:
    pip install fastapi uvicorn openai

Run:
    export OPENAI_API_KEY=...
    export OPENAI_MODEL=gpt-4.1-mini   # optional
    python openai_ws_proxy.py --host 127.0.0.1 --port 8000

WebSocket:
    ws://127.0.0.1:8000/api/v1/chat

Implements the event shape documented in g4b/serve.py:
    client -> server:
      {"type":"user.init", "chat":[...]}
      {"type":"user.prompt", "content":"...", "obfuscation":"..."}
      {"type":"tool.output", "output":"...", "obfuscation":"..."}
      {"type":"user.abort", "obfuscation":"..."}
      {"type":"ping"}

    server -> client:
      {"type":"assistant.response.chunk", "content":"...", "obfuscation":"..."}
      {"type":"assistant.response.done", "obfuscation":"..."}
      {"type":"pong"}
      {"type":"error", "code":1, "detail":"..."}

This proxy is text-only. It accepts tool.output events as context, but it does
not currently emit tool.call events.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import json
import os
import random
import secrets
from dataclasses import dataclass, field
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI


GENERIC_ERROR = 1
MAX_CONTEXT_WINDOW_EXCEEDED_ERROR = 2

DEFAULT_INSTRUCTIONS = (
    "You are a helpful assistant. Answer the user's latest message. "
    "Be concise by default, but give technical detail when the user asks for it."
)

MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
INSTRUCTIONS = os.environ.get("OPENAI_INSTRUCTIONS", DEFAULT_INSTRUCTIONS)
MAX_OUTPUT_TOKENS = int(os.environ.get("OPENAI_MAX_OUTPUT_TOKENS", "1024"))
OBFUSCATE = os.environ.get("G4B_PROXY_OBFUSCATE", "1").lower() not in {"0", "false", "no"}

client = AsyncOpenAI()
app = FastAPI(title="g4b websocket proxy", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@dataclass
class ChatState:
    initialized: bool = False
    history: list[dict[str, Any]] = field(default_factory=list)
    active_task: asyncio.Task[None] | None = None


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"ok": True, "model": MODEL}


@app.websocket("/api/v1/chat")
async def chat(ws: WebSocket) -> None:
    await ws.accept()
    state = ChatState()

    try:
        while True:
            event = await ws.receive_json()
            if not isinstance(event, dict):
                await send_error(ws, "websocket event must be a JSON object", close=True)
                return

            ty = event.get("type")
            if not isinstance(ty, str):
                await send_error(ws, "`type` must be string", close=True)
                return

            if ty == "ping":
                await ws.send_json({"type": "pong"})
                continue

            if ty == "user.init":
                if state.initialized:
                    await send_error(ws, "already initialized", close=True)
                    return
                chat_events = event.get("chat")
                if not isinstance(chat_events, list):
                    await send_error(ws, "`chat` must be list", close=True)
                    return
                normalized: list[dict[str, Any]] = []
                for ev in chat_events:
                    ok, normalized_ev_or_err = normalize_history_event(ev)
                    if not ok:
                        await send_error(ws, str(normalized_ev_or_err), close=True)
                        return
                    normalized.append(normalized_ev_or_err)  # type: ignore[arg-type]
                state.history = normalized
                state.initialized = True
                continue

            if not state.initialized:
                await send_error(ws, "not initialized", close=True)
                return

            if ty == "user.prompt":
                content = event.get("content")
                if not isinstance(content, str):
                    await send_error(ws, "`content` must be string", close=True)
                    return
                if state.active_task and not state.active_task.done():
                    await send_error(ws, "model response already in progress; send user.abort first")
                    continue
                state.history.append({"type": "user.prompt", "content": content})
                state.active_task = asyncio.create_task(stream_openai_response(ws, state))
                continue

            if ty == "tool.output":
                output = event.get("output")
                if not isinstance(output, str):
                    await send_error(ws, "`output` must be string", close=True)
                    return
                if state.active_task and not state.active_task.done():
                    await send_error(ws, "model response already in progress; send user.abort first")
                    continue
                state.history.append({"type": "tool.output", "output": output})
                state.active_task = asyncio.create_task(stream_openai_response(ws, state))
                continue

            if ty == "user.abort":
                if state.active_task and not state.active_task.done():
                    state.active_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await state.active_task
                state.active_task = None
                # Frontends usually want a terminal event so their spinner stops.
                await ws.send_json(add_obfuscation({"type": "assistant.response.done"}))
                continue

            await send_error(ws, f"unrecognized event type `{ty}`", close=True)
            return

    except WebSocketDisconnect:
        if state.active_task and not state.active_task.done():
            state.active_task.cancel()
    finally:
        if state.active_task and not state.active_task.done():
            state.active_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await state.active_task


async def stream_openai_response(ws: WebSocket, state: ChatState) -> None:
    """Stream one assistant turn from OpenAI back as g4b websocket events."""
    response_text_parts: list[str] = []
    transcript = render_transcript(state.history)

    try:
        async with client.responses.stream(
            model=MODEL,
            instructions=INSTRUCTIONS,
            input=transcript,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        ) as stream:
            async for ev in stream:
                ev_type = getattr(ev, "type", None)

                if ev_type == "response.output_text.delta":
                    delta = getattr(ev, "delta", "")
                    if delta:
                        response_text_parts.append(delta)
                        await ws.send_json(
                            add_obfuscation(
                                {
                                    "type": "assistant.response.chunk",
                                    "content": delta,
                                }
                            )
                        )

                elif ev_type == "response.failed":
                    detail = extract_event_error(ev) or "OpenAI response failed"
                    await send_error(ws, detail)
                    return

                elif ev_type == "response.incomplete":
                    detail = extract_event_error(ev) or "OpenAI response incomplete"
                    await send_error(ws, detail, code=MAX_CONTEXT_WINDOW_EXCEEDED_ERROR)
                    return

                # Other stream events exist; ignore them for this text-only shim.

        full_text = "".join(response_text_parts)
        if full_text:
            # Store one compact assistant chunk in history rather than thousands of deltas.
            state.history.append({"type": "assistant.response.chunk", "content": full_text})

        await ws.send_json(add_obfuscation({"type": "assistant.response.done"}))

    except asyncio.CancelledError:
        raise
    except Exception as exc:
        await send_error(ws, f"OpenAI proxy error: {type(exc).__name__}: {exc}")
    finally:
        state.active_task = None


def normalize_history_event(ev: Any) -> tuple[bool, dict[str, Any] | str]:
    if not isinstance(ev, dict):
        return False, "history event must be JSON object"
    ty = ev.get("type")
    if not isinstance(ty, str):
        return False, "history event `type` must be string"

    if ty == "user.prompt":
        content = ev.get("content")
        if not isinstance(content, str):
            return False, "user.prompt `content` must be string"
        return True, {"type": ty, "content": content}

    if ty == "assistant.response.chunk":
        content = ev.get("content")
        if not isinstance(content, str):
            return False, "assistant.response.chunk `content` must be string"
        return True, {"type": ty, "content": content}

    if ty == "tool.output":
        output = ev.get("output")
        if not isinstance(output, str):
            return False, "tool.output `output` must be string"
        return True, {"type": ty, "output": output}

    if ty == "tool.call":
        call = ev.get("call")
        if not isinstance(call, dict):
            return False, "tool.call `call` must be JSON object"
        return True, {"type": ty, "call": call}

    if ty == "assistant.response.done":
        # Harmless terminal marker in persisted frontend logs; no need to render it.
        return True, {"type": ty}

    return False, f"event type `{ty}` disallowed in user.init chat history"


def render_transcript(events: list[dict[str, Any]]) -> str:
    """
    Render the frontend's event log into one text input for Responses API.

    This avoids depending on the exact Responses conversation-item schema and is
    sufficient for a frontend demo. Later, replace this with structured
    Responses input or previous_response_id tracking.
    """
    out: list[str] = []
    assistant_buf: list[str] = []

    def flush_assistant() -> None:
        nonlocal assistant_buf
        if assistant_buf:
            out.append("Assistant: " + "".join(assistant_buf).strip())
            assistant_buf = []

    for ev in events:
        ty = ev.get("type")
        if ty == "assistant.response.chunk":
            assistant_buf.append(str(ev.get("content", "")))
            continue

        flush_assistant()

        if ty == "user.prompt":
            out.append("User: " + str(ev.get("content", "")).strip())
        elif ty == "tool.call":
            out.append("Assistant tool call: " + json.dumps(ev.get("call", {}), ensure_ascii=False))
        elif ty == "tool.output":
            out.append("Tool output: " + str(ev.get("output", "")).strip())
        elif ty == "assistant.response.done":
            pass

    flush_assistant()
    out.append("Assistant:")
    return "\n\n".join(part for part in out if part)


async def send_error(ws: WebSocket, detail: str, code: int = GENERIC_ERROR, close: bool = False) -> None:
    with contextlib.suppress(Exception):
        await ws.send_json({"type": "error", "code": code, "detail": detail})
    if close:
        with contextlib.suppress(Exception):
            await ws.close(code=4000, reason=detail[:120])


def add_obfuscation(resp: dict[str, Any]) -> dict[str, Any]:
    if not OBFUSCATE:
        return resp
    resp = dict(resp)
    payload_len = len(json.dumps(resp, ensure_ascii=False, separators=(",", ":")))
    n = max(0, round(random.uniform(payload_len // 2, 3 * payload_len // 2)))
    resp["obfuscation"] = base64.b64encode(secrets.token_bytes(n)).decode("ascii")
    return resp


def extract_event_error(ev: Any) -> str | None:
    # Be defensive across SDK versions / event shapes.
    response = getattr(ev, "response", None)
    error = getattr(response, "error", None) if response is not None else getattr(ev, "error", None)
    if error is None:
        return None
    if isinstance(error, str):
        return error
    message = getattr(error, "message", None)
    if message:
        return str(message)
    return str(error)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set")

    uvicorn.run("openai_ws_proxy:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
