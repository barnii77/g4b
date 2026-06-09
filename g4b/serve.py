"""
API SPEC:
- /api/v1/chat:
  - opens a text websocket. websocket json events below.
  - client -> server:
    - {"type": "user.init", "chat": [<event>, <event>, ...]}
    - {"type": "user.prompt", "content": "my prompt", "obfuscation": <obfuscation>}
    - {"type": "tool.output", "output": "... (may be serialized json) ...", "obfuscation": <obfuscation>}
    - {"type": "ping"}
  - server -> client:
    - {"type": "tool.call", "call": {"name": "mytool", args: {"arg1": <anything>, "arg2": <anything>}}, "obfuscation": <obfuscation>}
    - {"type": "assistant.response.chunk", content: "...", "obfuscation": <obfuscation>}
    - {"type": "assistant.response.done", "obfuscation": <obfuscation>}
    - {"type": "pong"}
    - {"type": "error", "detail": "..."}
- <obfuscation>: An optional but recommended crypto-random base64-encoded sequence of X bytes where X is the absolute value of a sample from a gaussian distribution N(L, L) and L is the length of the payload without obfuscation. The purpose of this is to prevent certain side channel attacks. Google "openai api obfuscation strings" for more information.
- <event>: any json event documented above except user.init
"""

import uvicorn
import threading
import asyncio
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from g4b import scheduler as scheduler_mod

app = FastAPI()
_scheduler: "scheduler_mod.Scheduler"


def register_scheduler(sched: "scheduler_mod.Scheduler"):
    global _scheduler
    _scheduler = sched


class _UserChatManager:
    def __init__(self):
        ...

    def process_prompt(self, content: str):
        ...

    def process_tool_output(self, output: str):
        ...

    def process_tool_call(self, call: dict):
        ...

    def process_response_chunk(self, content: str):
        ...

    def process_response_done(self):
        ...


@app.websocket("/api/v1/chat")
async def chat(ws: WebSocket):
    await ws.accept()

    is_init = False
    state = _UserChatManager()
    try:
        while True:
            event = await ws.receive_json()

            # Handle client -> server event
            ty = await _assert_string(event.get("type"), "`type` must be string", ws)
            match ty:
                case "user.init":
                    if is_init:
                        await _chat_err(ws, "already initialized")
                    is_init = True
                    prev_events = event.get("chat")
                    if not isinstance(prev_events, list):
                        await _chat_err(ws, "`chat` must be list")
                    for ev in prev_events:
                        if not isinstance(ev, dict):
                            await _chat_err(ws, "`event` must be json object")
                        ty = ev.get("type")
                        if not isinstance(ty, str):
                            await _chat_err(ws, "`type` must be string")
                        match ty:
                            case "user.prompt":
                                prompt = await _assert_string(ev.get("content"), "`content` must be string", ws)
                                state.process_prompt(prompt)
                            case "tool.output":
                                output = await _assert_string(ev.get("output"), "`output` must be string", ws)
                                state.process_tool_output(output)
                            case "tool.call":
                                call = await _assert_dict(ev.get("call"), "`call` must be json object", ws)
                                state.process_tool_call(call)
                            case "assistant.response.chunk":
                                content = await _assert_string(ev.get("content"), "`content` must be string", ws)
                                state.process_response_chunk(content)
                            case "assistant.response.done":
                                state.process_response_done()
                            case _:
                                await _chat_err(ws, f"event type `{ty}` disallowed in `events` for init")

                case "user.prompt":
                    if not is_init:
                        await _chat_err(ws, "not initialized")
                    content = await _assert_string(event.get("content"), "`content` must be string", ws)
                    state.process_prompt(content)

                case "tool.output":
                    if not is_init:
                        await _chat_err(ws, "not initialized")
                    output = await _assert_string(event.get("output"), "`output` must be string", ws)
                    state.process_tool_output(output)

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


async def _chat_err(ws: WebSocket, error: str):
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
