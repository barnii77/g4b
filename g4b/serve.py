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

app = FastAPI()


async def _chat_err(ws: WebSocket, error: str):
    await ws.send_json({"type": "error", "detail": error})
    await ws.close(code=4000, reason=error)
    raise RuntimeError(error)


@app.websocket("/api/v1/chat")
async def chat(ws: WebSocket):
    await ws.accept()

    is_init = False
    try:
        while True:
            event = await ws.receive_json()

            match event.get("type"):
                case "user.init":
                    if is_init:
                        await _chat_err(ws, "already initialized")
                    is_init = True
                    ...

                case "user.prompt":
                    if not is_init:
                        await _chat_err(ws, "not initialized")
                    ...

                case "tool.output":
                    if not is_init:
                        await _chat_err(ws, "not initialized")
                    ...

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
