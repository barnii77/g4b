#!/usr/bin/env python3
"""Extended smoke tests for the g4b websocket chat API.

Assumes an engine is running on 127.0.0.1:8080.  Run from within the project
venv so the `websocket-client` package is available:

    python -m websocket -h
    python tmp/test_api.py

You can point it at a different host/port with --url or run a single test with
--case <name>.
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from typing import Callable

import websocket

DEFAULT_URL = "ws://127.0.0.1:8080/api/v1/chat"
TIMEOUT_SECONDS = 60.0
RECV_SHORT_TIMEOUT = 3.0


@dataclass
class Failure:
    message: str


@dataclass
class Result:
    name: str
    ok: bool = False
    chunks: list[str] = field(default_factory=list)
    detail: str = ""
    events: list[dict] = field(default_factory=list)


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke-test the g4b chat websocket API.")
    parser.add_argument("--url", default=DEFAULT_URL, help="websocket URL to connect to")
    parser.add_argument("--timeout", type=float, default=TIMEOUT_SECONDS, help="default receive timeout in seconds")
    parser.add_argument("--case", default="", help="run only the named test case")
    return parser.parse_args()


def connect(url: str, timeout: float) -> websocket.WebSocket:
    print(f"Connecting to {url} ...")
    ws = websocket.create_connection(url, timeout=timeout, enable_multithread=True)
    print("Connected.")
    return ws


def send(ws: websocket.WebSocket, msg: dict) -> None:
    text = json.dumps(msg)
    print(f"  -> {msg['type']}")
    ws.send(text)


def recv(ws: websocket.WebSocket, timeout: float) -> dict | None:
    ws.settimeout(timeout)
    try:
        text = ws.recv()
    except websocket.WebSocketTimeoutException:
        return None
    except websocket.WebSocketConnectionClosedException as exc:
        raise exc
    finally:
        ws.settimeout(None)

    if not isinstance(text, str):
        text = text.decode("utf-8", errors="replace")
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        print(f"  ERROR: invalid JSON from server: {exc}")
        print(f"  Raw frame: {text!r}")
        return None


def recv_any(
    ws: websocket.WebSocket,
    timeout: float,
    stop_conditions: list[Callable[[dict], bool]] | None = None,
) -> tuple[list[dict], bool]:
    """Receive events until timeout, a stop condition fires, or the socket closes.

    Returns (events, stopped_early).
    """
    events: list[dict] = []
    started = time.monotonic()
    stopped = False
    while time.monotonic() - started < timeout:
        try:
            msg = recv(ws, max(0.1, timeout - (time.monotonic() - started)))
        except websocket.WebSocketConnectionClosedException:
            break
        if msg is None:
            continue
        events.append(msg)
        if stop_conditions and any(cond(msg) for cond in stop_conditions):
            stopped = True
            break
    return events, stopped


def collect_response(
    ws: websocket.WebSocket,
    timeout: float,
) -> tuple[list[str], list[dict]]:
    """Collect assistant.response.chunk events until assistant.response.done.

    Returns the concatenated text and the full event log.
    """
    chunks: list[str] = []
    events: list[dict] = []
    started = time.monotonic()

    while time.monotonic() - started < timeout:
        try:
            msg = recv(ws, max(0.5, timeout - (time.monotonic() - started)))
        except websocket.WebSocketConnectionClosedException as exc:
            raise AssertionError(f"connection closed before assistant.response.done: {exc}") from exc
        if msg is None:
            continue

        events.append(msg)
        pretty = json.dumps(msg, ensure_ascii=False, indent=None)
        print(f"  <- {pretty[:200]}{'...' if len(pretty) > 200 else ''}")

        msg_type = msg.get("type")
        if msg_type == "error":
            raise AssertionError(f"server error code={msg.get('code')}: {msg.get('detail')}")
        if msg_type == "assistant.response.chunk":
            chunks.append(msg.get("content", ""))
        elif msg_type == "assistant.response.done":
            return chunks, events
        elif msg_type == "tool.call":
            print(f"  Tool call: {msg.get('call')}")

    raise AssertionError(f"timed out after {timeout}s waiting for assistant.response.done")


def assert_no_error_events(events: list[dict]) -> None:
    for ev in events:
        if ev.get("type") == "error":
            raise AssertionError(f"unexpected error event: {ev}")


# -----------------------------------------------------------------------------
# Individual test cases
# -----------------------------------------------------------------------------


def test_basic_chat(args) -> Result:
    """Simple init + prompt -> response."""
    res = Result(name="basic_chat")
    ws = connect(args.url, args.timeout)
    try:
        send(ws, {"type": "user.init", "chat": []})
        send(ws, {"type": "user.prompt", "content": "Say hello in one word."})
        chunks, events = collect_response(ws, args.timeout)
        res.chunks = chunks
        res.events = events
        if not chunks:
            raise AssertionError("received no response chunks")
        print(f"  Response length: {sum(len(c) for c in chunks)} chars in {len(chunks)} chunk(s)")
    finally:
        ws.close()
    res.ok = True
    return res


def test_long_prompt(args) -> Result:
    """A very long prompt should cause response chunking and eventually finish."""
    res = Result(name="long_prompt")
    ws = connect(args.url, args.timeout)
    try:
        send(ws, {"type": "user.init", "chat": []})
        long_text = " word " * 2000  # well over the 48-token chunk buffer
        send(ws, {"type": "user.prompt", "content": f"Repeat the following text exactly:\n{long_text}"})
        chunks, events = collect_response(ws, args.timeout)
        res.chunks = chunks
        res.events = events
        print(f"  Received {len(chunks)} chunk(s); response length {sum(len(c) for c in chunks)} chars")
        if len(chunks) < 2:
            raise AssertionError("expected a multi-chunk response for a long prompt")
    finally:
        ws.close()
    res.ok = True
    return res


def test_multi_turn(args) -> Result:
    """Two prompts on the same websocket verify active_request cleanup."""
    res = Result(name="multi_turn")
    ws = connect(args.url, args.timeout)
    try:
        send(ws, {"type": "user.init", "chat": []})

        send(ws, {"type": "user.prompt", "content": "Count to three separated by commas."})
        chunks1, events1 = collect_response(ws, args.timeout)
        res.chunks.extend(chunks1)
        res.events.extend(events1)
        full1 = "".join(chunks1)
        print(f"  Turn 1 response length: {len(full1)}")

        send(ws, {"type": "user.prompt", "content": "Now count backwards from three."})
        chunks2, events2 = collect_response(ws, args.timeout)
        res.chunks.extend(chunks2)
        res.events.extend(events2)
        full2 = "".join(chunks2)
        print(f"  Turn 2 response length: {len(full2)}")

        if not full1 or not full2:
            raise AssertionError("one of the turns produced no text")
    finally:
        ws.close()
    res.ok = True
    return res


def test_explicit_abort(args) -> Result:
    """Abort an in-flight request and then successfully issue a second prompt."""
    res = Result(name="explicit_abort")
    ws = connect(args.url, args.timeout)
    try:
        send(ws, {"type": "user.init", "chat": []})
        send(ws, {"type": "user.prompt", "content": "Write a very long poem about CUDA."})

        # Let a few chunks (or at least some time) pass.
        time.sleep(0.5)
        send(ws, {"type": "user.abort"})

        # Drain any in-flight chunks for a short while.  We do not expect a
        # terminal event after an abort.
        events, _ = recv_any(ws, RECV_SHORT_TIMEOUT)
        res.events.extend(events)
        print(f"  Drained {len(events)} event(s) after abort")
        assert_no_error_events(events)

        # A follow-up prompt should still work.
        send(ws, {"type": "user.prompt", "content": "Say a single word."})
        chunks, events2 = collect_response(ws, args.timeout)
        res.chunks = chunks
        res.events.extend(events2)
        if not chunks:
            raise AssertionError("follow-up prompt produced no response after abort")
    finally:
        ws.close()
    res.ok = True
    return res


def test_disconnect_aborts(args) -> Result:
    """Disconnect while a request is active, then reconnect and chat again."""
    res = Result(name="disconnect_aborts")
    # First connection: start a request and immediately drop it.
    ws1 = connect(args.url, args.timeout)
    try:
        send(ws1, {"type": "user.init", "chat": []})
        send(ws1, {"type": "user.prompt", "content": "Explain quantum mechanics in great detail."})
        # Give the scheduler a moment to pick the request up.
        time.sleep(0.3)
    finally:
        ws1.close()

    # Start a fresh connection; the engine should still accept new requests.
    ws2 = connect(args.url, args.timeout)
    try:
        send(ws2, {"type": "user.init", "chat": []})
        send(ws2, {"type": "user.prompt", "content": "Hi."})
        chunks, events = collect_response(ws2, args.timeout)
        res.chunks = chunks
        res.events = events
        if not chunks:
            raise AssertionError("second connection produced no response after disconnect")
    finally:
        ws2.close()
    res.ok = True
    return res


def test_init_twice_errors(args) -> Result:
    """Sending user.init twice should error and close the socket."""
    res = Result(name="init_twice_errors")
    ws = connect(args.url, args.timeout)
    try:
        send(ws, {"type": "user.init", "chat": []})
        send(ws, {"type": "user.init", "chat": []})

        # Server should close after the error; collect whatever it sends.
        try:
            events, _ = recv_any(ws, RECV_SHORT_TIMEOUT)
        except websocket.WebSocketConnectionClosedException:
            events = []
        res.events = events
        if not any(e.get("type") == "error" for e in events):
            raise AssertionError("expected an error event for a duplicate user.init")
    finally:
        ws.close()
    res.ok = True
    return res


def test_prompt_before_init(args) -> Result:
    """Sending user.prompt before user.init should error and close the socket."""
    res = Result(name="prompt_before_init")
    ws = connect(args.url, args.timeout)
    try:
        send(ws, {"type": "user.prompt", "content": "hello?"})

        try:
            events, _ = recv_any(ws, RECV_SHORT_TIMEOUT)
        except websocket.WebSocketConnectionClosedException:
            events = []
        res.events = events
        if not any(e.get("type") == "error" for e in events):
            raise AssertionError("expected an error event for prompt before init")
    finally:
        ws.close()
    res.ok = True
    return res


def test_ping(args) -> Result:
    """A ping should produce a pong."""
    res = Result(name="ping")
    ws = connect(args.url, args.timeout)
    try:
        send(ws, {"type": "user.init", "chat": []})
        send(ws, {"type": "ping"})
        events, stopped = recv_any(ws, RECV_SHORT_TIMEOUT, stop_conditions=[lambda m: m.get("type") == "pong"])
        res.events = events
        if not stopped:
            raise AssertionError("did not receive a pong for ping")
    finally:
        ws.close()
    res.ok = True
    return res


def test_init_with_history(args) -> Result:
    """user.init can carry a prior turn and the engine should accept it."""
    res = Result(name="init_with_history")
    ws = connect(args.url, args.timeout)
    try:
        send(ws, {
            "type": "user.init",
            "chat": [
                {"type": "user.prompt", "content": "My name is Testbot."},
                {"type": "assistant.response.chunk", "content": "Hello Testbot."},
            ],
        })
        send(ws, {"type": "user.prompt", "content": "Do you remember my name?"})
        chunks, events = collect_response(ws, args.timeout)
        res.chunks = chunks
        res.events = events
        if not chunks:
            raise AssertionError("no response for prompt after initialized history")
    finally:
        ws.close()
    res.ok = True
    return res


TEST_CASES: list[Callable[[argparse.Namespace], Result]] = [
    test_basic_chat,
    test_long_prompt,
    test_multi_turn,
    test_explicit_abort,
    test_disconnect_aborts,
    test_init_twice_errors,
    test_prompt_before_init,
    test_ping,
    test_init_with_history,
]


def run_case(case: Callable[[argparse.Namespace], Result], args: argparse.Namespace) -> Result:
    print(f"\n=== Running: {case.__name__} ===")
    try:
        return case(args)
    except AssertionError as exc:
        return Result(name=case.__name__, ok=False, detail=str(exc))
    except Exception as exc:
        return Result(name=case.__name__, ok=False, detail=f"{type(exc).__name__}: {exc}")


def main() -> int:
    args = parse_args()

    cases = TEST_CASES
    if args.case:
        cases = [c for c in TEST_CASES if c.__name__ == args.case]
        if not cases:
            print(f"Unknown test case: {args.case}")
            print(f"Available: {', '.join(c.__name__ for c in TEST_CASES)}")
            return 1

    results = [run_case(c, args) for c in cases]

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    failures = 0
    for r in results:
        status = "PASS" if r.ok else "FAIL"
        print(f"  [{status}] {r.name:<30} {r.detail}")
        if not r.ok:
            failures += 1
        elif r.chunks:
            full = "".join(r.chunks)
            preview = full[:80].replace("\n", "\\n")
            if len(full) > 80:
                preview += "..."
            print(f"       response: {preview}")

    print("-" * 60)
    if failures:
        print(f"{failures}/{len(results)} test(s) failed.")
        return 1
    print(f"All {len(results)} test(s) passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
