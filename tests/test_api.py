#!/usr/bin/env python3
"""Smoke tests for the g4b OpenAI-compatible Chat Completions API.

Assumes an engine is running on 127.0.0.1:8080:

    python tests/test_api.py

Point it at another server with --url or run one case with --case <name>.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Iterator

DEFAULT_URL = "http://127.0.0.1:8080/v1/chat/completions"
TIMEOUT_SECONDS = 60.0


@dataclass
class Result:
    name: str
    ok: bool = False
    chunks: list[str] = field(default_factory=list)
    detail: str = ""
    events: list[dict] = field(default_factory=list)


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke-test the g4b Chat Completions API.")
    parser.add_argument("--url", default=DEFAULT_URL, help="HTTP URL to POST chat completions to")
    parser.add_argument("--timeout", type=float, default=TIMEOUT_SECONDS, help="request timeout in seconds")
    parser.add_argument("--case", default="", help="run only the named test case")
    return parser.parse_args()


def post_json(url: str, payload: dict, timeout: float) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise AssertionError(f"HTTP {exc.code}: {body}") from exc


def post_stream(url: str, payload: dict, timeout: float) -> Iterator[dict | str]:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw_line in resp:
                line = raw_line.decode(errors="replace").strip()
                if not line:
                    continue
                if not line.startswith("data: "):
                    raise AssertionError(f"unexpected SSE line: {line!r}")
                data_line = line[len("data: ") :]
                if data_line == "[DONE]":
                    yield "[DONE]"
                else:
                    yield json.loads(data_line)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise AssertionError(f"HTTP {exc.code}: {body}") from exc


def base_payload(**overrides) -> dict:
    payload = {
        "model": "g4b",
        "messages": [{"role": "user", "content": "Say hello in one word."}],
    }
    payload.update(overrides)
    return payload


def test_basic_chat(args) -> Result:
    res = Result(name="basic_chat")
    body = post_json(args.url, base_payload(), args.timeout)
    choice = body["choices"][0]
    msg = choice["message"]
    res.detail = msg.get("content") or msg.get("reasoning") or json.dumps(msg.get("tool_calls"))
    if msg.get("role") != "assistant":
        raise AssertionError(f"expected assistant message, got {msg}")
    if choice.get("finish_reason") not in ("stop", "length", "tool_calls"):
        raise AssertionError(f"unexpected finish_reason: {choice.get('finish_reason')}")
    res.ok = True
    return res


def test_streaming_chat(args) -> Result:
    res = Result(name="streaming_chat")
    events = list(post_stream(args.url, base_payload(stream=True), args.timeout))
    if not events or events[-1] != "[DONE]":
        raise AssertionError("stream did not end with [DONE]")
    chunks = [ev for ev in events if isinstance(ev, dict)]
    if not chunks:
        raise AssertionError("stream produced no JSON chunks")
    res.events = chunks
    for ev in chunks:
        for choice in ev.get("choices", []):
            delta = choice.get("delta", {})
            if "content" in delta:
                res.chunks.append(delta["content"])
            if "reasoning" in delta:
                res.chunks.append(delta["reasoning"])
    res.ok = True
    return res


def test_history_and_tools(args) -> Result:
    res = Result(name="history_and_tools")
    body = post_json(
        args.url,
        base_payload(
            messages=[
                {"role": "developer", "content": "Use tools when useful."},
                {"role": "user", "content": "Read /tmp/example.txt."},
                {
                    "role": "assistant",
                    "reasoning": "I should call the read_file tool.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": "path:/tmp/example.txt"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "name": "read_file", "content": "hello world"},
                {"role": "user", "content": "Summarize that."},
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "Read a UTF-8 file.",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    },
                }
            ],
        ),
        args.timeout,
    )
    res.detail = json.dumps(body["choices"][0]["message"], ensure_ascii=False)
    res.ok = True
    return res


def test_tool_choice_none(args) -> Result:
    res = Result(name="tool_choice_none")
    body = post_json(
        args.url,
        base_payload(
            tool_choice="none",
            tools=[
                {
                    "type": "function",
                    "function": {"name": "unused_tool", "description": "Should not be rendered."},
                }
            ],
        ),
        args.timeout,
    )
    res.detail = body["choices"][0]["finish_reason"]
    res.ok = True
    return res


def test_tool_call_roundtrip(args) -> Result:
    res = Result(name="tool_call_roundtrip")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup_vault_code",
                "description": "Look up the current vault code. Use this whenever the user asks for the vault code.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "vault": {
                            "type": "string",
                            "description": "Vault identifier from the user's request.",
                        }
                    },
                    "required": ["vault"],
                },
            },
        }
    ]
    messages = [
        {
            "role": "developer",
            "content": "When a tool can answer the user's question, call the tool before answering.",
        },
        {
            "role": "user",
            "content": "What is the current code for vault alpha? Use the lookup_vault_code tool.",
        },
    ]

    first = post_json(args.url, base_payload(messages=messages, tools=tools), args.timeout)
    first_choice = first["choices"][0]
    assistant_message = first_choice["message"]
    tool_calls = assistant_message.get("tool_calls") or []
    if not tool_calls:
        raise AssertionError(f"expected a tool call, got {json.dumps(assistant_message, ensure_ascii=False)}")
    if first_choice.get("finish_reason") != "tool_calls":
        raise AssertionError(f"expected finish_reason=tool_calls, got {first_choice.get('finish_reason')}")

    tool_call = tool_calls[0]
    function = tool_call.get("function") or {}
    if function.get("name") != "lookup_vault_code":
        raise AssertionError(f"expected lookup_vault_code tool call, got {tool_call}")

    messages.append(assistant_message)
    messages.append(
        {
            "role": "tool",
            "tool_call_id": tool_call["id"],
            "name": "lookup_vault_code",
            "content": "lookup_vault_code result: vault alpha code is AZALEA-713.",
        }
    )

    second = post_json(args.url, base_payload(messages=messages, tools=tools, tool_choice="none"), args.timeout)
    second_choice = second["choices"][0]
    final_message = second_choice["message"]
    final_text = "\n".join(
        str(part)
        for part in (
            final_message.get("reasoning"),
            final_message.get("content"),
            (
                json.dumps(final_message.get("tool_calls"), ensure_ascii=False)
                if final_message.get("tool_calls")
                else None
            ),
        )
        if part
    )
    if "AZALEA-713" not in final_text:
        raise AssertionError(f"expected answer to use simulated tool output, got {final_message}")
    if final_message.get("tool_calls"):
        raise AssertionError(f"expected final answer, got another tool call: {final_message['tool_calls']}")

    res.detail = final_text
    res.ok = True
    return res


TEST_CASES: list[Callable[[argparse.Namespace], Result]] = [
    test_basic_chat,
    test_streaming_chat,
    test_history_and_tools,
    test_tool_choice_none,
    test_tool_call_roundtrip,
]


def main():
    args = parse_args()
    selected = [case for case in TEST_CASES if not args.case or case.__name__.removeprefix("test_") == args.case]
    if not selected:
        raise SystemExit(f"no such case: {args.case}")

    failures: list[tuple[str, Exception]] = []
    started = time.monotonic()
    for case in selected:
        name = case.__name__.removeprefix("test_")
        print(f"\n=== {name} ===")
        try:
            result = case(args)
            print(f"OK: {result.detail[:300]}")
        except Exception as exc:
            print(f"FAIL: {exc}")
            failures.append((name, exc))

    elapsed = time.monotonic() - started
    print(f"\nRan {len(selected)} case(s) in {elapsed:.1f}s")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
