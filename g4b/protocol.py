"""
Example non-streaming request to POST /v1/chat/completions:

{
  "model": "local-gemma",
  "messages": [
    {"role": "developer", "content": "Use the available tools when needed."},
    {"role": "user", "content": "What is in /tmp/example.txt?"},
    {
      "role": "assistant",
      "reasoning": "I need to inspect the file before answering.",
      "tool_calls": [
        {
          "id": "call_1",
          "type": "function",
          "function": {
            "name": "read_file",
            "arguments": {"path": "/tmp/example.txt"}
          }
        }
      ]
    },
    {"role": "tool", "tool_call_id": "call_1", "content": "hello world"},
    {"role": "user", "content": [{"type": "text", "text": "Summarize that."}]}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "read_file",
        "description": "Read a UTF-8 text file.",
        "parameters": {
          "type": "object",
          "properties": {"path": {"type": "string"}},
          "required": ["path"]
        }
      }
    }
  ],
  "tool_choice": "auto",
  "stream": false,
  "n": 1
}

Example non-streaming response:

{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "local-gemma",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "reasoning": "The tool output contains a short greeting.",
        "content": "The file contains: hello world"
      },
      "finish_reason": "stop",
      "logprobs": null
    }
  ],
  "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
}

Example streaming response body:

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1234567890,"model":"local-gemma","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null,"logprobs":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1234567890,"model":"local-gemma","choices":[{"index":0,"delta":{"reasoning":"The tool output contains "},"finish_reason":null,"logprobs":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1234567890,"model":"local-gemma","choices":[{"index":0,"delta":{"content":"The file contains: hello world"},"finish_reason":null,"logprobs":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1234567890,"model":"local-gemma","choices":[{"index":0,"delta":{},"finish_reason":"stop","logprobs":null}]}

data: [DONE]

Example complete tool-call delta, emitted after the full call is generated:

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1234567890,"model":"local-gemma","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_2","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"/tmp/example.txt\"}"}}]},"finish_reason":null,"logprobs":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1234567890,"model":"local-gemma","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls","logprobs":null}]}

data: [DONE]

Example error response:

{
  "error": {
    "message": "messages must not be empty",
    "type": "invalid_request_error",
    "param": "messages",
    "code": null
  }
}
"""

from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class APIBase(BaseModel):
    """Base class for protocol structs.

    Unknown fields are intentionally ignored. This lets common OpenAI client
    fields pass parsing even when g4b does not implement them.
    """

    model_config = ConfigDict(extra="ignore")


class ContentPart(APIBase):
    """Text-only message content part: {"type": "text", "text": "..."}."""

    type: Literal["text"]
    text: str


class FunctionCall(APIBase):
    """Complete function call payload on a full assistant message."""

    name: str
    arguments: str | dict[str, Any] = ""


class FunctionCallDelta(APIBase):
    """Streaming function-call patch.

    Arguments are string fragments in the OpenAI wire protocol. g4b may emit
    exactly one complete fragment instead of partial syntactically invalid ones.
    """

    name: str | None = None
    arguments: str | None = None


class ToolCall(APIBase):
    """Complete assistant tool call."""

    id: str
    function: FunctionCall
    type: Literal["function"] = "function"


class ToolCallDelta(APIBase):
    """Streaming tool-call patch keyed by `index`."""

    index: int
    id: str | None = None
    function: FunctionCallDelta | None = None
    type: Literal["function"] | None = None


class FunctionDefinition(APIBase):
    """Function tool definition rendered into the chat template."""

    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None
    response: dict[str, Any] | None = None


class Tool(APIBase):
    """Supported tool type. Only OpenAI-style function tools are modeled."""

    function: FunctionDefinition
    type: Literal["function"] = "function"


class ChatMessage(APIBase):
    """Canonical conversation message.

    developer/system/user messages require content and cannot contain assistant
    fields. assistant messages require at least one of content, reasoning, or
    tool_calls. tool messages are tool results and require tool_call_id plus
    content. `reasoning` is the only supported g4b thinking field; incoming
    `reasoning_content` is ignored as an unknown field.
    """

    role: Literal["developer", "system", "user", "assistant", "tool"]
    content: str | list[ContentPart] | None = None
    reasoning: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None

    @model_validator(mode="after")
    def validate_message(self) -> ChatMessage:
        if self.role == "tool":
            if self.tool_call_id is None:
                raise ValueError("tool messages require tool_call_id")
            if self.content is None:
                raise ValueError("tool messages require content")
            if self.reasoning is not None or self.tool_calls is not None:
                raise ValueError("tool messages cannot contain reasoning or tool_calls")
            return self

        if self.role == "assistant":
            if self.content is None and self.reasoning is None and not self.tool_calls:
                raise ValueError("assistant messages require content, reasoning, or tool_calls")
            if self.tool_call_id is not None:
                raise ValueError("assistant messages cannot contain tool_call_id")
            return self

        if self.content is None:
            raise ValueError(f"{self.role} messages require content")
        if self.reasoning is not None or self.tool_calls is not None or self.tool_call_id is not None:
            raise ValueError(f"{self.role} messages cannot contain assistant/tool fields")
        return self


class StreamOptions(APIBase):
    """Streaming options.

    include_usage may request a final zero-usage chunk before [DONE].
    """

    include_usage: bool = False


class ChatCompletionRequest(APIBase):
    """Request body for POST /v1/chat/completions.

    `model` is required and echoed in responses, but does not select a model.
    `messages` is the full stateless conversation. `tool_choice == "none"` means
    no tools should be rendered; every other value is treated like default tool
    behavior. `n` must be 1. Sampling and token-limit parameters are intentionally
    not modeled; if clients send them, APIBase drops them as unknown fields.
    """

    model: str
    messages: list[ChatMessage]
    stream: bool = False
    stream_options: StreamOptions | None = None
    tools: list[Tool] | None = None
    tool_choice: str | dict[str, Any] | None = None
    n: int = 1

    @model_validator(mode="after")
    def validate_request(self) -> ChatCompletionRequest:
        if not self.messages:
            raise ValueError("messages must not be empty")
        if self.n != 1:
            raise ValueError("only n=1 is supported")
        return self


class ChatMessageDelta(APIBase):
    """Partial assistant message used in streamed ChatCompletionChunk objects."""

    role: Literal["assistant"] | None = None
    content: str | None = None
    reasoning: str | None = None
    tool_calls: list[ToolCallDelta] | None = None


class Usage(APIBase):
    """Token usage placeholder.

    g4b currently returns zeros until request/response token accounting is wired.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionChoice(APIBase):
    """Non-streaming choice wrapper."""

    index: int = 0
    message: ChatMessage
    finish_reason: Literal["stop", "length", "tool_calls"] | None = None
    logprobs: None = None


class ChatCompletion(APIBase):
    """Full non-streaming response object."""

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage = Field(default_factory=Usage)


class ChatCompletionChunkChoice(APIBase):
    """Streaming choice wrapper."""

    index: int = 0
    delta: ChatMessageDelta
    finish_reason: Literal["stop", "length", "tool_calls"] | None = None
    logprobs: None = None


class ChatCompletionChunk(APIBase):
    """SSE chunk object for stream=True responses."""

    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionChunkChoice]
    usage: Usage | None = None


class Error(APIBase):
    """OpenAI-style error object."""

    message: str
    type: str = "invalid_request_error"
    param: str | None = None
    code: str | int | None = None


class ErrorResponse(APIBase):
    """OpenAI-style error response envelope."""

    error: Error
