import unittest

from pydantic import ValidationError

from g4b.protocol import (
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionRequest,
    ChatMessageDelta,
    ToolCallDelta,
)


class ProtocolTest(unittest.TestCase):
    def test_request_drops_unknown_fields_and_keeps_reasoning(self):
        request = ChatCompletionRequest.model_validate(
            {
                "model": "g4b",
                "messages": [
                    {"role": "user", "content": "hello", "ignored": True},
                    {
                        "role": "assistant",
                        "reasoning": "thinking",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "lookup", "arguments": {"q": "x"}},
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call_1", "content": "result"},
                ],
                "reasoning_content": "ignored",
                "max_tokens": 128,
                "max_completion_tokens": 128,
                "temperature": 0.7,
                "top_p": 0.9,
                "presence_penalty": 0.1,
                "frequency_penalty": 0.1,
            }
        )

        self.assertEqual(request.messages[1].reasoning, "thinking")
        dumped = request.model_dump(exclude_none=True)
        self.assertEqual(dumped["model"], "g4b")
        self.assertNotIn("max_tokens", dumped)
        self.assertNotIn("max_completion_tokens", dumped)
        self.assertNotIn("temperature", dumped)
        self.assertNotIn("top_p", dumped)
        self.assertNotIn("presence_penalty", dumped)
        self.assertNotIn("frequency_penalty", dumped)
        self.assertNotIn("ignored", dumped["messages"][0])
        self.assertNotIn("reasoning_content", dumped)

    def test_tool_messages_require_tool_call_id(self):
        with self.assertRaises(ValidationError):
            ChatCompletionRequest.model_validate(
                {
                    "model": "g4b",
                    "messages": [{"role": "tool", "content": "result"}],
                }
            )

    def test_only_single_choice_requests_are_supported(self):
        with self.assertRaises(ValidationError):
            ChatCompletionRequest.model_validate(
                {
                    "model": "g4b",
                    "messages": [{"role": "user", "content": "hello"}],
                    "n": 2,
                }
            )

    def test_streamed_tool_call_arguments_can_be_partial(self):
        chunk = ChatCompletionChunk(
            id="chatcmpl-test",
            created=1,
            model="g4b",
            choices=[
                ChatCompletionChunkChoice(
                    delta=ChatMessageDelta(tool_calls=[ToolCallDelta(index=0, function={"arguments": "{"})])
                )
            ],
        )

        tool_call = chunk.choices[0].delta.tool_calls[0]
        self.assertEqual(tool_call.function.arguments, "{")


if __name__ == "__main__":
    unittest.main()
