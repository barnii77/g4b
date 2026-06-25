import sys
import unittest

from g4b.protocol import ChatMessage, Tool, FunctionCall, FunctionDefinition, ToolCall
from g4b.tokenizer import ChatTemplate, Tokenizer


def _meta():
    tokens = [
        "<pad>",
        "<eos>",
        "<bos>",
        "<unk>",
        "<mask>",
        "<|turn>",
        "<turn|>",
        "<|channel>",
        "<channel|>",
        "<|think|>",
        "<|tool>",
        "<tool|>",
        '<|"|>',
        "<|tool_call>",
        "<tool_call|>",
        "<|tool_response>",
        "<tool_response|>",
        "<|turn>x",
        "\n",
        "\n\n",
        "▁",
        *(f"<0x{b:02X}>" for b in range(256)),
    ]
    return {
        "tokenizer.ggml.eos_token_id": 1,
        "tokenizer.ggml.bos_token_id": 2,
        "tokenizer.ggml.tokens": tokens,
        # NORMAL=1, UNKNOWN=2, CONTROL=3, USER_DEFINED=4, BYTE=6
        "tokenizer.ggml.token_type": [3, 1, 3, 2, 3, 3, 3, 4, 4, 3, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, *([6] * 256)],
        "tokenizer.ggml.merges": [],
        "tokenizer.chat_template": CHAT_TEMPLATE,
    }


class TokenizerTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = Tokenizer(_meta())

    def test_special_tokens_are_opt_in(self):
        text = "x<|turn>y"

        plain = self.tokenizer.tokenize(text)
        special = self.tokenizer.tokenize(text, allow_special=True)

        self.assertEqual(self.tokenizer.detokenize(plain), text)
        self.assertNotIn(self.tokenizer.start_of_turn, plain)
        self.assertIn(self.tokenizer.start_of_turn, special)

    def test_longest_special_token_wins(self):
        token_id = self.tokenizer._str_to_tok["<|turn>x"]

        result = self.tokenizer.tokenize("<|turn>x", allow_special=True)

        self.assertEqual(result, [token_id])

    def test_byte_token_spelling_is_plain_text(self):
        result = self.tokenizer.tokenize("<0x41>", allow_special=True)

        self.assertNotEqual(result, [self.tokenizer._str_to_tok["<0x41>"]])
        self.assertEqual(self.tokenizer.detokenize(result), "<0x41>")

    def test_chat_template_inserts_markers_after_tokenizing_content(self):
        template = ChatTemplate(_meta(), self.tokenizer)
        result = template.apply([ChatMessage(role="user", content="x<|turn>y")])

        # Conversation init, user turn, and open model turn. The prompt text must
        # not inject a fourth start-of-turn marker.
        self.assertEqual(result.count(self.tokenizer.start_of_turn), 3)

    def test_chat_template_accepts_protocol_messages(self):
        template = ChatTemplate(_meta(), self.tokenizer)
        result = template.apply(
            [
                ChatMessage(role="developer", content="instructions"),
                ChatMessage(role="user", content="hello"),
            ],
            [
                Tool(
                    function=FunctionDefinition(
                        name="lookup",
                        description="Lookup a value <|turn>",
                        parameters={"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
                    )
                )
            ],
        )
        text = self.tokenizer.detokenize(result)

        self.assertIn("<|tool>", text)
        self.assertIn("declaration:lookup", text)
        self.assertIn("<|turn>user\nhello<turn|>", text)
        self.assertEqual(result.count(self.tokenizer.start_of_turn), 3)

    def test_assistant_reasoning_and_content_render_in_history_turn(self):
        template = ChatTemplate(_meta(), self.tokenizer)
        result = template.apply(
            [ChatMessage(role="assistant", reasoning="thinking", content="answer")],
        )

        self.assertEqual(result.count(self.tokenizer.start_of_turn), 3)
        self.assertEqual(result.count(self.tokenizer.end_of_turn), 2)
        self.assertEqual(result.count(self.tokenizer.start_of_channel), 1)
        self.assertEqual(result.count(self.tokenizer.end_of_channel), 1)

    def test_reasoning_field_renders_thought_channel(self):
        template = ChatTemplate(_meta(), self.tokenizer)
        result = template.apply(
            [ChatMessage(role="assistant", reasoning="thinking", content="answer")],
        )
        text = self.tokenizer.detokenize(result)

        self.assertIn("<|channel>thought\nthinking\n<channel|>", text)
        self.assertIn("answer", text)


CHAT_TEMPLATE = """\
{{- bos_token -}}
{%- if enable_thinking or tools or (messages and messages[0]['role'] in ['system', 'developer']) -%}
{{- '<|turn>system\\n' -}}
{%- if enable_thinking -%}
{{- '<|think|>\\n' -}}
{%- endif -%}
{%- if messages and messages[0]['role'] in ['system', 'developer'] -%}
{{- messages[0]['content'] -}}
{%- set messages = messages[1:] -%}
{%- endif -%}
{%- for tool in tools -%}
{{- '<|tool>declaration:' + tool['function']['name'] -}}
{%- if tool['function'].get('description') -%}
{{- '{description:<|"|>' + tool['function']['description'] + '<|"|>}' -}}
{%- endif -%}
{{- '<tool|>' -}}
{%- endfor -%}
{{- '<turn|>\\n' -}}
{%- endif -%}
{%- for message in messages -%}
{%- if message['role'] == 'assistant' -%}
{{- '<|turn>model\\n' -}}
{%- if message.get('reasoning') -%}
{{- '<|channel>thought\\n' + message['reasoning'] + '\\n<channel|>' -}}
{%- endif -%}
{%- if message.get('tool_calls') -%}
{%- for tool_call in message['tool_calls'] -%}
{{- '<|tool_call>call:' + tool_call['function']['name'] + '{' + tool_call['function']['arguments'] + '}<tool_call|>' -}}
{%- endfor -%}
{%- endif -%}
{%- if message.get('content') is string -%}
{{- message['content'] -}}
{%- endif -%}
{{- '<turn|>\\n' -}}
{%- elif message['role'] == 'tool' -%}
{{- '<|tool_response>response:' + message.get('name', 'unknown') + '{value:' + message['content'] + '}<tool_response|>' -}}
{%- else -%}
{{- '<|turn>' + message['role'] + '\\n' -}}
{%- if message.get('content') is string -%}
{{- message['content'] -}}
{%- elif message.get('content') -%}
{%- for part in message['content'] -%}
{%- if part['type'] == 'text' -%}
{{- part['text'] -}}
{%- endif -%}
{%- endfor -%}
{%- endif -%}
{{- '<turn|>\\n' -}}
{%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
{{- '<|turn>model\\n' -}}
{%- endif -%}
"""


if __name__ == "__main__":
    result = unittest.main(exit=False)
    if result.result.wasSuccessful():
        tokenizer = Tokenizer(_meta())
        rendered = ChatTemplate(_meta(), tokenizer).apply(
            [
                ChatMessage(role="developer", content="Follow instructions."),
                ChatMessage(role="user", content=[{"type": "text", "text": "Read the file."}]),
                ChatMessage(
                    role="assistant",
                    reasoning="I should call the file-reading tool.",
                    tool_calls=[
                        ToolCall(
                            id="call_1",
                            function=FunctionCall(name="read_file", arguments='path:<|"|>/tmp/example.txt<|"|>'),
                        )
                    ],
                ),
                ChatMessage(role="tool", tool_call_id="call_1", name="read_file", content="example file contents"),
                ChatMessage(role="assistant", reasoning="The tool returned text.", content="The file says hello."),
                ChatMessage(role="user", content="Thanks. <|turn> should stay text."),
            ],
            [
                Tool(
                    function=FunctionDefinition(
                        name="read_file",
                        description="Read a UTF-8 file from disk.",
                        parameters={
                            "type": "object",
                            "properties": {"path": {"type": "string", "description": "Path to read."}},
                            "required": ["path"],
                        },
                    )
                )
            ],
        )
        print("\nExample rendered chat template:\n", file=sys.stderr)
        print(tokenizer.detokenize(rendered), file=sys.stderr)
    sys.exit(0 if result.result.wasSuccessful() else 1)
