import unittest

from g4b.tokenizer import ChatTemplate, PromptFragment, ResponseFragment, Tokenizer


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
        "tokenizer.ggml.token_type": [3, 1, 3, 2, 3, 3, 3, 4, 4, 3, 4, 1, 1, 1, *([6] * 256)],
        "tokenizer.ggml.merges": [],
        "tokenizer.chat_template": "unused test template",
    }


class TokenizerTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = Tokenizer(_meta())

    def test_special_tokens_are_opt_in(self):
        text = "x<|turn>y"

        plain = self.tokenizer.tokenize(text, add_bos=False)
        special = self.tokenizer.tokenize(text, add_bos=False, allow_special=True)

        self.assertEqual(self.tokenizer.detokenize(plain), text)
        self.assertNotIn(self.tokenizer.start_of_turn, plain)
        self.assertIn(self.tokenizer.start_of_turn, special)

    def test_longest_special_token_wins(self):
        token_id = self.tokenizer._str_to_tok["<|turn>x"]

        result = self.tokenizer.tokenize("<|turn>x", add_bos=False, allow_special=True)

        self.assertEqual(result, [token_id])

    def test_byte_token_spelling_is_plain_text(self):
        result = self.tokenizer.tokenize("<0x41>", add_bos=False, allow_special=True)

        self.assertNotEqual(result, [self.tokenizer._str_to_tok["<0x41>"]])
        self.assertEqual(self.tokenizer.detokenize(result), "<0x41>")

    def test_chat_template_inserts_markers_after_tokenizing_content(self):
        template = ChatTemplate(_meta(), self.tokenizer)
        result = template.apply([PromptFragment("x<|turn>y")])

        # Conversation init, user turn, and open model turn. The prompt text must
        # not inject a fourth start-of-turn marker.
        self.assertEqual(result.count(self.tokenizer.start_of_turn), 3)

    def test_adjacent_response_channels_remain_one_model_turn(self):
        template = ChatTemplate(_meta(), self.tokenizer)
        result = template.apply(
            [ResponseFragment("thinking", "thought"), ResponseFragment("answer", None)],
            include_conversation_init=False,
            include_open_turn_to_complete=False,
            add_bos=False,
        )

        self.assertEqual(result.count(self.tokenizer.start_of_turn), 1)
        self.assertEqual(result.count(self.tokenizer.end_of_turn), 1)
        self.assertEqual(result.count(self.tokenizer.start_of_channel), 1)
        self.assertEqual(result.count(self.tokenizer.end_of_channel), 1)


if __name__ == "__main__":
    unittest.main()
