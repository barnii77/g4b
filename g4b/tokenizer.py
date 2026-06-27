import re
import heapq
import hashlib
import warnings
import ctypes
from abc import ABC, abstractmethod
from typing import Any, Iterator, Sequence
from jinja2 import Template
from pathlib import Path
from functools import cache
from g4b.gguf import GGUFMeta
from g4b import protocol, utils


class GenEndingTokensProvider(ABC):
    @abstractmethod
    def get(self) -> list[int]: ...


# TODO try to load and utilize the native tokenizer if possible. If compilation fails, use current python impl.
#  the native tokenizer exposes a c api which requires calling create() and destroy() for resource management, and all
#  python data must first be serialized into ctypes data in python explicitly to bridge the gap to C++ land.
#  The native impl must include _split_special_tokens, so the entire Tokenizer.tokenize method should be native E2E.
class Tokenizer:
    _NEWLINE_CHUNKS_RE = re.compile(r"[^\n]+|\n+")

    def __init__(self, meta: GGUFMeta):
        self.eos: int = meta["tokenizer.ggml.eos_token_id"]

        tokens: list[str] = meta["tokenizer.ggml.tokens"]
        self._str_to_tok: dict[str, int] = {tok: i for i, tok in enumerate(tokens)}
        self._tok_to_str = tokens
        self._byte_to_tok = [self._str_to_tok[Tokenizer._byte_token(b)] for b in range(256)]
        self._tok_to_byte = {tok: b for b, tok in enumerate(self._byte_to_tok)}

        token_types: list[int] = meta["tokenizer.ggml.token_type"]
        assert len(token_types) == len(tokens)
        self._special_toks: dict[str, int] = {
            tok: tok_id
            for tok_id, (tok, tok_type) in enumerate(zip(tokens, token_types))
            if tok_type in (2, 3, 4)  # UNKNOWN, CONTROL, USER_DEFINED. NORMAL = 1.
        }
        # llama.cpp promotes the configured EOG token to a control token even if
        #  the GGUF happens to label it NORMAL (Gemma 4's <eos> does this).
        self._special_toks[tokens[self.eos]] = self.eos
        self._max_special_tok_len = max(map(len, self._special_toks), default=0)

        merges: list[str] = meta["tokenizer.ggml.merges"]
        self._merges: dict[tuple[str, str], int] = {}
        for i, merge in enumerate(merges):
            assert merge
            parts = merge[1:].split(" ", maxsplit=1)
            assert len(parts) == 2
            a, b = parts
            a = merge[0] + a
            self._merges[a, b] = i

        self.bos: int = meta["tokenizer.ggml.bos_token_id"]
        self.start_of_turn = self._str_to_tok["<|turn>"]
        self.end_of_turn = self._str_to_tok["<turn|>"]
        self.start_of_channel = self._str_to_tok["<|channel>"]
        self.end_of_channel = self._str_to_tok["<channel|>"]
        self.start_of_tool_call = self._str_to_tok.get("<|tool_call>")
        self.end_of_tool_call = self._str_to_tok.get("<tool_call|>")
        self._gen_ending_tokens = [self.eos, self.end_of_turn]
        self._gen_ending_tokens_provider: GenEndingTokensProvider | None = None

        try:
            self._native_tokenizer_dll = _get_native_tokenizer_dll()
        except Exception:
            warnings.warn(
                "Failed to compile and load native tokenizer: switching to python implementation. "
                "This may be noticeably slower."
            )
            self._native_tokenizer_dll = None

    def gen_ending_tokens(self):
        if self._gen_ending_tokens_provider:
            return self._gen_ending_tokens_provider.get()
        return self._gen_ending_tokens.copy()

    @staticmethod
    def _byte_token(b: int) -> str:
        return f"<0x{b:02X}>"

    @staticmethod
    def _with_special_spaces(s: str) -> str:
        return s.replace(" ", "▁")

    @staticmethod
    def _without_special_spaces(s: str) -> str:
        return s.replace("▁", " ")

    def _bpe_merge(self, pieces: str) -> list[str]:
        pieces: list[str | None] = list(pieces)
        merges, links = [], []
        for i, (a, b) in enumerate(zip(pieces, pieces[1:])):
            i_prev = (i - 1) if i != 0 else None
            i_next = (i + 1) if i != len(pieces) - 1 else None
            links.append((i_prev, i_next))
            rank = self._merges.get((a, b))
            if rank is None:
                continue
            merges.append((rank, i, a, b))
        links.append((len(pieces) - 2, None))
        heapq.heapify(merges)

        def ll_unlink(i: int):
            i_prev, i_next = links[i]
            _, i_next_next = links[i_next] if i_next is not None else (None, None)
            i_prev_prev, _ = links[i_prev] if i_prev is not None else (None, None)
            links[i] = None, None
            if i_next is not None:
                links[i_next] = i_prev, i_next_next
            if i_prev is not None:
                links[i_prev] = i_prev_prev, i_next

        def try_add_merge(i: int):
            _, i_next = links[i]
            if i_next is None:
                return

            a, b = pieces[i], pieces[i_next]
            assert a is not None and b is not None

            rank = self._merges.get((a, b))
            if rank is None:
                return
            heapq.heappush(merges, (rank, i, a, b))

        while merges:
            _, pos, a_exp, b_exp = heapq.heappop(merges)
            assert a_exp is not None and b_exp is not None
            pos_prev, pos_next = links[pos]
            if pos_next is None:
                continue

            a, b = pieces[pos], pieces[pos_next]
            if a != a_exp or b != b_exp:
                continue  # stale merge

            pieces[pos] = a + b
            pieces[pos_next] = None
            ll_unlink(pos_next)
            if pos_prev is not None:
                try_add_merge(pos_prev)
            try_add_merge(pos)

        return [x for x in pieces if x is not None]

    def tokenize(self, sequence: str, *, allow_special: bool = False) -> list[int]:
        sequence = Tokenizer._with_special_spaces(sequence)
        out: list[int] = []
        chunks = self._split_special_tokens(sequence) if allow_special else self._split_newline_chunks(sequence)
        for chunk in chunks:
            if isinstance(chunk, int):
                out.append(chunk)
                continue
            chunk_tok = self._str_to_tok.get(chunk)
            if chunk[0] == "\n" and chunk_tok is not None:
                out.append(chunk_tok)
                continue
            for piece in self._bpe_merge(chunk):
                piece_tok = self._str_to_tok.get(piece)
                if piece_tok is not None:
                    out.append(piece_tok)
                else:
                    out.extend(self._byte_to_tok[b] for b in piece.encode())
        return out

    def split_special_tokens(self, sequence: str) -> Iterator[str | int]:
        return self._split_special_tokens(Tokenizer._with_special_spaces(sequence))

    @classmethod
    def _split_newline_chunks(cls, sequence: str) -> Iterator[str]:
        return iter(cls._NEWLINE_CHUNKS_RE.findall(sequence))

    def _split_special_tokens(self, sequence: str) -> Iterator[str | int]:
        text_start = 0
        i = 0
        while i < len(sequence):
            special_id = None
            special_len = min(self._max_special_tok_len, len(sequence) - i)
            while special_len:
                special_id = self._special_toks.get(sequence[i : i + special_len])
                if special_id is not None:
                    break
                special_len -= 1

            if special_id is not None:
                if text_start < i:
                    yield from self._split_newline_chunks(sequence[text_start:i])
                yield special_id
                i += special_len
                text_start = i
            else:
                i += 1
        if text_start < len(sequence):
            yield from self._split_newline_chunks(sequence[text_start:])

    def _tokens_to_bytes(self, tokens: list[int]) -> bytearray:
        out = bytearray()
        for tok in tokens:
            byte = self._tok_to_byte.get(tok)
            if byte is not None:
                out.append(byte)
            else:
                out.extend(self._tok_to_str[tok].encode())
        return out

    def detokenize(self, tokens: list[int]) -> str:
        return Tokenizer._without_special_spaces(self._tokens_to_bytes(tokens).decode(errors="replace"))

    def detokenize_streaming(self, tokens: list[int], *, flush: bool) -> tuple[str, list[int]]:
        """
        Detokenize for incremental streaming. When flush is False and `tokens`
        ends part-way through a multi-byte UTF-8 character, the trailing tokens
        making up that partial character are held back (rather than decoded into
        U+FFFD) and returned so the caller can prepend them to the next call.

        Returns (text, held_tokens). An incomplete UTF-8 tail can only be formed
        by single-byte byte-tokens, so the byte holdback maps to a token holdback
        one-to-one.
        """
        buf = self._tokens_to_bytes(tokens)
        hold = 0 if flush else _utf8_incomplete_tail_len(buf)
        if not hold:
            return Tokenizer._without_special_spaces(buf.decode(errors="replace")), []
        text = Tokenizer._without_special_spaces(buf[: len(buf) - hold].decode(errors="replace"))
        return text, tokens[len(tokens) - hold :]


class ChatTemplate:
    def __init__(self, meta: GGUFMeta, tokenizer: Tokenizer):
        self._template: str = meta["tokenizer.chat_template"]
        self._compiled_template = Template(self._template)
        self._template_hash = hashlib.sha512(self._template.encode()).hexdigest()
        self._tokenizer = tokenizer

    def apply(self, messages: list[protocol.ChatMessage], tools: list[protocol.Tool] | None = None) -> list[int]:
        placeholder_registry = _PlaceholderRegistry(self._template_hash)
        rendered = self._compiled_template.render(
            messages=[self._message_to_template_dict(m, placeholder_registry) for m in messages],
            tools=[self._tool_to_template_dict(tool, placeholder_registry) for tool in tools or []],
            bos_token=self._tokenizer._tok_to_str[self._tokenizer.bos],
            add_generation_prompt=True,
            enable_thinking=True,
        )
        mixed = self._tokenize_privileged(rendered)
        return self._tokenize_mixed_unprivileged(placeholder_registry.replace_in_mixed(mixed))

    def _message_to_template_dict(
        self, message: protocol.ChatMessage, placeholder_registry: "_PlaceholderRegistry"
    ) -> dict[str, Any]:
        out = message.model_dump(exclude_none=True)
        self._placeholder_message_strings(out, placeholder_registry)
        return out

    def _placeholder_message_strings(self, message: dict[str, Any], placeholder_registry: "_PlaceholderRegistry"):
        content = message.get("content")
        if isinstance(content, str):
            message["content"] = placeholder_registry.add(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                    part["text"] = placeholder_registry.add(part["text"])

        if isinstance(message.get("reasoning"), str):
            message["reasoning"] = placeholder_registry.add(message["reasoning"])

        for tool_call in message.get("tool_calls") or []:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            if not isinstance(function, dict):
                continue
            if isinstance(function.get("name"), str):
                function["name"] = placeholder_registry.add(function["name"])
            if isinstance(function.get("arguments"), str):
                function["arguments"] = placeholder_registry.add(function["arguments"])
            elif isinstance(function.get("arguments"), dict):
                function["arguments"] = self._placeholder_schema_strings(function["arguments"], placeholder_registry)

    def _tool_to_template_dict(
        self, tool: protocol.Tool, placeholder_registry: "_PlaceholderRegistry"
    ) -> dict[str, Any]:
        out = tool.model_dump(exclude_none=True)
        function = out.get("function")
        if isinstance(function, dict):
            if isinstance(function.get("name"), str):
                function["name"] = placeholder_registry.add(function["name"])
            if isinstance(function.get("description"), str):
                function["description"] = placeholder_registry.add(function["description"])
            if isinstance(function.get("parameters"), dict):
                function["parameters"] = self._placeholder_schema_strings(function["parameters"], placeholder_registry)
            response = function.get("response")
            if isinstance(response, dict):
                function["response"] = self._placeholder_schema_strings(response, placeholder_registry)
        return out

    def _placeholder_schema_strings(self, value: Any, placeholder_registry: "_PlaceholderRegistry", key: str = ""):
        if isinstance(value, str):
            return value if key == "type" else placeholder_registry.add(value)
        if isinstance(value, list):
            return [self._placeholder_schema_strings(item, placeholder_registry, key) for item in value]
        if isinstance(value, dict):
            return {
                item_key: self._placeholder_schema_strings(item_value, placeholder_registry, item_key)
                for item_key, item_value in value.items()
            }
        return value

    def _tokenize_privileged(self, text: str) -> list[int | str]:
        return list(self._tokenizer.split_special_tokens(text))

    def _tokenize_mixed_unprivileged(self, mixed: list[int | str]) -> list[int]:
        out: list[int] = []
        for item in mixed:
            if isinstance(item, int):
                out.append(item)
            else:
                out.extend(self._tokenizer.tokenize(item, allow_special=False))
        return out


class _PlaceholderRegistry:
    def __init__(self, template_hash: str):
        self._prefix = f"{template_hash}-"
        self._values_by_hash: dict[str, str] = {}

    def add(self, value: str) -> str:
        value_hash = hashlib.sha512(value.encode()).hexdigest()
        previous = self._values_by_hash.setdefault(value_hash, value)
        if previous != value:
            raise AssertionError("sha512 collision while building chat-template placeholders")
        return f"{self._prefix}{value_hash}-"

    def replace_in_mixed(self, mixed: list[int | str]) -> list[int | str]:
        return [self._replace_in_text(item) if isinstance(item, str) else item for item in mixed]

    def _replace_in_text(self, text: str) -> str:
        out: list[str] = []
        i = 0
        while True:
            j = text.find(self._prefix, i)
            if j == -1:
                out.append(text[i:])
                return "".join(out)

            value_hash_start = j + len(self._prefix)
            value_hash_end = value_hash_start + 128
            if value_hash_end >= len(text) or text[value_hash_end] != "-":
                raise AssertionError("template hash appeared outside a valid placeholder")

            value_hash = text[value_hash_start:value_hash_end]
            value = self._values_by_hash.get(value_hash)
            if value is None:
                raise AssertionError("chat-template placeholder referenced an unknown value hash")

            out.append(text[i:j])
            out.append(value)
            i = value_hash_end + 1


def _utf8_incomplete_tail_len(b: bytes | bytearray) -> int:
    """
    Number of trailing bytes that form the start of a not-yet-complete UTF-8
    character (0 if the buffer ends on a character boundary).
    """
    n = len(b)
    for back in range(1, min(4, n) + 1):
        byte = b[n - back]
        if byte & 0xC0 == 0x80:
            continue  # continuation byte; keep scanning back for the lead byte
        if byte < 0x80:
            seq = 1
        elif byte & 0xE0 == 0xC0:
            seq = 2
        elif byte & 0xF0 == 0xE0:
            seq = 3
        elif byte & 0xF8 == 0xF0:
            seq = 4
        else:
            return 0  # invalid lead byte; let errors="replace" deal with it
        return back if back < seq else 0
    return 0  # only continuation bytes seen (malformed); don't hold anything back


@cache
def _get_native_tokenizer_dll() -> ctypes.CDLL:
    return utils.compile_and_load_cpp(Path(__file__).parent / "tokenizer.cpp")
