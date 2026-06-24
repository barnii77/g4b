import re
import json
import heapq
from abc import ABC, abstractmethod
from typing import Sequence
from dataclasses import dataclass
from g4b.config import Config
from g4b.gguf import GGUFMeta


class GenEndingTokensProvider(ABC):
    @abstractmethod
    def get(self) -> list[int]: ...


class Tokenizer:
    def __init__(self, config: Config, meta: GGUFMeta):
        self.eos: int = meta["tokenizer.ggml.eos_token_id"]
        self.bos: int = meta["tokenizer.ggml.bos_token_id"]
        tokens: list[str] = meta["tokenizer.ggml.tokens"]
        self._str_to_tok: dict[str, int] = {tok: i for i, tok in enumerate(tokens)}
        self.start_of_turn = self._str_to_tok["<|turn>"]
        self.end_of_turn = self._str_to_tok["<turn|>"]
        self.start_of_channel = self._str_to_tok["<|channel>"]
        self.end_of_channel = self._str_to_tok["<channel|>"]
        self._gen_ending_tokens = [self.eos, self.end_of_turn]
        self._gen_ending_tokens_provider: GenEndingTokensProvider | None = None
        self._tok_to_str = tokens
        self._byte_toks = set(self._str_to_tok[Tokenizer._byte_token(b)] for b in range(256))
        self._special_toks = tuple(
            sorted(
                (
                    tok
                    for tok in tokens
                    if (tok.startswith("<") and tok.endswith(">")) or (tok.startswith("[") and tok.endswith("]"))
                ),
                key=len,
                reverse=True,
            )
        )
        merges: list[str] = meta["tokenizer.ggml.merges"]
        self._merges: dict[tuple[str, str], int] = {}
        for i, merge in enumerate(merges):
            assert merge
            parts = merge[1:].split(' ', maxsplit=1)
            assert len(parts) == 2
            a, b = parts
            a = merge[0] + a
            self._merges[a, b] = i

    def gen_ending_tokens(self):
        if self._gen_ending_tokens_provider:
            return self._gen_ending_tokens_provider.get()
        return self._gen_ending_tokens.copy()

    @staticmethod
    def _byte_token(b: int) -> str:
        return f"<0x{b:02X}>"

    def _bpe_merge(self, pieces: str) -> list[str]:
        pieces = list(pieces)
        while True:
            counts = {}
            for a, b in zip(pieces, pieces[1:]):
                if a + b in self._str_to_tok:
                    counts[(a, b)] = counts.get((a, b), 0) + 1
            if not counts:
                break

            top_a, top_b = max(counts, key=lambda t: counts[t])
            joined = top_a + top_b
            new_pieces = [pieces[0]]
            for b in pieces[1:]:
                a = new_pieces[-1]
                if a == top_a and b == top_b:
                    new_pieces.pop()
                    new_pieces.append(joined)
                else:
                    new_pieces.append(b)
            if new_pieces == pieces:
                break
            pieces = new_pieces
        return pieces

    def tokenize(self, sequence: str, *, add_bos: bool = True) -> list[int]:
        sequence = sequence.replace(" ", "▁")
        out: list[int] = [self.bos] if add_bos else []
        for chunk in self._split_special_tokens(sequence):
            if chunk and set(chunk) == {"\n"} and chunk in self._str_to_tok:
                out.append(self._str_to_tok[chunk])
                continue
            if chunk in self._str_to_tok and chunk in self._special_toks:
                out.append(self._str_to_tok[chunk])
                continue
            for piece in self._bpe_merge(chunk):
                if piece in self._str_to_tok:
                    out.append(self._str_to_tok[piece])
                else:
                    out.extend(self._str_to_tok[Tokenizer._byte_token(b)] for b in piece.encode())
        return out

    def _split_special_tokens(self, sequence: str) -> list[str]:
        # TODO unsafe: this lets user-provided text inject control-sequence tokens.
        #  Chat-template control tokens should be inserted out-of-band, while user text
        #  should be tokenized with special-token matching disabled or sanitized.
        out: list[str] = []
        buf: list[str] = []
        i = 0
        while i < len(sequence):
            special = None
            for tok in self._special_toks:
                if sequence.startswith(tok, i):
                    special = tok
                    break
            if special is not None:
                if buf:
                    out.extend(re.findall(r"[^\n]+|\n+", "".join(buf)))
                    buf.clear()
                out.append(special)
                i += len(special)
            else:
                buf.append(sequence[i])
                i += 1
        if buf:
            out.extend(re.findall(r"[^\n]+|\n+", "".join(buf)))
        return out

    def _tokens_to_bytes(self, tokens: list[int]) -> bytearray:
        out = bytearray()
        for tok in tokens:
            if tok in self._byte_toks:
                out.append(int(self._tok_to_str[tok][1:-1], base=16))
            else:
                out.extend(self._tok_to_str[tok].encode())
        return out

    def detokenize(self, tokens: list[int]) -> str:
        return self._tokens_to_bytes(tokens).decode(errors="replace").replace("▁", " ")

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
            return buf.decode(errors="replace").replace("▁", " "), []
        text = buf[: len(buf) - hold].decode(errors="replace").replace("▁", " ")
        return text, tokens[len(tokens) - hold:]


type ChatFragment = PromptFragment | ToolOutput | ResponseFragment | ToolCall


class ChatTemplate:
    _CONVERSATION_INIT = "<|turn>system\n<|think|><turn|>\n"
    _MODEL_TURN_OPEN = "<|turn>model\n"
    _TURN_CLOSE = "<turn|>\n"

    def __init__(self, config: Config, meta: GGUFMeta):
        self._template: str = meta["tokenizer.chat_template"]

    def apply(self, chat_fragments: list[ChatFragment], *, include_conversation_init: bool = True, include_open_turn_to_complete: bool = True) -> str:
        chat_fragments = _normalize_chat_fragments(chat_fragments)

        # TODO should be handled by prompt template
        out = [self._CONVERSATION_INIT] if include_conversation_init else []

        for frag in chat_fragments:
            if isinstance(frag, PromptFragment):
                out.append(f"<|turn>user\n{frag.content}{self._TURN_CLOSE}")
            elif isinstance(frag, ResponseFragment):
                # Bake here (idempotent for channel=None) so a model turn that is a
                # single un-baked channel run, e.g. a lone thought, keeps its markers.
                out.append(f"{self._MODEL_TURN_OPEN}{_bake_channel_into_content(frag).content}{self._TURN_CLOSE}")
            # TODO the two below are probably completely wrong
            elif isinstance(frag, ToolOutput):
                ...  # TODO
                # out.append(f"<|turn>tool\n{frag.content}<turn|>\n")
            elif isinstance(frag, ToolCall):
                ...  # TODO
                # out.append(f"<|turn>model\n{frag.content}<turn|>\n")

        if include_open_turn_to_complete:
            out.append(self._MODEL_TURN_OPEN)

        return "".join(out)


@dataclass(frozen=True)
class PromptFragment:
    content: str


@dataclass(frozen=True)
class ToolOutput:
    content: str


@dataclass(frozen=True)
class ResponseFragment:
    content: str
    channel: str | None


# TODO possibly this should store some predefined attributes that are definitely required like a name?
@dataclass(frozen=True)
class ToolCall:
    call: dict

    @property
    def content(self) -> str:
        return json.dumps(self.call)  # TODO model expects very different format actually


def _normalize_chat_fragments(frags: list[ChatFragment]) -> list[ChatFragment]:
    if not frags:
        return []

    frags = frags.copy()  # must not mutate the original list

    out = [frags.pop(0)]
    while frags:
        frag = frags.pop(0)
        last_frag = out[-1]
        if isinstance(last_frag, PromptFragment) and isinstance(frag, PromptFragment):  # adjacent user prompts
            out[-1] = PromptFragment(last_frag.content + frag.content)
        elif isinstance(last_frag, ResponseFragment) and isinstance(frag, ResponseFragment):  # adjacent response chunks
            # TODO maybe I shouldn't actually merge responses (thought or normal) that come from separate <|channel>'s
            if last_frag.channel == frag.channel:
                # Same channel: concatenate raw, keep the channel un-baked so a run
                # of e.g. thought chunks bakes into a single <|channel> span (not one per chunk).
                out[-1] = ResponseFragment(last_frag.content + frag.content, last_frag.channel)
            else:
                # Channel boundary within one model turn: bake both so the markers
                # are preserved, collapsing the turn into one channel-less fragment.
                merged = _bake_channel_into_content(last_frag).content + _bake_channel_into_content(frag).content
                out[-1] = ResponseFragment(merged, None)
        else:
            out.append(frag)
    return out


def _bake_channel_into_content(frag: ResponseFragment) -> ResponseFragment:
    if frag.channel == "thought":
        return ResponseFragment(f"<|channel>thought\n{frag.content}<channel|>", None)
    return frag


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
