import re
import json
import heapq
from abc import ABC, abstractmethod
from typing import Iterator, Sequence
from dataclasses import dataclass
from g4b.gguf import GGUFMeta


class GenEndingTokensProvider(ABC):
    @abstractmethod
    def get(self) -> list[int]: ...


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
        self._gen_ending_tokens = [self.eos, self.end_of_turn]
        self._gen_ending_tokens_provider: GenEndingTokensProvider | None = None

    def gen_ending_tokens(self):
        if self._gen_ending_tokens_provider:
            return self._gen_ending_tokens_provider.get()
        return self._gen_ending_tokens.copy()

    @staticmethod
    def _byte_token(b: int) -> str:
        return f"<0x{b:02X}>"

    def _bpe_merge(self, pieces: Sequence[str]) -> list[str]:
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

    def tokenize(self, sequence: str, *, add_bos: bool = True, allow_special: bool = False) -> list[int]:
        sequence = sequence.replace(" ", "▁")
        out: list[int] = [self.bos] if add_bos else []
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
        return text, tokens[len(tokens) - hold :]


type ChatFragment = PromptFragment | ToolOutput | ResponseFragment | ToolCall


class ChatTemplate:
    def __init__(self, meta: GGUFMeta, tokenizer: Tokenizer):
        # TODO render tokenizer.chat_template instead of hardcoding the Gemma 4 prompt format below
        self._template: str = meta["tokenizer.chat_template"]
        self._tokenizer = tokenizer
        tok_text = lambda text: tokenizer.tokenize(text, add_bos=False, allow_special=False)
        special = tokenizer._special_toks
        self._start_of_turn = special["<|turn>"]
        self._end_of_turn = special["<turn|>"]
        self._start_of_channel = special["<|channel>"]
        self._end_of_channel = special["<channel|>"]
        self._think = special["<|think|>"]
        self._newline = tok_text("\n")
        self._user_turn_open = [self._start_of_turn, *tok_text("user\n")]
        self._thought_channel_open = [self._start_of_channel, *tok_text("thought\n")]
        self._conversation_init = [
            self._start_of_turn,
            *tok_text("system\n"),
            self._think,
            self._end_of_turn,
            *self._newline,
        ]
        self._model_turn_open = [self._start_of_turn, *tok_text("model\n")]
        self._turn_close = [self._end_of_turn, *self._newline]

    def apply(
        self,
        chat_fragments: list[ChatFragment],
        *,
        include_conversation_init: bool = True,
        include_open_turn_to_complete: bool = True,
        add_bos: bool = True,
    ) -> list[int]:
        # TODO should be handled by prompt template
        out = [self._tokenizer.bos] if add_bos else []
        emit = out.extend
        emit_text = lambda text: out.extend(self._tokenizer.tokenize(text, add_bos=False, allow_special=False))
        if include_conversation_init:
            emit(self._conversation_init)

        i = 0
        while i < len(chat_fragments):
            frag = chat_fragments[i]
            if isinstance(frag, PromptFragment):
                emit(self._user_turn_open)
                while i < len(chat_fragments) and isinstance(chat_fragments[i], PromptFragment):
                    emit_text(chat_fragments[i].content)
                    i += 1
                emit(self._turn_close)
                continue
            elif isinstance(frag, ResponseFragment):
                # TODO maybe I shouldn't actually merge responses (thought or normal) that come from separate <|channel>'s
                emit(self._model_turn_open)
                last_channel = None
                while i < len(chat_fragments) and isinstance(chat_fragments[i], ResponseFragment):
                    response = chat_fragments[i]
                    if response.channel == "thought":
                        if last_channel != "thought":
                            emit(self._thought_channel_open)
                        emit_text(response.content)
                    else:
                        if last_channel == "thought":
                            out.append(self._end_of_channel)
                        emit_text(response.content)
                    last_channel = response.channel
                    i += 1
                if last_channel == "thought":
                    out.append(self._end_of_channel)
                emit(self._turn_close)
                continue
            # TODO the two below are probably completely wrong
            elif isinstance(frag, ToolOutput):
                ...  # TODO
            elif isinstance(frag, ToolCall):
                ...  # TODO
            i += 1

        if include_open_turn_to_complete:
            out.extend(self._model_turn_open)

        return out


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
