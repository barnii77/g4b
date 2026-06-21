import re
import os
from abc import ABC, abstractmethod
from g4b import lifecycle
from g4b.config import Config
from g4b.gguf import GGUFMeta
from g4b.utils import create_file_logger

_log_all_prompts_path = os.environ.get("G4B_LOG_ALL_PROMPTS_PATH")
_prompts_logger = create_file_logger(_log_all_prompts_path) if _log_all_prompts_path else None


class GenEndingTokensProvider(ABC):
    @abstractmethod
    def get(self) -> list[int]: ...


class Tokenizer:
    def __init__(self, config: Config, meta: GGUFMeta):
        self.eos: int = meta["tokenizer.ggml.eos_token_id"]
        self.bos: int = meta["tokenizer.ggml.bos_token_id"]
        tokens: list[str] = meta["tokenizer.ggml.tokens"]
        self._str_to_tok: dict[str, int] = {tok: i for i, tok in enumerate(tokens)}
        self.end_of_turn = self._str_to_tok["<turn|>"]
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

    def gen_ending_tokens(self):
        if self._gen_ending_tokens_provider:
            return self._gen_ending_tokens_provider.get()
        return self._gen_ending_tokens.copy()

    @staticmethod
    def _byte_token(b: int) -> str:
        return f"<0x{b:02X}>"

    def _bpe_merge(self, pieces: list[str]) -> list[str]:
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

    def tokenize(self, sequence: str) -> list[int]:
        sequence = sequence.replace(" ", "▁")
        out: list[int] = [self.bos]
        for chunk in self._split_special_tokens(sequence):
            if chunk and set(chunk) == {"\n"} and chunk in self._str_to_tok:
                out.append(self._str_to_tok[chunk])
                continue
            if chunk in self._str_to_tok and chunk in self._special_toks:
                out.append(self._str_to_tok[chunk])
                continue
            for piece in self._bpe_merge(list(chunk)):
                if piece in self._str_to_tok:
                    out.append(self._str_to_tok[piece])
                else:
                    out.extend(self._str_to_tok[Tokenizer._byte_token(b)] for b in piece.encode())
        return out

    def _split_special_tokens(self, sequence: str) -> list[str]:
        # TODO unsafe: this lets user-provided text inject control-sequence tokens.
        # Chat-template control tokens should be inserted out-of-band, while user text
        # should be tokenized with special-token matching disabled or sanitized.
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

    def detokenize(self, tokens: list[int]) -> str:
        out = bytearray()
        for tok in tokens:
            if tok in self._byte_toks:
                out.append(int(self._tok_to_str[tok][1:-1], base=16))
            else:
                out.extend(self._tok_to_str[tok].encode())
        return out.decode(errors="replace").replace("▁", " ")


type ChatFragment = PromptFragment | ToolOutput | ResponseFragment | ToolCall


class ChatTemplate:
    def __init__(self, config: Config, meta: GGUFMeta):
        self._template: str = meta["tokenizer.chat_template"]

    def apply(self, chat_fragments: list[ChatFragment]) -> str:
        # Gemma chat templates use these literal control strings; the tokenizer
        # BPE pass merges them to the corresponding control tokens.
        chat_fragments = chat_fragments  # TODO normalize
        out = ["<|turn>system\n<|think|><turn|>\n"]
        for frag in chat_fragments:
            if isinstance(frag, PromptFragment):
                out.append(f"<|turn>user\n{frag.content}<turn|>\n")
            elif isinstance(frag, ToolOutput):
                out.append(f"<|turn>tool\n{frag.content}<turn|>\n")
            elif isinstance(frag, ResponseFragment):
                out.append(f"<|turn>model\n{frag.content}<turn|>\n")
            elif isinstance(frag, ToolCall):
                out.append(f"<|turn>model\n{frag.call}<turn|>\n")
        out.append("<|turn>model\n")
        inp = "".join(out)
        if _prompts_logger and lifecycle.is_deployment():
            _prompts_logger.info(inp)
        return inp


class PromptFragment:
    def __init__(self, content: str):
        self.content = content


class ToolOutput:
    def __init__(self, content: str):
        self.content = content


class ResponseFragment:
    def __init__(self, content: str):
        self.content = content


# TODO possibly this should store some predefined attributes that are definitely required like a name?
class ToolCall:
    def __init__(self, call: dict):
        self.call = call


def _normalize_chat_fragments(frags: list[ChatFragment]) -> list[ChatFragment]:
    out = [frags.pop(0)]
    while frags:
        frag = frags.pop()
        last_frag = out[-1]
        if isinstance(last_frag, PromptFragment) and isinstance(frag, PromptFragment):
            out[-1] = PromptFragment(last_frag.content + frag.content)
        elif isinstance(last_frag, ResponseFragment) and isinstance(frag, ResponseFragment):
            out[-1] = ResponseFragment(last_frag.content + frag.content)
    return out
