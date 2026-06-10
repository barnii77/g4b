from g4b.config import Config
from g4b.gguf import GGUFMeta


class Tokenizer:
    def __init__(self, config: Config, meta: GGUFMeta):
        self.eos: int = meta["tokenizer.ggml.eos_token_id"]
        self.bos: int = meta["tokenizer.ggml.bos_token_id"]
        tokens: list[str] = meta["tokenizer.ggml.tokens"]
        self._str_to_tok: dict[str, int] = {tok: i for i, tok in enumerate(tokens)}
        self._tok_to_str = tokens
        self._byte_toks = set(self._str_to_tok[Tokenizer._byte_token(b)] for b in range(256))

    @staticmethod
    def _byte_token(b: int) -> str:
        return f"<0x{b:02X}>"

    def tokenize(self, sequence: str) -> list[int]:
        ...  # TODO efficient algorithm with fancy data structures

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
    # TODO normalize the chat fragments (cat runs of user.prompt and assistant.response.chunk)

    def __init__(self, config: Config, meta: GGUFMeta):
        self._template: str = meta["tokenizer.chat_template"]

    def apply(self, chat_fragments: list[ChatFragment]) -> str:
        ...


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
