from g4b.config import Config


class Tokenizer:
    def __init__(self, config: Config):
        ...

    @property
    def eos(self) -> int:
        ...  # TODO return EOS token id

    def tokenize(self, sequence: str) -> list[int]:
        ...

    def detokenize(self, tokens: list[int]) -> str:
        ...


type ChatFragment = PromptFragment | ToolOutput | ResponseFragment | ToolCall


class ChatTemplate:
    # TODO normalize the chat fragments (cat runs of user.prompt and assistant.response.chunk)

    def __init__(self, config: Config):
        # TODO
        ...

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
