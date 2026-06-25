from g4b.scheduler import Request, Scheduler
from g4b.protocol import ChatMessage
from g4b.tokenizer import Tokenizer, ChatTemplate, GenEndingTokensProvider

PROMPTS = (
    "Hello. Give a short answer.",
    "Write one sentence about CUDA graphs.",
    "List two colors.",
    "What is 2 plus 2?",
    "Name a programming language.",
    "Say something concise.",
    "xqzv blorpt nym nym flandar ##@@@ 918273645 !!!",
    "weird unicode 𐍈 ᚠ 𝛑 plus <not_a_real_control> [[[ ### $$$",
    "Line one.\nLine two with tabs\tand symbols <>[]{}.\nLine three.",
)

GIBBERISH = (
    "xqzv blorpt nym flandar ",
    "9182736450 ##@@!!?? ",
    "lorem_ipsum_but_wrong zzzzz ",
    "<<< user text should not become control >>> ",
    "multi\nline\nnoise ",
)


def submit_generated_interactions(
    scheduler: Scheduler,
    tokenizer: Tokenizer,
    chat_template: ChatTemplate,
    batch_size: int,
    max_prompt_tokens: int,
    max_context_len: int,
):
    for i in range(batch_size):
        prompt = PROMPTS[i % len(PROMPTS)]
        if i % 2 == 0:
            prompt = _make_long_prompt(chat_template, prompt, max_prompt_tokens)
        toks = chat_template.apply([ChatMessage(role="user", content=prompt)])
        if len(toks) < 2:
            toks = [tokenizer.bos, *toks, tokenizer.eos]
        toks = toks[: max(2, max_prompt_tokens)]
        initial_context_len = _seeded_context_len(i, batch_size, max_context_len)
        initial_context_len = max(0, min(initial_context_len, max_context_len - len(toks)))
        scheduler.submit(Request(toks, initial_context_len=initial_context_len))


def _make_long_prompt(chat_template: ChatTemplate, prefix: str, target_tokens: int) -> str:
    prompt = prefix + "\n"
    i = 0
    while True:
        if len(chat_template.apply([ChatMessage(role="user", content=prompt)])) >= target_tokens:
            return prompt
        prompt += GIBBERISH[i % len(GIBBERISH)]
        i += 1


def _seeded_context_len(batch_idx: int, batch_size: int, max_context_len: int) -> int:
    max_context_len = max(0, min(64 * 1024, max_context_len))
    if max_context_len == 0:
        return 0
    if batch_size <= 1:
        return max_context_len
    # Deterministic spread across [0, max_context_len], shuffled a bit so
    # adjacent batch slots do not only see adjacent lengths.
    x = (batch_idx * 1103515245 + 12345) & 0x7FFFFFFF
    frac = batch_idx / (batch_size - 1)
    base = round(frac * max_context_len)
    jitter = x % max(1, max_context_len // max(1, batch_size))
    return min(max_context_len, base + jitter)


class GuaranteePrefillFirstGenTokenDoesNotPreventDecodeWarmupGenEndingTokensProvider(GenEndingTokensProvider):
    """
    During warmup and record, since the prefill phase already predicts one token when it completes, that token
    must not end generation even if it is <eos> or <turn|> because that would prevent decode warmup and crash the
    engine once it attempts to decode for real requests. This class ensures no tokens terminate generation for count
    tokens, then all tokens terminate it.
    """

    def __init__(self, tokenizer: Tokenizer, count: int = 32):
        self.tokenizer = tokenizer
        self.count_left = count

    def get(self) -> list[int]:
        self.count_left -= 1
        all_toks = list(range(len(self.tokenizer._tok_to_str)))
        return [] if self.count_left > 0 else all_toks
