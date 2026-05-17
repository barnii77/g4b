import g4b
import re
from pathlib import Path

meta, tensors = g4b.gguf.load(Path("/mnt/C/models/gemma-4-E4B-it-UD-Q4_K_XL.gguf"))
for k, v in meta.items():
    preview = str(v)
    if len(preview) > 100:
        preview = preview[:100] + "..."
    print(k, ":", preview)
for t in tensors:
    print(t)

print("----------")

rm_positional_prefix = re.compile("blk\\.\\d+\\.(.*)")
layer_types = sorted(
    set(
        map(
            lambda p: (p[0].group(1) if p[0] else p[1]) + f' ({p[2]} {"x".join(reversed(list(map(str, p[3]))))})',
            ((rm_positional_prefix.match(t.name), t.name, t.dtype, t.shape) for t in tensors),
        )
    )
)
print("layer types:")
for lt in layer_types:
    print("\t", lt)

required_dtypes = set(t.dtype for t in tensors)
print("required dtypes:")
for dt in required_dtypes:
    print("\t", dt)

print("special tokens:")
tokens_requiring_sanitization = []
for token in meta["tokenizer.ggml.tokens"]:
    if token.startswith('<') and token.endswith('>') or token.startswith('[') and token.endswith(']'):
        print("\t", token)
        continue
    if ('<' in token or '>' in token or '[' in token or ']' in token) and any(c.isalpha() for c in token):
        # This token, if processed by a naive tokenizer, could potentially allow the user to inject meta
        #  tokens like <bos>, <turn|>, etc. because it contains either the starting marker ('<', '[') of a meta token
        #  or the end marker ('>', ']') but is not itself a meta token it seems. E.g. consider '<eos' and '<eos>' exist
        #  as tokens in the vocab. A malicious user could write <eos> into their prompt, then the tokenizer would
        #  BPE merge that into an actual EOS token since '<eos>' exists. If however, neither '<eos' nor 'eos>' exist,
        #  the tokenizer will merge the user's input into '<', 'eos', '>' but since neither '<eos' nor 'eos>' exist,
        #  it will not perform the intermediate merge that would be required as a prerequisite for the fatal
        #  '<' + 'eos>' -> '<eos>' merge. Therefore, if this list stays empty, then as a side effect of how BPE works,
        #  the user will not be able to inject meta tokens and therefore in theory no sanitization logic is needed.
        tokens_requiring_sanitization.append(token)
print("dangerous tokens requiring sanitization logic:")
for token in tokens_requiring_sanitization:
    print("\t", token)
