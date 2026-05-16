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
