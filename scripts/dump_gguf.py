import g4b
from pathlib import Path

meta, tensors = g4b.gguf.load(Path("/mnt/C/models/gemma-4-E2B-it-UD-Q4_K_XL.gguf"))
for k, v in meta.items():
    preview = str(v)
    if len(preview) > 100:
        preview = preview[:100] + '...'
    print(k, ':', preview)
for t in tensors:
    print(t)
