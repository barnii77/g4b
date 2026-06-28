# G4B

A Gemma 4 E4B inference engine, from scratch, using triton and cuda.

Originally built as part of a school project.
Therefore, a lot of shortcuts were taken during development, e.g. dequant to full fp16 instead of int8 during prefill,
questionable structure of `g4b/kernels/matmul.py`, the hacky model impl <-> scheduler boundary, the python tokenizer,
and more.

There are still many severe performance issues left. I'll try to make g4b fast soon.
Let's see how far I get before I lose interest ;)

For an overview of how it works... ask Codex!
