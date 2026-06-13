# TODO foreach KV group I should load all corresponding queries in a single thread-block so I can get a
#  (g x d_k) @ (d_k x tile_T) tl.dot operation  -> much better arithmetic intensity than loading KV tiles for each
#  query separately.
# TODO parallelism over T
# TODO try to make sure the tile layouts produced by triton minimize cross-warp reductions
# TODO make sure I only do .max() and .sum() *reduces* outside of the loop across the T dim
# TODO need an extra boolean (B,)-sized tensor that specifies whether each user is currently in prefill or decode. Then
#  I have to use this information in the relevant prefill or decode FA kernels to efficiently partition work.
# TODO I will also likely need to come up with fancy work partitioning strategies involving persistent kernels and
#  blackwell-style work-stealing-ish dynamically-scheduled work assignment to deal with the highly heterogeneous nature
#  of the decode phase especially but also prefill, since the context length between users differs greatly, likely
#  following a long-tailed distribution.
# TODO I need a temporary KV buffer for when the SWA window size is < chunked prefill size, from which I then memcpy to
#  the actual KV cache (which is only 512 tokens long for SWA layers)
# TODO since this kernel is probably the one which deals with the largest indices, I'll have to consider doing some
#  indexing computations in int64 explicitly instead of the implicit int32 default that you get with naive triton.
#  This does however come with a performance penalty, so maybe gate it conditionally based on input sizes. A O(GB) KV
#  cache is pretty common after all.