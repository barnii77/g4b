# TODO need an extra boolean (B,)-sized tensor that specifies whether each user is currently in prefill or decode. Then
#  I have to use this information in the relevant prefill or decode FA kernels to efficiently partition work.
# TODO foreach KV group I should load all corresponding queries in a single thread-block so I can get a
#  ((g*tile_T1) x d_k) @ (d_k x tile_T2) tl.dot operation -> better arithmetic intensity than loading KV tiles for each
#  query separately.
# TODO flash decode, i.e. tile across not only Q time dim but also with different tile size across KV time dim and have
#  it write into temp buffers and have a second kernel reduce the partial results.
# TODO I will also likely need to come up with fancy work partitioning strategies involving persistent kernels and
#  blackwell-style work-stealing-ish dynamically-scheduled work assignment to deal with the highly heterogeneous nature
#  of the decode phase especially but also prefill, since the context length between users differs greatly, likely
#  following a long-tailed distribution.
# TODO since this kernel is probably the one which deals with the largest indices, I'll have to consider doing some
#  indexing computations in int64 explicitly instead of the implicit int32 default that you get with naive triton.
#  This does however come with a performance penalty, so maybe gate it conditionally based on input sizes. A O(GB) KV
#  cache is pretty common after all.
# TODO one could potentially try introducing an extra reduction loop across the head dim (innermost dim), though that
#  would increase flash attn memory traffic by ~50% so probably bad unless it massively boosts MMA throughput because
#  of better tile shapes. May be interesting though for gemma 4 specifically because of the huge 512 head dim, which
#  implies small M_tile and N_tile due to SMEM constraints.
