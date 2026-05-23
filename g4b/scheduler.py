from g4b.tensor import Tensor


class Scheduler:
    token_ids_ring_buffer_TB_uint32: Tensor
    token_ids_ring_buffer_offset_scalar_uint32: Tensor
    context_window_sizes_B_uint32: Tensor  # time dim is dynamically sized
    # TODO additionally this class should maintain a host-side python queue of requests and do scheduler things.
    ...
