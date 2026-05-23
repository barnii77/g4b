class Scheduler:
    # TODO should contain an (max_n_users,) uint32 tensor which stores how long the context window
    #  is which each user is using respectively. It will be updated by the relevant model impl.
    #  It should also contain a (max_n_users, T)-sized ring buffer uint32 output tokens tensor where
    #  the model impl's sampling kernel writes its outputs (as token ids) to.
    # TODO additionally this class will maintain a host-side python queue of requests and do scheduler things.
    ...
