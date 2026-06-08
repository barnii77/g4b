import g4b


class Scheduler:
    # TODO this class should maintain a host-side python queue of requests and do scheduler things.

    def __init__(self, model: "g4b.models.Model"):
        self.model = model
        ...
