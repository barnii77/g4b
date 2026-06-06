"""
This file implements a fake pytorch stub that provides just the necessary methods triton needs to init to a point
where g4b can register its hooks and take over triton's stream and device management.
"""

from cuda.core import Device


# torch.cuda stub
class cuda:
    @staticmethod
    def is_available():
        return True

    @staticmethod
    def get_device_capability(device: int):
        return Device(device).compute_capability

    @staticmethod
    def current_device():
        raise RuntimeError("must not be called")

    @staticmethod
    def set_device():
        raise RuntimeError("must not be called")


# torch.version stub
class version:
    hip = None
