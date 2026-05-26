# TODO after _init_cuda and _init_triton but before cuda graph capture, run triton on fake input tensors so it can
#  autotune first and only then when that is done, record. Also, before recording, make sure to call stream.sync()
#  on all streams.

import importlib.util
import sys
from cuda.core import Device, Stream, Buffer, Event, EventOptions, PinnedMemoryResource

from g4b.utils import runtime_error
from g4b import _torch_stub

device: Device
stream: Stream
_alloc_stream: Stream
_pinned_mr: PinnedMemoryResource

_buffers: list[Buffer] = []
_events: list[Event] = []
_triton_current_stream = None
_triton_current_device = None


def init(device_id: int = 0):
    _init_cuda(device_id)
    _init_triton()


def alloc(size: int) -> Buffer:
    buf = device.allocate(size, stream=_alloc_stream)
    _buffers.append(buf)
    return buf


def alloc_pinned_host(size: int) -> Buffer:
    h_buf = _pinned_mr.allocate(size, stream=_alloc_stream)
    assert h_buf.is_host_accessible
    _buffers.append(h_buf)
    return h_buf


def free(buf: Buffer):
    _buffers.remove(buf)
    buf.close()


def event(timing_enabled: bool = False) -> Event:
    evt = device.create_event(options=EventOptions(timing_enabled=timing_enabled))
    _events.append(evt)
    return evt


def free_event(evt: Event):
    _events.remove(evt)
    evt.close()


def teardown():
    global _triton_current_device, _triton_current_stream
    _triton_current_device = None
    _triton_current_stream = None
    stream.sync()
    for buf in _buffers:
        buf.close()
    _buffers.clear()
    for evt in _events:
        evt.close()
    _events.clear()
    _alloc_stream.sync()
    stream.close()
    _alloc_stream.close()
    _pinned_mr.close()


def _init_cuda(device_id: int):
    global device, stream, _alloc_stream, _pinned_mr
    device = Device(device_id)
    device.set_current()
    stream = device.create_stream()
    _alloc_stream = device.create_stream()
    _pinned_mr = PinnedMemoryResource()


def _init_triton():
    """Monkey patch the triton device and stream getters to avoid a torch dependency."""
    global _triton_current_device, _triton_current_stream

    # install torch stub first so even without pytorch, triton will register an active runtime driver for cuda
    if not _real_torch_available():
        sys.modules["torch"] = _torch_stub

    # now we're all set up to allow evil magic
    import triton.runtime.driver

    triton.runtime.driver.active.get_current_stream = lambda device_id: (
        int(_triton_current_stream.handle)
        if _triton_current_stream.device.device_id == device_id
        else runtime_error("got unexpected device id from triton")
    )
    triton.runtime.driver.active.get_current_device = lambda: int(_triton_current_device.device_id)
    _triton_current_device = device
    _triton_current_stream = stream


def _real_torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None
