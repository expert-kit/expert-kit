from contextlib import AbstractContextManager
from dataclasses import dataclass

import torch
import torch_npu
from torch_npu.npu import Event, Stream

from .runtime import DeviceWork


@dataclass(frozen=True, slots=True)
class AscendWork:
    event: Event

    def wait_host(self) -> None:
        self.event.synchronize()


class AscendWorkerRuntime:
    """Ascend runtime for EK workers"""

    def __init__(self, device: torch.device) -> None:
        if device.type != "npu" or device.index is None:
            raise ValueError("Ascend worker runtime requires an indexed Ascend device")

        self.device = device

    def device_context(self) -> AbstractContextManager[None]:
        return torch_npu.npu.device(self.device)

    def capture_current_work(self) -> DeviceWork:
        with self.device_context():
            stream = self.current_stream()
            event = self.create_event(enable_timing=False)
            self.record_event(event, stream)

        return AscendWork(event)

    def memory_info(self) -> tuple[int, int]:
        return torch_npu.npu.mem_get_info(self.device)

    def create_stream(self, *, priority: int = 0) -> Stream:
        return Stream(device=self.device, priority=priority)

    def current_stream(self) -> Stream:
        return torch_npu.npu.current_stream(self.device)

    def synchronize_stream(self, stream: Stream) -> None:
        stream.synchronize()

    def stream_context(self, stream: Stream) -> AbstractContextManager[None]:
        return torch_npu.npu.stream(stream)

    def create_event(self, *, enable_timing: bool = False) -> Event:
        return Event(enable_timing=enable_timing)

    def record_event(self, event: Event, stream: Stream) -> None:
        event.record(stream)

    def wait_event(self, stream: Stream, event: Event) -> None:
        event.wait(stream)

    def synchronize_event(self, event: Event) -> None:
        event.synchronize()

    def event_done(self, event: Event) -> bool:
        return event.query()

    def set_current_device(self) -> None:
        torch_npu.npu.set_device(self.device)

    def synchronize_device(self) -> None:
        torch_npu.npu.synchronize(self.device)

    def elapsed_time_ms(self, start: Event, end: Event) -> float:
        return start.elapsed_time(end)
