from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass

import torch
from torch.cuda import Event, Stream

from .runtime import DeviceWork


@dataclass(frozen=True, slots=True)
class CudaWork:
    event: Event

    def wait_host(self) -> None:
        self.event.synchronize()


class CudaWorkerRuntime:
    """CUDA runtime for EK workers"""

    def __init__(self, device: torch.device) -> None:
        if device.type != "cuda" or device.index is None:
            raise ValueError("CUDA worker runtime requires an indexed CUDA device")

        self.device = device

    def device_context(self) -> AbstractContextManager[None]:
        return torch.cuda.device(self.device)

    def memory_info(self) -> tuple[int, int]:
        return torch.cuda.mem_get_info(self.device)

    def capture_current_work(self) -> DeviceWork:
        with self.device_context():
            stream = self.current_stream()
            event = self.create_event(enable_timing=False)
            self.record_event(event, stream)

        return CudaWork(event)

    def create_stream(self, *, priority: int = 0) -> Stream:
        return Stream(device=self.device, priority=priority)

    def current_stream(self) -> Stream:
        return torch.cuda.current_stream(self.device)

    def synchronize_stream(self, stream: Stream) -> None:
        stream.synchronize()

    def stream_context(self, stream: Stream) -> AbstractContextManager[None]:
        return torch.cuda.stream(stream)

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
        torch.cuda.set_device(self.device)

    def synchronize_device(self) -> None:
        torch.cuda.synchronize(self.device)

    def elapsed_time_ms(self, start: Event, end: Event) -> float:
        return start.elapsed_time(end)
