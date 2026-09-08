import os
from contextlib import AbstractContextManager, nullcontext

import torch

from .runtime import DeviceWork


class CpuWork:
    __slots__ = ()

    def wait_host(self) -> None:
        return None


_COMPLETION_WORK = CpuWork()


class CpuWorkerRuntime:
    """CPU runtime for EK workers"""

    def __init__(self, device: torch.device) -> None:
        if device.type != "cpu":
            raise ValueError("CPU worker runtime requires a CPU device")

        self.device = device

    def device_context(self) -> AbstractContextManager[None]:
        return nullcontext()

    def memory_info(self) -> tuple[int, int]:
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            available_pages = os.sysconf("SC_AVPHYS_PAGES")
            total_pages = os.sysconf("SC_PHYS_PAGES")
        except (OSError, ValueError) as error:
            raise RuntimeError("cannot query available CPU memory") from error
        if min(page_size, available_pages, total_pages) <= 0:
            raise RuntimeError("the operating system returned invalid CPU memory information")
        return int(available_pages * page_size), int(total_pages * page_size)

    def capture_current_work(self) -> DeviceWork:
        return _COMPLETION_WORK
