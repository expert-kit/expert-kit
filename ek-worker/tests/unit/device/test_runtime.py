"""Hardware-independent tests for worker device runtime behavior."""

from contextlib import nullcontext

import pytest
import torch

from expertkit_worker.device import CpuWorkerRuntime, CudaWorkerRuntime


def test_cpu_runtime_exposes_completed_work_and_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    values = {
        "SC_PAGE_SIZE": 4096,
        "SC_AVPHYS_PAGES": 100,
        "SC_PHYS_PAGES": 250,
    }
    monkeypatch.setattr(
        "expertkit_worker.device.cpu.os.sysconf",
        values.__getitem__,
    )
    runtime = CpuWorkerRuntime(torch.device("cpu"))

    with runtime.device_context():
        pass

    assert runtime.memory_info() == (409_600, 1_024_000)
    first = runtime.capture_current_work()
    second = runtime.capture_current_work()
    assert first is second
    first.wait_host()


def test_cpu_runtime_rejects_non_cpu_device() -> None:
    with pytest.raises(ValueError, match="requires a CPU device"):
        CpuWorkerRuntime(torch.device("cuda:0"))


def test_cpu_runtime_rejects_invalid_memory_information(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "expertkit_worker.device.cpu.os.sysconf",
        lambda _name: 0,
    )
    runtime = CpuWorkerRuntime(torch.device("cpu"))

    with pytest.raises(RuntimeError, match="invalid CPU memory information"):
        runtime.memory_info()


class _Stream:
    def __init__(self) -> None:
        self.synchronized = False

    def synchronize(self) -> None:
        self.synchronized = True


def test_cuda_runtime_captures_current_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = CudaWorkerRuntime(torch.device("cuda:3"))
    stream = _Stream()
    monkeypatch.setattr(runtime, "device_context", nullcontext)
    monkeypatch.setattr(runtime, "current_stream", lambda: stream)

    work = runtime.capture_current_work()

    assert stream.synchronized is False
    work.wait_host()
    assert stream.synchronized is True
