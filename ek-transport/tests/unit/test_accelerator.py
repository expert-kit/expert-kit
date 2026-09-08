"""Tests for transport-local accelerator stream and event dispatch."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from expertkit_transport._accelerator import accelerator_for


class FakeStream:
    def __init__(self) -> None:
        self.waited: FakeEvent | None = None

    def wait_event(self, event: FakeEvent) -> None:
        self.waited = event


class FakeEvent:
    def __init__(self, *, enable_timing: bool, blocking: bool) -> None:
        assert enable_timing is False
        assert blocking is False
        self.recorded: FakeStream | None = None
        self.synchronized = False

    def record(self, stream: FakeStream) -> None:
        self.recorded = stream

    def synchronize(self) -> None:
        self.synchronized = True


def test_cpu_has_no_accelerator_adapter() -> None:
    assert accelerator_for("cpu") is None


def test_adapter_dispatches_through_the_device_module(monkeypatch) -> None:
    stream = FakeStream()
    selected: list[tuple[FakeStream, torch.device]] = []
    module = SimpleNamespace(
        current_stream=lambda device: stream,
        Event=FakeEvent,
        stream=lambda value: (
            selected.append((value, torch.device("cuda:3"))),
            nullcontext(),
        )[1],
    )
    monkeypatch.setattr(torch, "get_device_module", lambda device: module)

    accelerator = accelerator_for("cuda:3")
    assert accelerator is not None
    current = accelerator.current_stream()
    event = accelerator.create_event()
    event.record(current)
    current.wait_event(event)
    with accelerator.stream_context(current):
        pass
    event.synchronize()

    assert event.recorded is current
    assert current.waited is event
    assert event.synchronized is True
    assert selected == [(stream, torch.device("cuda:3"))]


def test_adapter_rejects_non_transport_device() -> None:
    with pytest.raises(ValueError, match="CPU, CUDA, or NPU"):
        accelerator_for("meta")
