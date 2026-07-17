"""Tests for Worker process lifecycle and graceful-shutdown ordering."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any

import pytest

from expertkit_worker.app import WorkerApplication, WorkerShutdownTimeout


def run(coroutine: Coroutine[Any, Any, Any]) -> Any:
    return asyncio.run(coroutine)


class AsyncService:
    def __init__(self, name: str, events: list[str]) -> None:
        self.name = name
        self.events = events

    async def start(self) -> None:
        self.events.append(f"start:{self.name}")

    async def close(self) -> None:
        self.events.append(f"close:{self.name}")


class FakeExecution(AsyncService):
    def __init__(self, events: list[str]) -> None:
        super().__init__("execution", events)
        self.closed = asyncio.Event()

    async def wait(self) -> None:
        await self.closed.wait()

    async def close(self) -> None:
        self.events.append("close:execution")
        self.closed.set()


class FakeManager:
    def __init__(self, events: list[str], fatal: BaseException | None = None) -> None:
        self.events = events
        self.fatal = fatal
        self.fatal_ready = asyncio.Event()
        if fatal is not None:
            self.fatal_ready.set()

    def start(self) -> None:
        self.events.append("start:manager")

    async def wait_fatal(self) -> BaseException:
        await self.fatal_ready.wait()
        if self.fatal is None:
            raise RuntimeError("fake fatal event has no error")
        return self.fatal

    async def close(self) -> None:
        self.events.append("close:manager")


class FakeControl:
    def __init__(
        self,
        events: list[str],
        *,
        deadline: float = 100.0,
        complete_on_shutdown: bool = True,
    ) -> None:
        self.events = events
        self.deadline_value = deadline
        self.complete_on_shutdown = complete_on_shutdown
        self.closed = asyncio.Event()
        self.completed = asyncio.Event()
        self.shutdown_deadline: float | None = None

    async def run(self) -> None:
        self.events.append("run:control")
        await self.closed.wait()

    async def begin_shutdown(self) -> bool:
        self.events.append("shutdown:control")
        self.shutdown_deadline = self.deadline_value
        if self.complete_on_shutdown:
            self.completed.set()
        return True

    async def wait_shutdown_completion_sent(self) -> None:
        await self.completed.wait()

    async def close(self) -> None:
        self.events.append("close:control")
        self.closed.set()


class FakeDiskCache:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def close(self) -> None:
        self.events.append("close:disk")


def _application(
    events: list[str],
    *,
    manager: FakeManager | None = None,
    control: FakeControl | None = None,
) -> WorkerApplication:
    return WorkerApplication(
        transfer=AsyncService("transfer", events),
        manager=manager or FakeManager(events),
        peer_server=AsyncService("peer", events),
        computation_server=AsyncService("computation", events),
        execution=FakeExecution(events),
        control=control or FakeControl(events),
        disk_cache=FakeDiskCache(events),
        observability=AsyncService("observability", events),
        clock=lambda: 0.0,
    )


def test_application_starts_then_completes_controller_authorized_shutdown() -> None:
    async def scenario() -> None:
        events: list[str] = []
        application = _application(events)
        serving = asyncio.create_task(application.run())
        await application.started.wait()

        assert application.request_shutdown() is True
        assert application.request_shutdown() is False
        await serving

        assert events[:6] == [
            "start:observability",
            "start:transfer",
            "start:manager",
            "start:peer",
            "start:computation",
            "start:execution",
        ]
        assert "shutdown:control" in events
        assert events[-7:] == [
            "close:control",
            "close:execution",
            "close:peer",
            "close:manager",
            "close:transfer",
            "close:disk",
            "close:observability",
        ]

    run(scenario())


def test_fatal_weight_failure_terminates_and_cleans_up() -> None:
    async def scenario() -> None:
        events: list[str] = []
        fatal = RuntimeError("device placement failed")
        application = _application(events, manager=FakeManager(events, fatal))

        with pytest.raises(RuntimeError, match="device placement failed"):
            await application.run()

        assert events[-2:] == ["close:disk", "close:observability"]
        assert "shutdown:control" not in events

    run(scenario())


def test_expired_shutdown_deadline_forces_cleanup() -> None:
    async def scenario() -> None:
        events: list[str] = []
        control = FakeControl(events, deadline=0.0, complete_on_shutdown=False)
        application = _application(events, control=control)
        serving = asyncio.create_task(application.run())
        await application.started.wait()
        application.request_shutdown()

        with pytest.raises(WorkerShutdownTimeout, match="deadline expired"):
            await serving

        assert events[-2:] == ["close:disk", "close:observability"]

    run(scenario())
