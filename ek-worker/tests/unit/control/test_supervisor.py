"""Tests for Controller registration and stream reconnect supervision."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any

import pytest
import torch

from expertkit_worker.control import (
    ControllerSupervisor,
    RegistrationResult,
    WorkerRegistration,
)


def run(coroutine: Coroutine[Any, Any, Any]) -> Any:
    return asyncio.run(coroutine)


def _registration() -> WorkerRegistration:
    return WorkerRegistration(
        worker_id="worker-0",
        start_id="start-0",
        instance_id=7,
        computation_endpoint="worker-0:50051",
        peer_weight_endpoint="http://worker-0:50052",
        backend="torch",
        activation_dtype=torch.bfloat16,
        device="cuda:0",
        max_experts=8,
        max_batch_tokens=4096,
        max_active_batches=1,
        max_pending_batches=1,
        transport_type="grpc",
    )


class FakeConnection:
    def __init__(self) -> None:
        self.started = False
        self.closed = False
        self.registrations = 0
        self.registered_twice = asyncio.Event()

    async def start(self) -> None:
        self.started = True

    async def register(
        self,
        registration: WorkerRegistration,
        *,
        timeout_secs: float,
    ) -> RegistrationResult:
        assert registration == _registration()
        assert timeout_secs == 2
        self.registrations += 1
        if self.registrations >= 2:
            self.registered_twice.set()
        return RegistrationResult(11, 5)

    async def close(self) -> None:
        self.closed = True


class FakeHeartbeat:
    def __init__(self, *, fail: Exception | None = None) -> None:
        self.calls = 0
        self.shutting_down = False
        self.closed = asyncio.Event()
        self.fail = fail

    async def run_once(self, _connection: object) -> None:
        self.calls += 1
        if self.fail is not None:
            raise self.fail
        if self.calls == 1:
            return
        await self.closed.wait()

    def set_shutting_down(self) -> bool:
        if self.shutting_down:
            return False
        self.shutting_down = True
        return True

    def close(self) -> None:
        self.closed.set()


class FakeWeights:
    def __init__(self) -> None:
        self.calls = 0
        self.shutting_down = False
        self.closed = asyncio.Event()
        self.completion_sent = asyncio.Event()
        self.shutdown_deadline: float | None = None

    async def run_once(self, _connection: object) -> None:
        self.calls += 1
        await self.closed.wait()

    async def begin_shutdown(self) -> bool:
        if self.shutting_down:
            return False
        self.shutting_down = True
        self.shutdown_deadline = 42.0
        return True

    async def wait_shutdown_completion_sent(self) -> None:
        await self.completion_sent.wait()

    async def close(self) -> None:
        self.closed.set()


def _supervisor(
    connection: Any,
    heartbeat: Any,
    weights: Any,
) -> ControllerSupervisor:
    return ControllerSupervisor(
        connection=connection,
        registration=_registration(),
        heartbeat=heartbeat,
        weights=weights,
        registration_timeout_secs=2,
        retry_initial_secs=0.001,
        retry_max_secs=0.002,
        stable_stream_secs=1,
    )


def test_clean_stream_close_registers_again_and_close_stops_supervision() -> None:
    async def scenario() -> None:
        connection = FakeConnection()
        heartbeat = FakeHeartbeat()
        weights = FakeWeights()
        supervisor = _supervisor(connection, heartbeat, weights)
        serving = asyncio.create_task(supervisor.run())

        async with asyncio.timeout(1):
            await connection.registered_twice.wait()
        await supervisor.close()
        await serving

        assert connection.started is True
        assert connection.closed is True
        assert connection.registrations == 2
        assert heartbeat.calls == 2
        assert weights.calls == 2

    run(scenario())


def test_shutdown_updates_both_streams_and_exposes_completion() -> None:
    async def scenario() -> None:
        connection = FakeConnection()
        heartbeat = FakeHeartbeat()
        weights = FakeWeights()
        supervisor = _supervisor(connection, heartbeat, weights)

        assert await supervisor.begin_shutdown() is True
        assert await supervisor.begin_shutdown() is False
        assert heartbeat.shutting_down is True
        assert weights.shutting_down is True
        assert supervisor.shutdown_deadline == 42.0

        waiting = asyncio.create_task(supervisor.wait_shutdown_completion_sent())
        await asyncio.sleep(0)
        assert waiting.done() is False
        weights.completion_sent.set()
        await waiting
        await supervisor.close()

    run(scenario())


def test_non_rpc_stream_failure_is_fatal_instead_of_retried() -> None:
    async def scenario() -> None:
        connection = FakeConnection()
        heartbeat = FakeHeartbeat(fail=ValueError("invalid heartbeat acknowledgement"))
        weights = FakeWeights()
        supervisor = _supervisor(connection, heartbeat, weights)

        with pytest.raises(ValueError, match="invalid heartbeat"):
            await supervisor.run()
        assert connection.registrations == 1
        await supervisor.close()

    run(scenario())
