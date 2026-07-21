"""Plaintext Controller registration and application-heartbeat client."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Protocol
from uuid import uuid4

import grpc
import torch
from expertkit_proto.ek.control.v2 import (
    lifecycle_pb2,
    lifecycle_pb2_grpc,
    weight_control_pb2,
    weight_control_pb2_grpc,
)
from expertkit_proto.ek.worker.v2 import common_pb2
from expertkit_transport.contracts import ACTIVATION_DTYPES

_CONTROL_MESSAGE_BYTES = 1024 * 1024
_UINT32_MAX = (1 << 32) - 1
_UINT64_MAX = (1 << 64) - 1
_DTYPE_TO_PROTO = {
    torch.float16: common_pb2.ACTIVATION_DTYPE_FP16,
    torch.bfloat16: common_pb2.ACTIVATION_DTYPE_BF16,
    torch.float32: common_pb2.ACTIVATION_DTYPE_FP32,
}
_TRANSPORT_TO_PROTO = {
    "grpc": lifecycle_pb2.WORKER_TRANSPORT_GRPC,
    "shm": lifecycle_pb2.WORKER_TRANSPORT_SHM,
}


def new_start_id() -> str:
    """Return a random identifier for one Worker process lifetime."""

    return str(uuid4())


def _require_text(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must not be empty")


def _require_positive_int(name: str, value: int, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= maximum:
        raise ValueError(f"{name} must be a positive integer no larger than {maximum}")


@dataclass(frozen=True, slots=True)
class WorkerRegistration:
    """Describe static information sent once for one Worker process start."""

    worker_id: str
    start_id: str
    instance_id: int
    computation_endpoint: str
    peer_weight_endpoint: str
    backend: str
    activation_dtype: torch.dtype
    device: str
    max_experts: int
    max_batch_tokens: int
    max_active_batches: int
    max_pending_batches: int
    transport_type: str

    def __post_init__(self) -> None:
        for name in (
            "worker_id",
            "start_id",
            "computation_endpoint",
            "peer_weight_endpoint",
            "backend",
            "device",
        ):
            _require_text(name, getattr(self, name))
        _require_positive_int("instance_id", self.instance_id, _UINT64_MAX)
        for name in (
            "max_experts",
            "max_batch_tokens",
            "max_active_batches",
            "max_pending_batches",
        ):
            _require_positive_int(name, getattr(self, name), _UINT32_MAX)
        if self.activation_dtype not in ACTIVATION_DTYPES:
            raise ValueError("activation_dtype must be FP16, BF16, or FP32")
        if self.transport_type not in _TRANSPORT_TO_PROTO:
            raise ValueError("transport_type must be grpc or shm")

    def to_protobuf(self) -> lifecycle_pb2.RegisterWorkerRequest:
        """Encode the v2 registration request."""

        return lifecycle_pb2.RegisterWorkerRequest(
            worker_id=self.worker_id,
            start_id=self.start_id,
            instance_id=self.instance_id,
            computation_endpoint=self.computation_endpoint,
            peer_weight_endpoint=self.peer_weight_endpoint,
            backend=self.backend,
            activation_dtype=_DTYPE_TO_PROTO[self.activation_dtype],
            device=lifecycle_pb2.WorkerDevice(
                device=self.device,
                max_experts=self.max_experts,
            ),
            max_batch_tokens=self.max_batch_tokens,
            max_active_batches_per_device=self.max_active_batches,
            max_pending_batches_per_device=self.max_pending_batches,
            transport_type=_TRANSPORT_TO_PROTO[self.transport_type],
        )


@dataclass(frozen=True, slots=True)
class RegistrationResult:
    """Return Controller versions observed during registration."""

    topology_version: int
    placement_generation: int


class _HeartbeatRpc(Protocol):
    async def send_heartbeats(
        self,
        requests: AsyncIterator[lifecycle_pb2.HeartbeatRequest],
    ) -> lifecycle_pb2.HeartbeatSummary:
        """Send one heartbeat stream until it closes or fails."""


class ControllerConnection:
    """Own the one plaintext gRPC channel shared by both control streams."""

    def __init__(self, endpoint: str) -> None:
        _require_text("Controller endpoint", endpoint)
        self._endpoint = endpoint
        self._channel: grpc.aio.Channel | None = None
        self._lifecycle: Any = None
        self._weight_control: Any = None
        self._start_lock = asyncio.Lock()
        self._close_task: asyncio.Task[None] | None = None
        self._closed = False

    async def start(self) -> None:
        """Create the reusable insecure Controller channel and v2 stubs."""

        async with self._start_lock:
            if self._closed:
                raise RuntimeError("Controller connection is closed")
            if self._channel is not None:
                return
            channel = grpc.aio.insecure_channel(
                self._endpoint,
                options=(
                    ("grpc.max_send_message_length", _CONTROL_MESSAGE_BYTES),
                    ("grpc.max_receive_message_length", _CONTROL_MESSAGE_BYTES),
                ),
            )
            self._channel = channel
            self._lifecycle = lifecycle_pb2_grpc.WorkerLifecycleServiceStub(channel)
            self._weight_control = weight_control_pb2_grpc.WeightControlServiceStub(channel)

    async def register(
        self,
        registration: WorkerRegistration,
        *,
        timeout_secs: float,
    ) -> RegistrationResult:
        """Register idempotently and return the Controller's current versions."""

        if self._lifecycle is None:
            raise RuntimeError("Controller connection has not been started")
        if not isinstance(registration, WorkerRegistration):
            raise TypeError("registration must be a WorkerRegistration")
        if (
            isinstance(timeout_secs, bool)
            or not isinstance(timeout_secs, int | float)
            or timeout_secs <= 0
        ):
            raise ValueError("registration timeout must be positive")
        response = await self._lifecycle.RegisterWorker(
            registration.to_protobuf(),
            timeout=timeout_secs,
            wait_for_ready=False,
        )
        return RegistrationResult(
            topology_version=response.current_topology_version,
            placement_generation=response.current_placement_generation,
        )

    async def send_heartbeats(
        self,
        requests: AsyncIterator[lifecycle_pb2.HeartbeatRequest],
    ) -> lifecycle_pb2.HeartbeatSummary:
        """Open one heartbeat stream on the shared Controller channel."""

        if self._lifecycle is None:
            raise RuntimeError("Controller connection has not been started")
        return await self._lifecycle.Heartbeat(requests, wait_for_ready=False)

    async def sync_weights(
        self,
        requests: AsyncIterator[weight_control_pb2.WorkerWeightMessage],
    ) -> AsyncIterator[weight_control_pb2.ControllerWeightMessage]:
        """Open one weight-control stream on the shared Controller channel."""

        if self._weight_control is None:
            raise RuntimeError("Controller connection has not been started")
        call = self._weight_control.Sync(requests, wait_for_ready=False)
        async for response in call:
            yield response

    async def close(self) -> None:
        """Close the shared Controller channel exactly once."""

        if self._close_task is None:
            self._close_task = asyncio.create_task(
                self._close(),
                name="controller-channel-close",
            )
        await asyncio.shield(self._close_task)

    async def _close(self) -> None:
        self._closed = True
        channel = self._channel
        self._channel = None
        self._lifecycle = None
        self._weight_control = None
        if channel is not None:
            await channel.close()


class HeartbeatSender:
    """Preserve one increasing sequence across heartbeat-stream reconnects."""

    def __init__(
        self,
        *,
        worker_id: str,
        start_id: str,
        interval_secs: float,
    ) -> None:
        _require_text("worker_id", worker_id)
        _require_text("start_id", start_id)
        if (
            isinstance(interval_secs, bool)
            or not isinstance(interval_secs, int | float)
            or interval_secs <= 0
        ):
            raise ValueError("heartbeat interval must be positive")
        self._worker_id = worker_id
        self._start_id = start_id
        self._interval_secs = interval_secs
        self._sequence = 0
        self._state = lifecycle_pb2.WORKER_RUNNING
        self._state_revision = 0
        self._wake = asyncio.Event()
        self._run_lock = asyncio.Lock()
        self._closed = False

    @property
    def last_sequence(self) -> int:
        """Return the last sequence yielded on any heartbeat stream."""

        return self._sequence

    def set_shutting_down(self) -> bool:
        """Switch future heartbeats to SHUTTING_DOWN and wake the stream."""

        if self._state == lifecycle_pb2.WORKER_SHUTTING_DOWN:
            return False
        self._state = lifecycle_pb2.WORKER_SHUTTING_DOWN
        self._state_revision += 1
        self._wake.set()
        return True

    def close(self) -> None:
        """End the current request iterator without resetting its sequence."""

        self._closed = True
        self._wake.set()

    async def run_once(self, rpc: _HeartbeatRpc) -> None:
        """Run one stream attempt; the caller owns registration and reconnect."""

        if self._closed:
            raise RuntimeError("Heartbeat sender is closed")
        async with self._run_lock:
            summary = await rpc.send_heartbeats(self._requests())
            if summary.last_sequence != self._sequence:
                raise RuntimeError(
                    "Controller heartbeat summary did not acknowledge the last sequence"
                )

    async def _requests(self) -> AsyncIterator[lifecycle_pb2.HeartbeatRequest]:
        while not self._closed:
            self._sequence += 1
            sent_revision = self._state_revision
            yield lifecycle_pb2.HeartbeatRequest(
                worker_id=self._worker_id,
                start_id=self._start_id,
                sequence=self._sequence,
                state=self._state,
            )
            if self._closed:
                return
            if self._state_revision != sent_revision:
                continue
            self._wake.clear()
            if self._state_revision != sent_revision:
                continue
            try:
                async with asyncio.timeout(self._interval_secs):
                    await self._wake.wait()
            except TimeoutError:
                pass
