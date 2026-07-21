"""Tests for Controller registration and heartbeat sequencing."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Coroutine
from dataclasses import replace
from typing import Any

import grpc
import pytest
import torch
from expertkit_proto.ek.control.v2 import (
    lifecycle_pb2,
    lifecycle_pb2_grpc,
    weight_control_pb2,
    weight_control_pb2_grpc,
)
from expertkit_proto.ek.worker.v2 import common_pb2

from expertkit_worker.control import (
    ControllerConnection,
    HeartbeatSender,
    WorkerRegistration,
    new_start_id,
)


def run(coroutine: Coroutine[Any, Any, Any]) -> Any:
    return asyncio.run(coroutine)


def _registration(start_id: str = "start-1") -> WorkerRegistration:
    return WorkerRegistration(
        worker_id="worker-0",
        start_id=start_id,
        instance_id=7,
        computation_endpoint="worker-0:50051",
        peer_weight_endpoint="http://worker-0:50052",
        backend="torch",
        activation_dtype=torch.bfloat16,
        device="cuda:0",
        max_experts=8,
        max_batch_tokens=4096,
        max_active_batches=2,
        max_pending_batches=2,
        transport_type="grpc",
    )


class _TwoHeartbeatRpc:
    def __init__(self) -> None:
        self.requests: list[lifecycle_pb2.HeartbeatRequest] = []
        self.first_received = asyncio.Event()

    async def send_heartbeats(
        self,
        requests: AsyncIterator[lifecycle_pb2.HeartbeatRequest],
    ) -> lifecycle_pb2.HeartbeatSummary:
        first = await anext(requests)
        self.requests.append(first)
        self.first_received.set()
        second = await anext(requests)
        self.requests.append(second)
        return lifecycle_pb2.HeartbeatSummary(last_sequence=second.sequence)


class _OneHeartbeatRpc:
    def __init__(self) -> None:
        self.sequence = 0

    async def send_heartbeats(
        self,
        requests: AsyncIterator[lifecycle_pb2.HeartbeatRequest],
    ) -> lifecycle_pb2.HeartbeatSummary:
        request = await anext(requests)
        self.sequence = request.sequence
        return lifecycle_pb2.HeartbeatSummary(last_sequence=request.sequence)


def test_heartbeat_state_change_sends_immediately_without_resetting_sequence() -> None:
    async def scenario() -> None:
        sender = HeartbeatSender(
            worker_id="worker-0",
            start_id="start-1",
            interval_secs=60,
        )
        rpc = _TwoHeartbeatRpc()
        running = asyncio.create_task(sender.run_once(rpc))
        await rpc.first_received.wait()

        assert sender.set_shutting_down() is True
        assert sender.set_shutting_down() is False
        await running

        assert [request.sequence for request in rpc.requests] == [1, 2]
        assert rpc.requests[0].state == lifecycle_pb2.WORKER_RUNNING
        assert rpc.requests[1].state == lifecycle_pb2.WORKER_SHUTTING_DOWN
        assert sender.last_sequence == 2

    run(scenario())


def test_heartbeat_sequence_continues_across_stream_reconnect() -> None:
    async def scenario() -> None:
        sender = HeartbeatSender(
            worker_id="worker-0",
            start_id="start-1",
            interval_secs=60,
        )
        first = _OneHeartbeatRpc()
        second = _OneHeartbeatRpc()

        await sender.run_once(first)
        await sender.run_once(second)

        assert first.sequence == 1
        assert second.sequence == 2
        assert sender.last_sequence == 2

    run(scenario())


class _LifecycleServicer(lifecycle_pb2_grpc.WorkerLifecycleServiceServicer):
    def __init__(self) -> None:
        self.registrations: list[lifecycle_pb2.RegisterWorkerRequest] = []
        self.heartbeats: list[lifecycle_pb2.HeartbeatRequest] = []

    async def RegisterWorker(
        self,
        request: lifecycle_pb2.RegisterWorkerRequest,
        _context: grpc.aio.ServicerContext,
    ) -> lifecycle_pb2.RegisterWorkerResponse:
        self.registrations.append(request)
        return lifecycle_pb2.RegisterWorkerResponse(
            current_topology_version=11,
            current_placement_generation=5,
        )

    async def Heartbeat(
        self,
        request_iterator: AsyncIterator[lifecycle_pb2.HeartbeatRequest],
        _context: grpc.aio.ServicerContext,
    ) -> lifecycle_pb2.HeartbeatSummary:
        request = await anext(request_iterator)
        self.heartbeats.append(request)
        return lifecycle_pb2.HeartbeatSummary(last_sequence=request.sequence)


class _WeightControlServicer(weight_control_pb2_grpc.WeightControlServiceServicer):
    def __init__(self) -> None:
        self.open: weight_control_pb2.OpenWeightStream | None = None

    async def Sync(
        self,
        request_iterator: AsyncIterator[weight_control_pb2.WorkerWeightMessage],
        _context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[weight_control_pb2.ControllerWeightMessage]:
        request = await anext(request_iterator)
        self.open = request.open
        yield weight_control_pb2.ControllerWeightMessage(
            targets=weight_control_pb2.TargetExpertListPart(
                placement_generation=5,
                part_index=0,
                part_count=1,
            )
        )


def test_plaintext_connection_registers_and_reuses_channel_for_heartbeat() -> None:
    async def scenario() -> None:
        servicer = _LifecycleServicer()
        weight_servicer = _WeightControlServicer()
        server = grpc.aio.server()
        lifecycle_pb2_grpc.add_WorkerLifecycleServiceServicer_to_server(servicer, server)
        weight_control_pb2_grpc.add_WeightControlServiceServicer_to_server(
            weight_servicer,
            server,
        )
        port = server.add_insecure_port("127.0.0.1:0")
        await server.start()
        connection = ControllerConnection(f"127.0.0.1:{port}")
        sender = HeartbeatSender(
            worker_id="worker-0",
            start_id="start-1",
            interval_secs=60,
        )
        try:
            await connection.start()
            result = await connection.register(_registration(), timeout_secs=2)
            await sender.run_once(connection)

            async def weight_requests() -> AsyncIterator[weight_control_pb2.WorkerWeightMessage]:
                yield weight_control_pb2.WorkerWeightMessage(
                    open=weight_control_pb2.OpenWeightStream(
                        worker_id="worker-0",
                        start_id="start-1",
                    )
                )

            weight_responses = [
                response async for response in connection.sync_weights(weight_requests())
            ]

            assert result.topology_version == 11
            assert result.placement_generation == 5
            assert servicer.registrations[0].device.max_experts == 8
            assert servicer.registrations[0].activation_dtype == common_pb2.ACTIVATION_DTYPE_BF16
            assert servicer.registrations[0].transport_type == lifecycle_pb2.WORKER_TRANSPORT_GRPC
            assert servicer.heartbeats[0].sequence == 1
            assert weight_servicer.open is not None
            assert weight_servicer.open.start_id == "start-1"
            assert weight_responses[0].targets.placement_generation == 5
        finally:
            sender.close()
            await connection.close()
            await server.stop(None)

    run(scenario())


def test_registration_and_identity_validation_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="max_experts"):
        replace(_registration(), max_experts=0)
    with pytest.raises(ValueError, match="transport_type"):
        replace(_registration(), transport_type="rdma")
    assert new_start_id() != new_start_id()
