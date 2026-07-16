"""Tests for Controller topology assembly and Worker resource lifetime."""

import asyncio
import time
from collections.abc import Awaitable
from typing import Any

import grpc
import pytest
import torch

from expertkit_transport._proto.ek.control.v2 import lifecycle_pb2, lifecycle_pb2_grpc
from expertkit_transport.adapters.grpc.topology import (
    GrpcTopologyProvider,
    _WorkerResource,
)
from expertkit_transport.adapters.grpc.topology_messages import (
    GrpcWorkerRoute,
    TopologyProtocolError,
)
from expertkit_transport.contracts import TransportError
from expertkit_transport.orchestration import WorkerTarget


class FakeTransport:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class FakePool:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class FakeResourceFactory:
    def __init__(self) -> None:
        self.created: list[_WorkerResource] = []

    async def __call__(self, route: GrpcWorkerRoute) -> _WorkerResource:
        transport = FakeTransport()
        pool = FakePool()
        target = WorkerTarget(
            identity=route.identity,
            transport=transport,  # type: ignore[arg-type]
            max_batch_tokens=route.max_batch_tokens,
            max_active_batches=route.max_active_batches,
            max_pending_batches=route.max_pending_batches,
        )
        resource = _WorkerResource(route, target, pool)  # type: ignore[arg-type]
        self.created.append(resource)
        return resource


def worker(
    worker_id: str,
    *,
    start_id: str = "start-1",
    endpoint: str = "127.0.0.1:50051",
) -> lifecycle_pb2.WorkerRoute:
    return lifecycle_pb2.WorkerRoute(
        worker_id=worker_id,
        start_id=start_id,
        computation_endpoint=endpoint,
        device="cuda:0",
        max_active_batches=1,
        max_pending_batches=2,
        max_batch_tokens=16,
    )


def route(
    layer_id: int,
    expert_id: int,
    *replicas: lifecycle_pb2.WorkerRoute,
) -> lifecycle_pb2.ExpertRoute:
    return lifecycle_pb2.ExpertRoute(
        layer_id=layer_id,
        expert_id=expert_id,
        replicas=replicas,
    )


def snapshot(
    version: int,
    routes: list[lifecycle_pb2.ExpertRoute],
    *,
    part_index: int = 0,
    part_count: int = 1,
) -> lifecycle_pb2.TopologyMessage:
    return lifecycle_pb2.TopologyMessage(
        snapshot=lifecycle_pb2.TopologySnapshotPart(
            instance_id=7,
            topology_version=version,
            part_index=part_index,
            part_count=part_count,
            routes=routes,
        )
    )


def update(
    previous_version: int,
    version: int,
    changes: list[lifecycle_pb2.RouteChange],
) -> lifecycle_pb2.TopologyMessage:
    return lifecycle_pb2.TopologyMessage(
        update=lifecycle_pb2.TopologyUpdatePart(
            instance_id=7,
            previous_version=previous_version,
            topology_version=version,
            part_index=0,
            part_count=1,
            changes=changes,
        )
    )


def provider(factory: Any) -> GrpcTopologyProvider:
    return GrpcTopologyProvider(
        "127.0.0.1:1",
        instance_id=7,
        num_layers=10,
        experts_per_layer=10,
        hidden_dim=8,
        top_k=2,
        dtype=torch.float16,
        device="cpu",
        resource_factory=factory,
    )


def run(coroutine: Awaitable[None]) -> None:
    asyncio.run(coroutine)


def test_multipart_snapshot_is_installed_only_after_resources_are_ready() -> None:
    async def scenario() -> None:
        factory = FakeResourceFactory()
        topology = provider(factory)
        shared_worker = worker("worker-a")

        await topology._consume(
            snapshot(1, [route(0, 0, shared_worker)], part_index=0, part_count=2)
        )
        with pytest.raises(TransportError):
            topology.current(7)
        assert factory.created == []

        await topology._consume(
            snapshot(1, [route(0, 1, shared_worker)], part_index=1, part_count=2)
        )
        installed = topology.current(7)
        assert installed.version == 1
        assert set(installed.routes) == {(0, 0), (0, 1)}
        assert len(factory.created) == 1
        assert len(topology.pools) == 1

        await topology._consume(
            update(
                1,
                2,
                [lifecycle_pb2.RouteChange(layer_id=0, expert_id=0)],
            )
        )
        assert set(topology.current(7).routes) == {(0, 1)}
        assert len(factory.created) == 1

        resource = factory.created[0]
        await topology._consume(
            update(
                2,
                3,
                [lifecycle_pb2.RouteChange(layer_id=0, expert_id=1)],
            )
        )
        await asyncio.sleep(0)
        assert resource.target.transport.closed is True
        assert resource.pool.closed is True
        assert topology.pools == {}
        await topology.close()

    run(scenario())


def test_changed_worker_metadata_replaces_its_resources_atomically() -> None:
    async def scenario() -> None:
        factory = FakeResourceFactory()
        topology = provider(factory)
        await topology._consume(snapshot(1, [route(0, 0, worker("worker-a"))]))
        old = factory.created[0]

        replacement = worker("worker-a", endpoint="127.0.0.1:50052")
        await topology._consume(
            update(
                1,
                2,
                [
                    lifecycle_pb2.RouteChange(
                        layer_id=0,
                        expert_id=0,
                        replicas=[replacement],
                    )
                ],
            )
        )
        assert len(factory.created) == 2
        assert topology.current(7).routes[(0, 0)][0] is factory.created[1].target
        await asyncio.sleep(0)
        assert old.target.transport.closed is True
        assert old.pool.closed is True
        assert next(iter(topology.pools.values())) is factory.created[1].pool
        await topology.close()

    run(scenario())


@pytest.mark.parametrize(
    "messages",
    [
        [lifecycle_pb2.TopologyMessage()],
        [snapshot(1, [route(0, 0)])],
        [snapshot(1, [route(10, 0, worker("worker-a"))])],
        [
            snapshot(1, [route(0, 0, worker("worker-a"))]),
            update(0, 2, []),
        ],
        [
            snapshot(1, [], part_index=0, part_count=2),
            snapshot(1, [], part_index=0, part_count=2),
        ],
    ],
)
def test_malformed_or_discontinuous_topology_is_rejected(
    messages: list[lifecycle_pb2.TopologyMessage],
) -> None:
    async def scenario() -> None:
        topology = provider(FakeResourceFactory())
        with pytest.raises(TopologyProtocolError):
            for message in messages:
                await topology._consume(message)
        await topology.close()

    run(scenario())


class ReconnectingTopology(lifecycle_pb2_grpc.TopologyServiceServicer):
    def __init__(self) -> None:
        self.requests: list[int] = []
        self.second_stream = asyncio.Event()
        self.finish = asyncio.Event()

    async def WatchTopology(self, request, context):
        self.requests.append(request.current_version)
        if len(self.requests) == 1:
            yield snapshot(1, [route(0, 0, worker("worker-a"))])
            await context.abort(grpc.StatusCode.UNAVAILABLE, "restart stream")
        self.second_stream.set()
        yield update(
            1,
            2,
            [lifecycle_pb2.RouteChange(layer_id=0, expert_id=0)],
        )
        await self.finish.wait()


def test_watcher_reconnects_from_the_last_complete_version() -> None:
    async def scenario() -> None:
        service = ReconnectingTopology()
        server = grpc.aio.server()
        lifecycle_pb2_grpc.add_TopologyServiceServicer_to_server(service, server)
        port = server.add_insecure_port("127.0.0.1:0")
        assert port > 0
        await server.start()
        topology = GrpcTopologyProvider(
            f"127.0.0.1:{port}",
            instance_id=7,
            num_layers=2,
            experts_per_layer=2,
            hidden_dim=8,
            top_k=2,
            dtype=torch.float16,
            device="cpu",
            reconnect_delay_seconds=0.001,
            resource_factory=FakeResourceFactory(),
        )
        try:
            await topology.start(monotonic_deadline=time.monotonic() + 1)
            await asyncio.wait_for(service.second_stream.wait(), timeout=1)
            while topology.current(7).version != 2:
                await asyncio.sleep(0)
            assert service.requests[:2] == [0, 1]
            assert topology.current(7).routes == {}
        finally:
            service.finish.set()
            await topology.close()
            await server.stop(None)

    run(scenario())
