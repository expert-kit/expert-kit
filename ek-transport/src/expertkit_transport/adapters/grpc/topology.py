"""Frontend topology stream and per-Worker gRPC resources."""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import grpc
import torch

from expertkit_transport._proto.ek.control.v2 import lifecycle_pb2, lifecycle_pb2_grpc
from expertkit_transport.adapters.grpc.client import GrpcWorkerTransport
from expertkit_transport.adapters.grpc.spec import GrpcBatchSpec
from expertkit_transport.adapters.grpc.topology_messages import (
    GrpcWorkerRoute,
    RouteDescriptions,
    TopologyMessageAssembler,
    TopologyProtocolError,
)
from expertkit_transport.buffers import OutputPool
from expertkit_transport.contracts import OutputSpec, TransportError, TransportErrorCode
from expertkit_transport.orchestration import (
    TopologyProvider,
    TopologySnapshot,
    WorkerIdentity,
    WorkerTarget,
)

_CONTROL_MESSAGE_BYTES = 1024 * 1024

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _WorkerResource:
    route: GrpcWorkerRoute
    target: WorkerTarget
    pool: OutputPool


ResourceFactory = Callable[[GrpcWorkerRoute], Awaitable[_WorkerResource]]


def _transport_error(
    code: TransportErrorCode,
    diagnostic: str,
    *,
    retryable: bool = True,
) -> TransportError:
    return TransportError(code, retryable=retryable, diagnostic=diagnostic)


class GrpcTopologyProvider(TopologyProvider):
    """Watch one Controller topology and own matching Worker connections.

    A complete snapshot or update is validated and its new Worker connections
    and output pools are prepared before the routing view becomes visible.
    This keeps allocation and channel creation out of Routed-MoE execution.

    Note:
        Create, start, execute, and close this object on one asyncio event loop.
        The `pools` mapping is live and must be passed directly to
        `execute_routed_layer`.
    """

    def __init__(
        self,
        controller_endpoint: str,
        *,
        instance_id: int,
        num_layers: int,
        experts_per_layer: int,
        hidden_dim: int,
        top_k: int,
        dtype: torch.dtype,
        device: torch.device | str,
        reconnect_delay_seconds: float = 0.1,
        clock: Callable[[], float] = time.monotonic,
        resource_factory: ResourceFactory | None = None,
    ) -> None:
        if not controller_endpoint:
            raise ValueError("controller_endpoint must not be empty")
        for name, value in (
            ("instance_id", instance_id),
            ("num_layers", num_layers),
            ("experts_per_layer", experts_per_layer),
            ("hidden_dim", hidden_dim),
            ("top_k", top_k),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if top_k > experts_per_layer:
            raise ValueError("top_k must not exceed experts_per_layer")
        if dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("dtype must be FP16, BF16, or FP32")
        if not math.isfinite(reconnect_delay_seconds) or reconnect_delay_seconds <= 0:
            raise ValueError("reconnect_delay_seconds must be finite and positive")

        self._controller_endpoint = controller_endpoint
        self._instance_id = instance_id
        self._num_layers = num_layers
        self._experts_per_layer = experts_per_layer
        self._hidden_dim = hidden_dim
        self._top_k = top_k
        self._dtype = dtype
        self._device = torch.device(device)
        self._reconnect_delay_seconds = reconnect_delay_seconds
        self._clock = clock
        self._resource_factory = resource_factory or self._create_resource

        self._snapshot = TopologySnapshot(instance_id=instance_id, version=0, routes={})
        self._route_descriptions: RouteDescriptions = {}
        self._resources: dict[WorkerIdentity, _WorkerResource] = {}
        self._pools: dict[WorkerIdentity, OutputPool] = {}
        self._pools_view: Mapping[WorkerIdentity, OutputPool] = MappingProxyType(self._pools)
        self._messages = TopologyMessageAssembler(
            instance_id=instance_id,
            num_layers=num_layers,
            experts_per_layer=experts_per_layer,
        )
        self._channel: grpc.aio.Channel | None = None
        self._watch_task: asyncio.Task[None] | None = None
        self._cleanup_tasks: set[asyncio.Task[None]] = set()
        self._first_install = asyncio.Event()
        self._failure: BaseException | None = None
        self._closing = False
        self._close_task: asyncio.Task[None] | None = None

    @property
    def pools(self) -> Mapping[WorkerIdentity, OutputPool]:
        """Return the live mapping of preallocated output pools."""

        return self._pools_view

    async def start(self, *, monotonic_deadline: float) -> None:
        """Start watching and wait for the Controller's first complete snapshot."""

        if self._closing:
            raise RuntimeError("topology provider is closing")
        if self._watch_task is not None:
            if self._failure is not None:
                raise _transport_error(
                    TransportErrorCode.PROTOCOL,
                    f"the topology stream failed: {self._failure}",
                    retryable=False,
                )
            return
        if monotonic_deadline - self._clock() <= 0:
            raise _transport_error(
                TransportErrorCode.DEADLINE_EXCEEDED,
                "the topology startup deadline already expired",
                retryable=False,
            )
        self._channel = grpc.aio.insecure_channel(
            self._controller_endpoint,
            options=(
                ("grpc.max_send_message_length", _CONTROL_MESSAGE_BYTES),
                ("grpc.max_receive_message_length", _CONTROL_MESSAGE_BYTES),
            ),
        )
        self._watch_task = asyncio.create_task(self._watch(), name="expertkit-topology-watch")
        remaining = monotonic_deadline - self._clock()
        try:
            async with asyncio.timeout(remaining):
                await self._first_install.wait()
        except TimeoutError as error:
            await self.close()
            raise _transport_error(
                TransportErrorCode.DEADLINE_EXCEEDED,
                "the initial topology snapshot did not arrive before the deadline",
                retryable=False,
            ) from error
        if self._failure is not None:
            await self.close()
            raise _transport_error(
                TransportErrorCode.PROTOCOL,
                f"the topology stream failed: {self._failure}",
                retryable=False,
            ) from self._failure

    def current(self, instance_id: int) -> TopologySnapshot:
        """Return the latest fully installed routing snapshot."""

        if instance_id != self._instance_id:
            raise ValueError("requested instance does not match this topology provider")
        if self._failure is not None:
            raise _transport_error(
                TransportErrorCode.PROTOCOL,
                f"the topology stream failed: {self._failure}",
                retryable=False,
            )
        if not self._first_install.is_set():
            raise _transport_error(
                TransportErrorCode.UNAVAILABLE,
                "the initial topology snapshot is not installed",
            )
        return self._snapshot

    async def refresh(
        self,
        instance_id: int,
        *,
        observed_version: int,
        monotonic_deadline: float,
    ) -> TopologySnapshot:
        """Return the newest complete snapshot already received by the watcher."""

        if monotonic_deadline - self._clock() <= 0:
            raise _transport_error(
                TransportErrorCode.DEADLINE_EXCEEDED,
                "the topology refresh deadline expired",
                retryable=False,
            )
        await asyncio.sleep(0)
        snapshot = self.current(instance_id)
        if snapshot.version < observed_version:
            raise _transport_error(
                TransportErrorCode.PROTOCOL,
                "the installed topology version moved backwards",
                retryable=False,
            )
        return snapshot

    async def close(self) -> None:
        """Stop the watcher and release every Worker connection and output pool."""

        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close())
        await asyncio.shield(self._close_task)

    async def _close(self) -> None:
        self._closing = True
        if self._watch_task is not None:
            self._watch_task.cancel()
            await asyncio.gather(self._watch_task, return_exceptions=True)
        if self._channel is not None:
            await self._channel.close()
        cleanup = tuple(self._cleanup_tasks)
        if cleanup:
            await asyncio.gather(*cleanup, return_exceptions=True)
        resources = tuple(self._resources.values())
        self._resources.clear()
        self._pools.clear()
        if resources:
            await asyncio.gather(
                *(self._close_resource(resource) for resource in resources),
                return_exceptions=False,
            )

    async def _watch(self) -> None:
        assert self._channel is not None
        stub = lifecycle_pb2_grpc.TopologyServiceStub(self._channel)
        while not self._closing:
            request = lifecycle_pb2.WatchTopologyRequest(
                instance_id=self._instance_id,
                current_version=self._snapshot.version,
            )
            try:
                async for message in stub.WatchTopology(request, wait_for_ready=True):
                    await self._consume(message)
                if self._closing:
                    return
                self._messages.discard_incomplete()
                logger.warning(
                    "topology stream closed",
                    extra={"instance_id": self._instance_id},
                )
                await asyncio.sleep(self._reconnect_delay_seconds)
            except asyncio.CancelledError:
                raise
            except grpc.aio.AioRpcError as error:
                if self._closing:
                    return
                self._messages.discard_incomplete()
                logger.warning(
                    "topology stream disconnected",
                    extra={"instance_id": self._instance_id, "grpc_status": error.code().name},
                )
                await asyncio.sleep(self._reconnect_delay_seconds)
            except BaseException as error:
                self._failure = error
                self._first_install.set()
                return

    async def _consume(self, message: lifecycle_pb2.TopologyMessage) -> None:
        completed = self._messages.consume(
            message,
            installed_version=self._snapshot.version,
            installed_routes=self._route_descriptions,
        )
        if completed is not None:
            version, descriptions = completed
            await self._install(version, descriptions)

    async def _install(
        self,
        version: int,
        descriptions: RouteDescriptions,
    ) -> None:
        required: dict[WorkerIdentity, GrpcWorkerRoute] = {}
        for replicas in descriptions.values():
            for route in replicas:
                existing = required.setdefault(route.identity, route)
                if existing != route:
                    raise TopologyProtocolError(
                        "one Worker process has inconsistent topology metadata"
                    )

        created: dict[WorkerIdentity, _WorkerResource] = {}
        try:
            for identity, route in required.items():
                current = self._resources.get(identity)
                if current is not None and current.route == route:
                    continue
                created[identity] = await self._resource_factory(route)
        except BaseException:
            if created:
                await asyncio.gather(
                    *(self._close_resource(resource) for resource in created.values()),
                    return_exceptions=True,
                )
            raise

        old_resources = self._resources
        new_resources: dict[WorkerIdentity, _WorkerResource] = {}
        for identity in required:
            resource = created.get(identity)
            if resource is None:
                resource = old_resources[identity]
            new_resources[identity] = resource

        targets = {
            key: tuple(new_resources[route.identity].target for route in replicas)
            for key, replicas in descriptions.items()
        }
        snapshot = TopologySnapshot(
            instance_id=self._instance_id,
            version=version,
            routes=targets,
        )

        self._resources = new_resources
        for identity, resource in new_resources.items():
            self._pools[identity] = resource.pool
        self._route_descriptions = descriptions
        self._snapshot = snapshot
        self._first_install.set()

        retired = tuple(
            resource
            for identity, resource in old_resources.items()
            if new_resources.get(identity) is not resource
        )
        for resource in retired:
            task = asyncio.create_task(self._retire_resource(resource))
            self._cleanup_tasks.add(task)
            task.add_done_callback(self._cleanup_tasks.discard)

    async def _create_resource(self, route: GrpcWorkerRoute) -> _WorkerResource:
        batch_spec = GrpcBatchSpec(
            instance_id=self._instance_id,
            num_layers=self._num_layers,
            experts_per_layer=self._experts_per_layer,
            max_batch_tokens=route.max_batch_tokens,
            hidden_dim=self._hidden_dim,
            top_k=self._top_k,
            dtype=self._dtype,
        )
        transport = GrpcWorkerTransport(
            route.endpoint,
            batch_spec,
            max_in_flight=route.max_in_flight,
        )
        try:
            pool = OutputPool(
                transport.output_buffers,
                OutputSpec(
                    max_batch_tokens=route.max_batch_tokens,
                    hidden_dim=self._hidden_dim,
                    dtype=self._dtype,
                    device=self._device,
                ),
                route.max_in_flight,
            )
        except BaseException:
            await transport.close()
            raise
        try:
            await transport.start()
        except BaseException:
            await pool.close()
            await transport.close()
            raise
        target = WorkerTarget(
            identity=route.identity,
            transport=transport,
            max_batch_tokens=route.max_batch_tokens,
            max_active_batches=route.max_active_batches,
            max_pending_batches=route.max_pending_batches,
        )
        return _WorkerResource(route=route, target=target, pool=pool)

    async def _retire_resource(self, resource: _WorkerResource) -> None:
        await self._close_resource(resource)
        if (
            self._resources.get(resource.route.identity) is not resource
            and self._pools.get(resource.route.identity) is resource.pool
        ):
            self._pools.pop(resource.route.identity, None)

    @staticmethod
    async def _close_resource(resource: _WorkerResource) -> None:
        try:
            await resource.target.transport.close()
        finally:
            await resource.pool.close()
