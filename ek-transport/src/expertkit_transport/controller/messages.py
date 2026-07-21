"""Validation and assembly of multipart Controller topology messages."""

from __future__ import annotations

import math
from dataclasses import dataclass

from expertkit_proto.ek.control.v2 import lifecycle_pb2

from expertkit_transport.routing import WorkerIdentity

_MAX_ROUTES_PER_PART = 64


class TopologyProtocolError(ValueError):
    """Report an invalid or discontinuous Controller topology message."""


@dataclass(frozen=True, slots=True)
class WorkerRoute:
    """Hold validated connection and admission data for one Worker process."""

    identity: WorkerIdentity
    endpoint: str
    device: str
    max_active_batches: int
    max_pending_batches: int
    max_batch_tokens: int
    transport_type: int

    @property
    def max_in_flight(self) -> int:
        """Return the published active plus pending request bound."""

        return self.max_active_batches + self.max_pending_batches


RouteDescriptions = dict[tuple[int, int], tuple[WorkerRoute, ...]]


@dataclass(slots=True)
class _MultipartMessage:
    kind: str
    version: int
    previous_version: int | None
    part_count: int
    parts: dict[int, object]


def _validate_positive(name: str, value: int) -> None:
    if value <= 0:
        raise TopologyProtocolError(f"{name} must be positive")


class TopologyMessageAssembler:
    """Validate complete snapshot and update messages against model bounds."""

    def __init__(
        self,
        *,
        instance_id: int,
        num_layers: int,
        experts_per_layer: int,
    ) -> None:
        self._instance_id = instance_id
        self._num_layers = num_layers
        self._experts_per_layer = experts_per_layer
        self._pending: _MultipartMessage | None = None

    def discard_incomplete(self) -> None:
        """Discard parts retained from a stream that ended early."""

        self._pending = None

    def consume(
        self,
        message: lifecycle_pb2.TopologyMessage,
        *,
        installed_version: int,
        installed_routes: RouteDescriptions,
    ) -> tuple[int, RouteDescriptions] | None:
        """Return a complete validated routing view or retain one partial message."""

        kind = message.WhichOneof("message")
        if kind == "snapshot":
            part = message.snapshot
            if part.instance_id != self._instance_id:
                raise TopologyProtocolError("snapshot instance_id does not match the request")
            return self._consume_part(
                kind="snapshot",
                version=part.topology_version,
                previous_version=None,
                part_index=part.part_index,
                part_count=part.part_count,
                entry_count=len(part.routes),
                part=part,
                installed_version=installed_version,
                installed_routes=installed_routes,
            )
        if kind == "update":
            part = message.update
            if part.instance_id != self._instance_id:
                raise TopologyProtocolError("update instance_id does not match the request")
            return self._consume_part(
                kind="update",
                version=part.topology_version,
                previous_version=part.previous_version,
                part_index=part.part_index,
                part_count=part.part_count,
                entry_count=len(part.changes),
                part=part,
                installed_version=installed_version,
                installed_routes=installed_routes,
            )
        raise TopologyProtocolError("topology message has no snapshot or update")

    def _consume_part(
        self,
        *,
        kind: str,
        version: int,
        previous_version: int | None,
        part_index: int,
        part_count: int,
        entry_count: int,
        part: object,
        installed_version: int,
        installed_routes: RouteDescriptions,
    ) -> tuple[int, RouteDescriptions] | None:
        if part_count <= 0 or part_index >= part_count:
            raise TopologyProtocolError("topology part index or count is invalid")
        if entry_count > _MAX_ROUTES_PER_PART:
            raise TopologyProtocolError("topology part contains more than 64 routes")
        max_routes = self._num_layers * self._experts_per_layer
        max_parts = max(1, math.ceil(max_routes / _MAX_ROUTES_PER_PART))
        if part_count > max_parts:
            raise TopologyProtocolError("topology part count exceeds the model route bound")

        pending = self._pending
        if pending is None:
            pending = _MultipartMessage(
                kind=kind,
                version=version,
                previous_version=previous_version,
                part_count=part_count,
                parts={},
            )
            self._pending = pending
        elif (
            pending.kind != kind
            or pending.version != version
            or pending.previous_version != previous_version
            or pending.part_count != part_count
        ):
            raise TopologyProtocolError("topology multipart messages were interleaved")
        if part_index in pending.parts:
            raise TopologyProtocolError("topology part index was repeated")
        pending.parts[part_index] = part
        if len(pending.parts) != part_count:
            return None

        self._pending = None
        ordered = [pending.parts[index] for index in range(part_count)]
        if kind == "snapshot":
            routes = [route for completed in ordered for route in completed.routes]
            descriptions = self._decode_routes(routes, allow_empty=False)
            if version < installed_version:
                raise TopologyProtocolError("topology snapshot version moved backwards")
            if descriptions and version == 0:
                raise TopologyProtocolError("a nonempty topology must have a positive version")
            if version == installed_version and descriptions != installed_routes:
                raise TopologyProtocolError("one topology version has conflicting snapshots")
            return version, descriptions

        if previous_version != installed_version or version <= previous_version:
            raise TopologyProtocolError("topology update does not join the installed version")
        changes = [change for completed in ordered for change in completed.changes]
        if not changes:
            raise TopologyProtocolError("topology update must contain at least one change")
        decoded = self._decode_routes(changes, allow_empty=True)
        descriptions = dict(installed_routes)
        seen: set[tuple[int, int]] = set()
        for change in changes:
            key = (change.layer_id, change.expert_id)
            if key in seen:
                raise TopologyProtocolError("topology update repeats one expert route")
            seen.add(key)
            replicas = decoded.get(key, ())
            if replicas:
                descriptions[key] = replicas
            else:
                descriptions.pop(key, None)
        return version, descriptions

    def _decode_routes(
        self,
        routes: object,
        *,
        allow_empty: bool,
    ) -> RouteDescriptions:
        decoded: RouteDescriptions = {}
        worker_metadata: dict[WorkerIdentity, WorkerRoute] = {}
        for route in routes:  # type: ignore[union-attr]
            if route.layer_id >= self._num_layers or route.expert_id >= self._experts_per_layer:
                raise TopologyProtocolError("topology route is outside the model bounds")
            key = (route.layer_id, route.expert_id)
            if key in decoded:
                raise TopologyProtocolError("topology repeats one expert route")
            replicas: list[WorkerRoute] = []
            identities: set[WorkerIdentity] = set()
            for replica in route.replicas:
                if (
                    not replica.worker_id
                    or not replica.start_id
                    or not replica.computation_endpoint
                    or not replica.device
                ):
                    raise TopologyProtocolError(
                        "topology Worker identity, endpoint, and device are required"
                    )
                _validate_positive("max_active_batches", replica.max_active_batches)
                _validate_positive("max_pending_batches", replica.max_pending_batches)
                _validate_positive("max_batch_tokens", replica.max_batch_tokens)
                if replica.transport_type not in {
                    lifecycle_pb2.WORKER_TRANSPORT_GRPC,
                    lifecycle_pb2.WORKER_TRANSPORT_SHM,
                }:
                    raise TopologyProtocolError("topology Worker Transport type is invalid")
                identity = WorkerIdentity(replica.worker_id, replica.start_id)
                if identity in identities:
                    raise TopologyProtocolError("topology route repeats one Worker process")
                identities.add(identity)
                description = WorkerRoute(
                    identity=identity,
                    endpoint=replica.computation_endpoint,
                    device=replica.device,
                    max_active_batches=replica.max_active_batches,
                    max_pending_batches=replica.max_pending_batches,
                    max_batch_tokens=replica.max_batch_tokens,
                    transport_type=replica.transport_type,
                )
                existing = worker_metadata.setdefault(identity, description)
                if existing != description:
                    raise TopologyProtocolError(
                        "one Worker process has inconsistent topology metadata"
                    )
                replicas.append(description)
            if not replicas and not allow_empty:
                raise TopologyProtocolError("snapshot route has no ready Worker replica")
            decoded[key] = tuple(replicas)
        return decoded
