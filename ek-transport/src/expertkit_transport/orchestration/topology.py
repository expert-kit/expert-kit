"""Immutable in-process Topology used to route Worker computations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from expertkit_transport.contracts import WorkerTransport

_UINT32_MAX = (1 << 32) - 1
_UINT64_MAX = (1 << 64) - 1


def _require_unsigned(name: str, value: int, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [0, {maximum}]")


@dataclass(frozen=True, slots=True, order=True)
class WorkerIdentity:
    """Identify one specific Worker process start."""

    worker_id: str
    start_id: str

    def __post_init__(self) -> None:
        if not self.worker_id:
            raise ValueError("worker_id must not be empty")
        if not self.start_id:
            raise ValueError("start_id must not be empty")


@dataclass(frozen=True, slots=True)
class WorkerTarget:
    """Bind one live Topology route to a Transport connection and limits."""

    identity: WorkerIdentity
    transport: WorkerTransport
    max_batch_tokens: int
    max_active_batches: int
    max_pending_batches: int

    def __post_init__(self) -> None:
        for name in ("max_batch_tokens", "max_active_batches", "max_pending_batches"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be positive")

    @property
    def max_in_flight(self) -> int:
        """Return the published active plus pending capacity hint."""

        return self.max_active_batches + self.max_pending_batches


RouteKey = tuple[int, int]


@dataclass(frozen=True, slots=True)
class TopologySnapshot:
    """Hold one complete, atomically installed set of ready expert routes."""

    instance_id: int
    version: int
    routes: Mapping[RouteKey, tuple[WorkerTarget, ...]]

    def __post_init__(self) -> None:
        _require_unsigned("instance_id", self.instance_id, _UINT64_MAX)
        _require_unsigned("version", self.version, _UINT64_MAX)

        copied: dict[RouteKey, tuple[WorkerTarget, ...]] = {}
        targets_by_identity: dict[WorkerIdentity, WorkerTarget] = {}
        for (layer_id, expert_id), replicas in self.routes.items():
            _require_unsigned("layer_id", layer_id, _UINT32_MAX)
            _require_unsigned("expert_id", expert_id, _UINT32_MAX)
            normalized = tuple(replicas)
            if not normalized:
                raise ValueError("a published expert route must have at least one replica")
            identities = [target.identity for target in normalized]
            if len(set(identities)) != len(identities):
                raise ValueError("an expert route must not repeat a Worker process")
            for target in normalized:
                existing = targets_by_identity.setdefault(target.identity, target)
                if existing != target:
                    raise ValueError("one Worker process must have consistent route metadata")
            copied[(layer_id, expert_id)] = normalized
        object.__setattr__(self, "routes", MappingProxyType(copied))

    def layer_routes(self, layer_id: int) -> Mapping[int, tuple[WorkerTarget, ...]]:
        """Return the ready replicas indexed by expert number for one layer."""

        return MappingProxyType(
            {
                expert_id: replicas
                for (route_layer, expert_id), replicas in self.routes.items()
                if route_layer == layer_id
            }
        )
