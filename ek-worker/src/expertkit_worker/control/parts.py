"""Validation and atomic assembly of multipart weight-control messages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from expertkit_proto.ek.control.v2 import weight_control_pb2

from expertkit_worker.weights import TargetExpert
from expertkit_worker.weights.dram_cache import WeightKey

MAX_EXPERTS_PER_CONTROL_PART = 64
MAX_CONTROL_PART_BYTES = 1024 * 1024
_MAX_PENDING_DRAINS = 64
_UINT64_MAX = (1 << 64) - 1


class _WirePart(Protocol):
    def ByteSize(self) -> int:
        """Return the encoded protobuf size."""

    def SerializeToString(self, *, deterministic: bool = False) -> bytes:
        """Encode the protobuf part."""


@dataclass(frozen=True, slots=True)
class PlacementTargets:
    """Hold one complete Controller target generation."""

    placement_generation: int
    experts: tuple[TargetExpert, ...]


@dataclass(frozen=True, slots=True)
class DrainAuthorization:
    """Hold one complete Controller authorization to drain admitted work."""

    drain_id: int
    placement_generation: int
    min_topology_version: int
    stop_accepting_all_computation: bool
    experts: tuple[WeightKey, ...]


@dataclass(slots=True)
class _TargetParts:
    part_count: int
    encoded: dict[int, bytes] = field(default_factory=dict)
    parts: dict[int, weight_control_pb2.TargetExpertListPart] = field(default_factory=dict)


@dataclass(slots=True)
class _DrainParts:
    placement_generation: int
    min_topology_version: int
    stop_all: bool
    part_count: int
    encoded: dict[int, bytes] = field(default_factory=dict)
    parts: dict[int, weight_control_pb2.DrainAuthorizationPart] = field(default_factory=dict)


def _validate_positive_uint64(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= _UINT64_MAX:
        raise ValueError(f"{name} must be a positive uint64")


def _validate_part(
    part: _WirePart,
    *,
    part_index: int,
    part_count: int,
    expert_count: int,
    maximum_parts: int,
) -> bytes:
    if part_count <= 0 or part_count > maximum_parts:
        raise ValueError("control part_count is outside the configured expert bound")
    if part_index < 0 or part_index >= part_count:
        raise ValueError("control part_index must be smaller than part_count")
    if expert_count > MAX_EXPERTS_PER_CONTROL_PART:
        raise ValueError("one control part contains more than 64 experts")
    byte_size = part.ByteSize()
    if byte_size > MAX_CONTROL_PART_BYTES:
        raise ValueError("one control part exceeds the implementation byte limit")
    return part.SerializeToString(deterministic=True)


class TargetListAssembler:
    """Apply only a complete newest target generation."""

    def __init__(self, max_experts: int) -> None:
        if isinstance(max_experts, bool) or not isinstance(max_experts, int) or max_experts <= 0:
            raise ValueError("max_experts must be a positive integer")
        self._max_experts = max_experts
        self._generation = 0
        self._current: _TargetParts | None = None

    def add(
        self,
        part: weight_control_pb2.TargetExpertListPart,
    ) -> PlacementTargets | None:
        """Validate one part and return a target list only when it is complete."""

        if not isinstance(part, weight_control_pb2.TargetExpertListPart):
            raise TypeError("target part has the wrong protobuf type")
        _validate_positive_uint64("placement_generation", part.placement_generation)
        generation = part.placement_generation
        if generation < self._generation:
            return None
        encoded = _validate_part(
            part,
            part_index=part.part_index,
            part_count=part.part_count,
            expert_count=len(part.experts),
            maximum_parts=self._max_experts,
        )
        if generation > self._generation:
            self._generation = generation
            self._current = _TargetParts(part.part_count)
        current = self._current
        if current is None:
            current = _TargetParts(part.part_count)
            self._current = current
        if current.part_count != part.part_count:
            raise ValueError("target parts disagree about part_count")
        previous = current.encoded.get(part.part_index)
        if previous is not None and previous != encoded:
            raise ValueError("target stream repeated one part with different content")
        if previous is None:
            current.encoded[part.part_index] = encoded
            current.parts[part.part_index] = weight_control_pb2.TargetExpertListPart.FromString(
                encoded
            )
        if len(current.parts) != current.part_count:
            return None

        experts: list[TargetExpert] = []
        seen: set[WeightKey] = set()
        for part_index in range(current.part_count):
            for wire in current.parts[part_index].experts:
                expert = TargetExpert(
                    layer_id=wire.layer_id,
                    expert_id=wire.expert_id,
                    target_device=wire.target_device,
                    peer_endpoints=tuple(wire.peer_weight_endpoints),
                )
                if expert.key in seen:
                    raise ValueError("complete target list contains a duplicate expert")
                seen.add(expert.key)
                experts.append(expert)
        if len(experts) > self._max_experts:
            raise ValueError("complete target list exceeds max_experts")
        return PlacementTargets(generation, tuple(experts))


class DrainAuthorizationAssembler:
    """Assemble complete drain authorizations independently by drain ID."""

    def __init__(self, max_experts: int) -> None:
        if isinstance(max_experts, bool) or not isinstance(max_experts, int) or max_experts <= 0:
            raise ValueError("max_experts must be a positive integer")
        self._max_experts = max_experts
        self._pending: dict[int, _DrainParts] = {}

    def add(
        self,
        part: weight_control_pb2.DrainAuthorizationPart,
    ) -> DrainAuthorization | None:
        """Validate one part and return an authorization only when complete."""

        if not isinstance(part, weight_control_pb2.DrainAuthorizationPart):
            raise TypeError("drain part has the wrong protobuf type")
        _validate_positive_uint64("drain_id", part.drain_id)
        _validate_positive_uint64("placement_generation", part.placement_generation)
        _validate_positive_uint64("min_topology_version", part.min_topology_version)
        encoded = _validate_part(
            part,
            part_index=part.part_index,
            part_count=part.part_count,
            expert_count=len(part.experts),
            maximum_parts=self._max_experts,
        )
        current = self._pending.get(part.drain_id)
        if current is None:
            if len(self._pending) >= _MAX_PENDING_DRAINS:
                raise ValueError("too many incomplete drain authorizations")
            current = _DrainParts(
                placement_generation=part.placement_generation,
                min_topology_version=part.min_topology_version,
                stop_all=part.stop_accepting_all_computation,
                part_count=part.part_count,
            )
            self._pending[part.drain_id] = current
        metadata = (
            part.placement_generation,
            part.min_topology_version,
            part.stop_accepting_all_computation,
            part.part_count,
        )
        expected = (
            current.placement_generation,
            current.min_topology_version,
            current.stop_all,
            current.part_count,
        )
        if metadata != expected:
            raise ValueError("drain parts disagree about authorization metadata")
        previous = current.encoded.get(part.part_index)
        if previous is not None and previous != encoded:
            raise ValueError("drain stream repeated one part with different content")
        if previous is None:
            current.encoded[part.part_index] = encoded
            current.parts[part.part_index] = weight_control_pb2.DrainAuthorizationPart.FromString(
                encoded
            )
        if len(current.parts) != current.part_count:
            return None

        experts: list[WeightKey] = []
        seen: set[WeightKey] = set()
        for part_index in range(current.part_count):
            for wire in current.parts[part_index].experts:
                key = WeightKey(wire.layer_id, wire.expert_id)
                if key in seen:
                    raise ValueError("complete drain authorization contains a duplicate expert")
                seen.add(key)
                experts.append(key)
        if len(experts) > self._max_experts:
            raise ValueError("complete drain authorization exceeds max_experts")
        if not experts and not current.stop_all:
            raise ValueError("expert drain authorization must name at least one expert")
        self._pending.pop(part.drain_id)
        return DrainAuthorization(
            drain_id=part.drain_id,
            placement_generation=current.placement_generation,
            min_topology_version=current.min_topology_version,
            stop_accepting_all_computation=current.stop_all,
            experts=tuple(experts),
        )
