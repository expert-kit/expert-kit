"""Worker-side shared-memory session and slot ownership."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from expertkit_transport.batches import WorkerBatch
from expertkit_transport.transports.grpc.codec import (
    GrpcProtocolError,
    validate_received_routing,
)
from expertkit_transport.transports.grpc.spec import GrpcBatchSpec
from expertkit_transport.transports.shm.codec import ExecuteSlot, OpenSession
from expertkit_transport.transports.shm.memory import SharedMemoryRegion, SharedMemorySlot


class SharedMemorySlotBusy(RuntimeError):
    """Indicate that an earlier request still owns one reusable slot."""


@dataclass(frozen=True, slots=True)
class ClaimedSharedMemoryBatch:
    """Hold one validated batch and its direct response destination."""

    batch: WorkerBatch
    output_destination: torch.Tensor
    slot_index: int
    generation: int


class WorkerSharedMemorySession:
    """Own one mapped client segment and serialize reuse of each slot."""

    def __init__(
        self,
        description: OpenSession,
        *,
        spec: GrpcBatchSpec,
        device: torch.device | str,
        directory: Path = Path("/dev/shm"),
    ) -> None:
        self.session_id = description.session_id
        self._spec = spec
        region = SharedMemoryRegion.open(
            description.segment_name,
            description.layout,
            device=device,
            directory=directory,
        )
        try:
            region.unlink()
            slots = [region.slot(index) for index in range(description.layout.slot_count)]
        except BaseException:
            region.close()
            raise
        self._region: SharedMemoryRegion | None = region
        self._slots: list[SharedMemorySlot] = slots
        self._active_generations: list[int | None] = [None] * len(self._slots)
        self._last_generations: list[int] = [0] * len(self._slots)
        self._closed = False

    @property
    def active_count(self) -> int:
        """Return the number of slots owned by admitted requests."""

        return sum(generation is not None for generation in self._active_generations)

    def claim(self, request: ExecuteSlot) -> ClaimedSharedMemoryBatch:
        """Claim and independently validate one slot generation."""

        self._require_open()
        if request.session_id != self.session_id:
            raise GrpcProtocolError("shared-memory request names the wrong session")
        if not 0 <= request.slot_index < len(self._slots):
            raise GrpcProtocolError("shared-memory slot index is outside the session")
        active = self._active_generations[request.slot_index]
        if active is not None:
            raise SharedMemorySlotBusy("shared-memory slot is still active")
        if request.generation <= self._last_generations[request.slot_index]:
            raise GrpcProtocolError("shared-memory slot generation is stale")
        self._active_generations[request.slot_index] = request.generation
        self._last_generations[request.slot_index] = request.generation

        slot = self._slots[request.slot_index]
        token_count = request.token_count
        hidden_states = slot.hidden_states[:token_count]
        expert_ids = slot.expert_ids[:token_count]
        routing_weights = slot.routing_weights[:token_count]
        output_destination = slot.partial_output[:token_count]
        try:
            distinct = validate_received_routing(
                expert_ids,
                routing_weights,
                self._spec.experts_per_layer,
            )
            batch = WorkerBatch(
                instance_id=self._spec.instance_id,
                layer_id=request.layer_id,
                topology_version=request.topology_version,
                hidden_states=hidden_states,
                token_indices=None,
                expert_ids=expert_ids,
                routing_weights=routing_weights,
                distinct_expert_ids=distinct,
            )
        except BaseException:
            self.release(request.slot_index, request.generation)
            raise
        return ClaimedSharedMemoryBatch(
            batch=batch,
            output_destination=output_destination,
            slot_index=request.slot_index,
            generation=request.generation,
        )

    def release(self, slot_index: int, generation: int) -> None:
        """Release exactly the generation that owns one slot."""

        self._require_open()
        if not 0 <= slot_index < len(self._slots):
            raise RuntimeError("shared-memory release names an unknown slot")
        if self._active_generations[slot_index] != generation:
            raise RuntimeError("shared-memory release does not own the active generation")
        self._active_generations[slot_index] = None

    def close(self) -> None:
        """Close an idle session; repeated calls are safe."""

        if self._closed:
            return
        if self.active_count:
            raise RuntimeError("cannot close a shared-memory session with active slots")
        self._closed = True
        self._slots.clear()
        region = self._region
        self._region = None
        if region is not None:
            region.close()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("shared-memory session is closed")
