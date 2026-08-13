"""Fixed registered-memory arenas used by the Transfer Engine data path."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from expertkit_transport.transports.base import WorkerEndpointConfig

_ALIGNMENT = 256
_UINT64_MAX = (1 << 64) - 1
_MAX_ARENA_SLOTS = 4096


def _align(value: int) -> int:
    return (value + _ALIGNMENT - 1) // _ALIGNMENT * _ALIGNMENT


@dataclass(frozen=True, slots=True)
class TransferArenaLayout:
    """Describe byte offsets for one fixed request/response slot."""

    hidden_offset: int
    expert_ids_offset: int
    routing_weights_offset: int
    output_offset: int
    slot_stride: int

    @classmethod
    def for_endpoint(cls, spec: WorkerEndpointConfig) -> TransferArenaLayout:
        hidden_bytes = spec.max_batch_tokens * spec.hidden_dim * spec.activation_element_bytes
        routing_elements = spec.max_batch_tokens * spec.top_k
        expert_ids_offset = _align(hidden_bytes)
        routing_weights_offset = _align(expert_ids_offset + routing_elements * 4)
        output_offset = _align(routing_weights_offset + routing_elements * 4)
        slot_stride = _align(output_offset + hidden_bytes)
        return cls(
            hidden_offset=0,
            expert_ids_offset=expert_ids_offset,
            routing_weights_offset=routing_weights_offset,
            output_offset=output_offset,
            slot_stride=slot_stride,
        )


@dataclass(frozen=True, slots=True)
class TransferArenaDescriptor:
    """Wire-safe description of a peer's registered arena."""

    base_address: int
    slot_count: int
    layout: TransferArenaLayout

    def __post_init__(self) -> None:
        if (
            isinstance(self.base_address, bool)
            or not isinstance(self.base_address, int)
            or not 0 < self.base_address <= _UINT64_MAX
        ):
            raise ValueError("Transfer Engine arena base address must be a positive uint64")
        if (
            isinstance(self.slot_count, bool)
            or not isinstance(self.slot_count, int)
            or self.slot_count <= 0
        ):
            raise ValueError("Transfer Engine arena slot count must be positive")

    def validate(
        self,
        spec: WorkerEndpointConfig,
        *,
        max_slots: int = _MAX_ARENA_SLOTS,
    ) -> None:
        if self.slot_count > max_slots:
            raise ValueError("Transfer Engine arena slot count exceeds the protocol limit")
        if self.layout != TransferArenaLayout.for_endpoint(spec):
            raise ValueError("Transfer Engine arena layout does not match the endpoint")
        last = self.base_address + self.slot_count * self.layout.slot_stride
        if last > _UINT64_MAX + 1:
            raise ValueError("Transfer Engine arena address range overflows uint64")

    def addresses(self, slot_index: int) -> tuple[int, int, int, int]:
        if (
            isinstance(slot_index, bool)
            or not isinstance(slot_index, int)
            or not 0 <= slot_index < self.slot_count
        ):
            raise ValueError("Transfer Engine arena slot index is out of range")
        base = self.base_address + slot_index * self.layout.slot_stride
        return (
            base + self.layout.hidden_offset,
            base + self.layout.expert_ids_offset,
            base + self.layout.routing_weights_offset,
            base + self.layout.output_offset,
        )


@dataclass(slots=True, eq=False)
class TransferArenaSlot:
    """Typed Tensor views over one slot of a registered byte slab."""

    index: int
    hidden_states: torch.Tensor
    expert_ids: torch.Tensor
    routing_weights: torch.Tensor
    partial_output: torch.Tensor
    generation: int = 0
    in_use: bool = False
    copy_event: torch.cuda.Event | None = None


class TransferArena:
    """Allocate one contiguous slab and expose a bounded reusable slot pool."""

    def __init__(
        self,
        endpoint_config: WorkerEndpointConfig,
        *,
        device: torch.device | str,
        capacity: int,
    ) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        if capacity > _MAX_ARENA_SLOTS:
            raise ValueError(f"Transfer Engine arena capacity cannot exceed {_MAX_ARENA_SLOTS}")
        self._spec = endpoint_config
        self._device = torch.device(device)
        if self._device.type not in {"cpu", "cuda"}:
            raise ValueError("Transfer Engine arena device must be CPU or CUDA")
        self._layout = TransferArenaLayout.for_endpoint(endpoint_config)
        self._slab = torch.empty(
            capacity * self._layout.slot_stride,
            dtype=torch.uint8,
            device=self._device,
        )
        self._slots = tuple(self._make_slot(index) for index in range(capacity))
        self._available = list(reversed(self._slots))
        self._closed = False

    @property
    def slab(self) -> torch.Tensor:
        return self._slab

    @property
    def slots(self) -> tuple[TransferArenaSlot, ...]:
        return self._slots

    @property
    def descriptor(self) -> TransferArenaDescriptor:
        return TransferArenaDescriptor(
            base_address=self._slab.data_ptr(),
            slot_count=len(self._slots),
            layout=self._layout,
        )

    def take(self) -> TransferArenaSlot | None:
        if self._closed:
            raise RuntimeError("Transfer Engine arena is closed")
        if not self._available:
            return None
        slot = self._available.pop()
        if slot.in_use:
            raise RuntimeError("Transfer Engine arena availability is corrupted")
        if slot.generation >= _UINT64_MAX:
            raise RuntimeError("Transfer Engine slot generation is exhausted")
        slot.generation += 1
        slot.in_use = True
        return slot

    def put(self, slot: TransferArenaSlot) -> None:
        if all(candidate is not slot for candidate in self._slots) or not slot.in_use:
            raise RuntimeError("invalid Transfer Engine arena slot return")
        slot.in_use = False
        self._available.append(slot)

    def close(self) -> None:
        if self._closed:
            return
        if len(self._available) != len(self._slots):
            raise RuntimeError("cannot close Transfer Engine arena while slots are active")
        for slot in self._slots:
            if slot.copy_event is not None:
                slot.copy_event.synchronize()
        self._available.clear()
        self._closed = True

    def _make_slot(self, index: int) -> TransferArenaSlot:
        base = index * self._layout.slot_stride
        spec = self._spec
        hidden_elements = spec.max_batch_tokens * spec.hidden_dim
        routing_elements = spec.max_batch_tokens * spec.top_k
        return TransferArenaSlot(
            index=index,
            hidden_states=self._view(
                base + self._layout.hidden_offset,
                hidden_elements,
                spec.dtype,
                (spec.max_batch_tokens, spec.hidden_dim),
            ),
            expert_ids=self._view(
                base + self._layout.expert_ids_offset,
                routing_elements,
                torch.int32,
                (spec.max_batch_tokens, spec.top_k),
            ),
            routing_weights=self._view(
                base + self._layout.routing_weights_offset,
                routing_elements,
                torch.float32,
                (spec.max_batch_tokens, spec.top_k),
            ),
            partial_output=self._view(
                base + self._layout.output_offset,
                hidden_elements,
                spec.dtype,
                (spec.max_batch_tokens, spec.hidden_dim),
            ),
            copy_event=torch.cuda.Event() if self._device.type == "cuda" else None,
        )

    def _view(
        self,
        offset: int,
        elements: int,
        dtype: torch.dtype,
        shape: tuple[int, int],
    ) -> torch.Tensor:
        element_bytes = torch.empty((), dtype=dtype).element_size()
        byte_view = self._slab.narrow(0, offset, elements * element_bytes)
        return byte_view.view(dtype).view(shape)


def transfer_lengths(
    spec: WorkerEndpointConfig,
    token_count: int,
) -> tuple[int, int, int, int]:
    if (
        isinstance(token_count, bool)
        or not isinstance(token_count, int)
        or not 0 < token_count <= spec.max_batch_tokens
    ):
        raise ValueError("Transfer Engine token count is out of range")
    hidden_bytes = token_count * spec.hidden_dim * spec.activation_element_bytes
    routing_bytes = token_count * spec.top_k * 4
    return hidden_bytes, routing_bytes, routing_bytes, hidden_bytes


__all__ = [
    "TransferArena",
    "TransferArenaDescriptor",
    "TransferArenaLayout",
    "TransferArenaSlot",
    "transfer_lengths",
]
