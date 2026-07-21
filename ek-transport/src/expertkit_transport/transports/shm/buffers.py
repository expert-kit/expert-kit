"""Fixed shared-memory transfer slots owned by one SHM Worker connection."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from expertkit_transport.transports.grpc.spec import GrpcBatchSpec
from expertkit_transport.transports.shm.memory import (
    SharedMemoryLayout,
    SharedMemoryRegion,
    SharedMemorySlot,
    new_session_id,
)


@dataclass(slots=True)
class ShmTransferBuffers:
    """Hold one shared-memory slot and its device-transfer ordering state."""

    slot: SharedMemorySlot
    request_copy_event: torch.cuda.Event | None
    receive_event: torch.cuda.Event | None
    request_copy_recorded: bool = False
    receive_recorded: bool = False
    generation: int = 0


class ShmTransferBufferPool:
    """Own one fixed shared-memory region and recycle all of its slots."""

    def __init__(
        self,
        batch_spec: GrpcBatchSpec,
        *,
        capacity: int,
        device: torch.device | str,
    ) -> None:
        self._device = torch.device(device)
        self.layout = SharedMemoryLayout(
            slot_count=capacity,
            max_batch_tokens=batch_spec.max_batch_tokens,
            hidden_dim=batch_spec.hidden_dim,
            top_k=batch_spec.top_k,
            dtype=batch_spec.dtype,
        )
        self.session_id = new_session_id()
        self._region: SharedMemoryRegion | None = SharedMemoryRegion.create(
            self.layout,
            device=self._device,
        )
        uses_cuda = self._device.type == "cuda"
        slots: list[ShmTransferBuffers] = []
        try:
            for index in range(capacity):
                slot = self._region.slot(index)
                if uses_cuda:
                    for tensor in (
                        slot.hidden_states,
                        slot.expert_ids,
                        slot.routing_weights,
                        slot.partial_output,
                    ):
                        if not tensor.is_pinned():
                            raise RuntimeError(
                                "CUDA shared-memory Tensor was not registered as pinned"
                            )
                slots.append(
                    ShmTransferBuffers(
                        slot=slot,
                        request_copy_event=torch.cuda.Event() if uses_cuda else None,
                        receive_event=torch.cuda.Event() if uses_cuda else None,
                    )
                )
        except BaseException:
            self._region.close()
            self._region = None
            raise
        self._all = tuple(slots)
        self._available = list(self._all)
        self._closed = False

    @property
    def segment_name(self) -> str:
        """Return the shared-memory basename sent to the same-host Worker."""

        region = self._region
        if region is None:
            raise RuntimeError("shared-memory transfer buffers are closed")
        return region.name

    @property
    def allocated(self) -> tuple[ShmTransferBuffers, ...]:
        """Return fixed slots for diagnostics and allocation tests."""

        return self._all

    def take(self) -> ShmTransferBuffers:
        """Take one slot after the caller has acquired connection admission."""

        if self._closed:
            raise RuntimeError("shared-memory transfer buffers are closed")
        try:
            return self._available.pop()
        except IndexError as error:
            raise RuntimeError("SHM admission and transfer buffers diverged") from error

    def put(self, buffers: ShmTransferBuffers) -> None:
        """Return one slot after the request has stopped accessing it."""

        if all(candidate is not buffers for candidate in self._all) or any(
            candidate is buffers for candidate in self._available
        ):
            raise RuntimeError("invalid shared-memory transfer buffer return")
        self._available.append(buffers)

    def close(self) -> None:
        """Wait for device copies, release slot views, and close the region."""

        if self._closed:
            return
        if len(self._available) != len(self._all):
            raise RuntimeError("cannot close shared-memory buffers while calls are active")
        for buffers in self._all:
            for recorded, event in (
                (buffers.request_copy_recorded, buffers.request_copy_event),
                (buffers.receive_recorded, buffers.receive_event),
            ):
                if recorded:
                    assert event is not None
                    event.synchronize()
        self._available.clear()
        region = self._region
        self._region = None
        if region is not None:
            region.close()
        self._closed = True
