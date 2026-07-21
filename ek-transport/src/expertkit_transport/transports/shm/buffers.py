"""Reusable Frontend output buffers backed by fixed shared-memory slots."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from expertkit_transport.buffers.base import OutputBufferProvider, OutputSpec, PreparedOutput
from expertkit_transport.transports.grpc.spec import GrpcBatchSpec
from expertkit_transport.transports.shm.memory import (
    SharedMemoryLayout,
    SharedMemoryRegion,
    SharedMemorySlot,
    new_session_id,
)


@dataclass(slots=True)
class ShmPreparedOutput(PreparedOutput):
    """Hold one Frontend result and its private shared-memory slot."""

    _tensor: torch.Tensor | None
    slot: SharedMemorySlot | None
    request_copy_event: torch.cuda.Event | None
    receive_event: torch.cuda.Event | None
    consume_event: torch.cuda.Event | None
    request_copy_recorded: bool = False
    receive_recorded: bool = False
    consume_recorded: bool = False
    generation: int = 0
    released: bool = False

    @property
    def tensor(self) -> torch.Tensor:
        """Return the maximum-size Frontend result Tensor."""

        if self._tensor is None:
            raise RuntimeError("shared-memory prepared output is released")
        return self._tensor


class ShmOutputBufferProvider(OutputBufferProvider):
    """Own one pinned shared-memory region and its Frontend result Tensors."""

    def __init__(
        self,
        batch_spec: GrpcBatchSpec,
        *,
        slot_count: int,
        device: torch.device | str,
    ) -> None:
        self._batch_spec = batch_spec
        self._device = torch.device(device)
        self.layout = SharedMemoryLayout(
            slot_count=slot_count,
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
        self._next_slot = 0
        self._released_count = 0
        self._close_requested = False

    @property
    def segment_name(self) -> str:
        """Return the basename sent to the same-host Worker."""

        region = self._region
        if region is None:
            raise RuntimeError("shared-memory provider is closed")
        return region.name

    def prepare(self, spec: OutputSpec) -> PreparedOutput:
        """Assign one fixed slot and allocate its Frontend result Tensor."""

        self._validate_spec(spec)
        if self._close_requested:
            raise RuntimeError("shared-memory provider is closing")
        if self._next_slot >= self.layout.slot_count:
            raise RuntimeError("shared-memory provider has no unassigned slot")
        region = self._require_region()
        slot = region.slot(self._next_slot)
        self._next_slot += 1
        uses_cuda = self._device.type == "cuda"
        if uses_cuda:
            for tensor in (
                slot.hidden_states,
                slot.expert_ids,
                slot.routing_weights,
                slot.partial_output,
            ):
                if not tensor.is_pinned():
                    raise RuntimeError("CUDA shared-memory Tensor was not registered as pinned")
        return ShmPreparedOutput(
            _tensor=torch.empty(
                (spec.max_batch_tokens, spec.hidden_dim),
                dtype=spec.dtype,
                device=self._device,
            ),
            slot=slot,
            request_copy_event=torch.cuda.Event() if uses_cuda else None,
            receive_event=torch.cuda.Event() if uses_cuda else None,
            consume_event=torch.cuda.Event() if uses_cuda else None,
        )

    def validate(self, output: PreparedOutput, spec: OutputSpec) -> None:
        """Reject a buffer from another provider or with an inconsistent spec."""

        self._validate_spec(spec)
        prepared = self.require_prepared(output)
        expected = (spec.max_batch_tokens, spec.hidden_dim)
        if prepared.tensor.shape != expected:
            raise ValueError("shared-memory output Tensor has the wrong shape")
        if prepared.tensor.dtype != spec.dtype:
            raise ValueError("shared-memory output Tensor has the wrong dtype")
        if prepared.tensor.device != self._device:
            raise ValueError("shared-memory output Tensor is on the wrong device")

    def before_receive(self, output: PreparedOutput) -> None:
        """Order a result copy after prior receive and consumption work."""

        prepared = self.require_prepared(output)
        if self._device.type != "cuda":
            return
        stream = torch.cuda.current_stream(self._device)
        if prepared.receive_recorded:
            assert prepared.receive_event is not None
            stream.wait_event(prepared.receive_event)
        if prepared.consume_recorded:
            assert prepared.consume_event is not None
            stream.wait_event(prepared.consume_event)

    def after_consume(self, output: PreparedOutput) -> None:
        """Record Frontend stream work that consumed this result."""

        prepared = self.require_prepared(output)
        if self._device.type == "cuda":
            assert prepared.consume_event is not None
            prepared.consume_event.record(torch.cuda.current_stream(self._device))
            prepared.consume_recorded = True

    def release(self, output: PreparedOutput) -> None:
        """Release one slot view after all CUDA operations have completed."""

        if not isinstance(output, ShmPreparedOutput):
            raise TypeError("shared-memory Transport requires a shared-memory prepared output")
        if output.released:
            return
        prepared = self.require_prepared(output)
        for recorded, event in (
            (prepared.request_copy_recorded, prepared.request_copy_event),
            (prepared.receive_recorded, prepared.receive_event),
            (prepared.consume_recorded, prepared.consume_event),
        ):
            if recorded:
                assert event is not None
                event.synchronize()
        prepared.released = True
        prepared.slot = None
        prepared._tensor = None
        prepared.request_copy_event = None
        prepared.receive_event = None
        prepared.consume_event = None
        self._released_count += 1
        self._close_if_unused()

    def request_close(self) -> None:
        """Close the region once every prepared slot view is released."""

        self._close_requested = True
        self._close_if_unused()

    def require_prepared(self, output: PreparedOutput) -> ShmPreparedOutput:
        """Return one live shared-memory output or reject another provider."""

        if not isinstance(output, ShmPreparedOutput):
            raise TypeError("shared-memory Transport requires a shared-memory prepared output")
        if output.released:
            raise RuntimeError("shared-memory prepared output is released")
        if output.slot is None:
            raise RuntimeError("shared-memory prepared output has no slot")
        return output

    def _validate_spec(self, spec: OutputSpec) -> None:
        expected = (
            self.layout.max_batch_tokens,
            self.layout.hidden_dim,
            self.layout.dtype,
            self._device,
        )
        actual = (spec.max_batch_tokens, spec.hidden_dim, spec.dtype, torch.device(spec.device))
        if actual != expected:
            raise ValueError("output spec does not match the shared-memory endpoint")

    def _require_region(self) -> SharedMemoryRegion:
        if self._region is None:
            raise RuntimeError("shared-memory provider is closed")
        return self._region

    def _close_if_unused(self) -> None:
        if not self._close_requested or self._released_count != self._next_slot:
            return
        region = self._region
        self._region = None
        if region is not None:
            region.close()
