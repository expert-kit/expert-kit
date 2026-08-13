"""Worker execution-slot copy behavior for Transfer Engine input."""

from __future__ import annotations

import torch

from expertkit_transport.batches import WorkerBatch
from expertkit_transport.errors import TransportProtocolError
from expertkit_transport.transports.base import BatchBufferConfig, WorkerBatchBuffers
from expertkit_transport.transports.validation import validate_received_routing


def _require_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    shape: tuple[int, int],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if tensor.shape != shape:
        raise ValueError(f"{name} has an unexpected shape")
    if tensor.dtype != dtype:
        raise ValueError(f"{name} has an unexpected dtype")
    if tensor.device != device:
        raise ValueError(f"{name} has an unexpected device")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


class TransferEngineWorkerBatchBuffers(WorkerBatchBuffers):
    """Copy a registered receive slot into one fixed Backend execution slot."""

    def __init__(self, spec: BatchBufferConfig, *, experts_per_layer: int) -> None:
        if (
            isinstance(experts_per_layer, bool)
            or not isinstance(experts_per_layer, int)
            or experts_per_layer <= 0
        ):
            raise ValueError("experts_per_layer must be a positive integer")
        self._spec = spec
        self._experts_per_layer = experts_per_layer
        self._closed = False
        uses_cuda = spec.device.type == "cuda"
        host_options = {"device": "cpu", "pin_memory": uses_cuda}
        self._host_expert_ids = torch.empty(
            (spec.max_batch_tokens, spec.top_k),
            dtype=torch.int32,
            **host_options,
        )
        self._host_routing_weights = torch.empty(
            (spec.max_batch_tokens, spec.top_k),
            dtype=torch.float32,
            **host_options,
        )

    @property
    def host_staging_bytes(self) -> int:
        return self._spec.max_batch_tokens * self._spec.top_k * 8

    def copy_input(
        self,
        batch: WorkerBatch,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> None:
        self._require_open()
        if batch.token_indices is not None:
            raise ValueError("a Worker-side Transfer Engine batch must contain compact rows")
        if batch.hidden_states.device != self._spec.device:
            raise ValueError("Transfer Engine batch must use the execution device")
        if not 0 < batch.token_count <= self._spec.max_batch_tokens:
            raise ValueError("received batch token count exceeds the fixed slot")
        if batch.hidden_dim != self._spec.hidden_dim or batch.top_k != self._spec.top_k:
            raise ValueError("received batch shape does not match the fixed slot")
        shape = (batch.token_count, self._spec.hidden_dim)
        routing_shape = (batch.token_count, self._spec.top_k)
        for name, tensor, dtype, expected_shape in (
            ("received hidden_states", batch.hidden_states, self._spec.dtype, shape),
            ("received expert_ids", batch.expert_ids, torch.int32, routing_shape),
            ("received routing_weights", batch.routing_weights, torch.float32, routing_shape),
            ("hidden_states", hidden_states, self._spec.dtype, shape),
            ("expert_ids", expert_ids, torch.int32, routing_shape),
            ("routing_weights", routing_weights, torch.float32, routing_shape),
        ):
            _require_tensor(
                name,
                tensor,
                shape=expected_shape,
                dtype=dtype,
                device=self._spec.device,
            )
        hidden_states.copy_(batch.hidden_states)
        expert_ids.copy_(batch.expert_ids)
        routing_weights.copy_(batch.routing_weights)

        host_expert_ids = self._host_expert_ids[: batch.token_count]
        host_routing_weights = self._host_routing_weights[: batch.token_count]
        non_blocking = self._spec.device.type == "cuda"
        host_expert_ids.copy_(expert_ids, non_blocking=non_blocking)
        host_routing_weights.copy_(routing_weights, non_blocking=non_blocking)
        if non_blocking:
            torch.cuda.current_stream(self._spec.device).synchronize()
        distinct = validate_received_routing(
            host_expert_ids,
            host_routing_weights,
            self._experts_per_layer,
        )
        if distinct != batch.distinct_expert_ids:
            raise TransportProtocolError(
                "Transfer Engine routing values do not match distinct_expert_ids metadata"
            )

    def copy_output(
        self,
        partial_output: torch.Tensor,
        destination: torch.Tensor | None,
    ) -> torch.Tensor:
        self._require_open()
        if not 0 < partial_output.shape[0] <= self._spec.max_batch_tokens:
            raise ValueError("partial_output token count exceeds the fixed slot")
        _require_tensor(
            "partial_output",
            partial_output,
            shape=(partial_output.shape[0], self._spec.hidden_dim),
            dtype=self._spec.dtype,
            device=self._spec.device,
        )
        if destination is None:
            raise ValueError("Transfer Engine output requires a registered destination")
        _require_tensor(
            "output destination",
            destination,
            shape=tuple(partial_output.shape),
            dtype=self._spec.dtype,
            device=self._spec.device,
        )
        destination.copy_(partial_output)
        return destination

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._host_expert_ids = torch.empty(0)
        self._host_routing_weights = torch.empty(0)

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("Worker batch buffers are closed")


__all__ = ["TransferEngineWorkerBatchBuffers"]
