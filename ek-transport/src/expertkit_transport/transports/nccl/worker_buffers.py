"""Fixed NCCL receive slots and Worker execution-slot copy behavior."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from expertkit_transport.batches import WorkerBatch
from expertkit_transport.errors import TransportProtocolError
from expertkit_transport.transports.base import (
    BatchBufferConfig,
    WorkerBatchBuffers,
    WorkerEndpointConfig,
)
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


@dataclass(slots=True, eq=False)
class NcclReceiveBuffers:
    """Hold one fixed request/response slot on the Worker's device."""

    hidden_states: torch.Tensor
    expert_ids: torch.Tensor
    routing_weights: torch.Tensor
    partial_output: torch.Tensor


class NcclReceiveBufferPool:
    """Bound device memory retained by admitted and active requests."""

    def __init__(
        self,
        endpoint_config: WorkerEndpointConfig,
        *,
        device: torch.device | str,
        capacity: int,
    ) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        self._device = torch.device(device)
        self._all = tuple(
            NcclReceiveBuffers(
                hidden_states=torch.empty(
                    (endpoint_config.max_batch_tokens, endpoint_config.hidden_dim),
                    dtype=endpoint_config.dtype,
                    device=self._device,
                ),
                expert_ids=torch.empty(
                    (endpoint_config.max_batch_tokens, endpoint_config.top_k),
                    dtype=torch.int32,
                    device=self._device,
                ),
                routing_weights=torch.empty(
                    (endpoint_config.max_batch_tokens, endpoint_config.top_k),
                    dtype=torch.float32,
                    device=self._device,
                ),
                partial_output=torch.empty(
                    (endpoint_config.max_batch_tokens, endpoint_config.hidden_dim),
                    dtype=endpoint_config.dtype,
                    device=self._device,
                ),
            )
            for _ in range(capacity)
        )
        self._available = list(self._all)
        self._closed = False

    @property
    def allocated(self) -> tuple[NcclReceiveBuffers, ...]:
        return self._all

    def take(self) -> NcclReceiveBuffers | None:
        if self._closed:
            raise RuntimeError("NCCL receive buffers are closed")
        if not self._available:
            return None
        return self._available.pop()

    def put(self, buffers: NcclReceiveBuffers) -> None:
        if all(candidate is not buffers for candidate in self._all) or any(
            candidate is buffers for candidate in self._available
        ):
            raise RuntimeError("invalid NCCL receive buffer return")
        self._available.append(buffers)

    def close(self) -> None:
        if self._closed:
            return
        if len(self._available) != len(self._all):
            raise RuntimeError("cannot close NCCL receive buffers while batches are active")
        self._available.clear()
        self._closed = True


class NcclWorkerBatchBuffers(WorkerBatchBuffers):
    """Copy device-resident NCCL input into one fixed Backend execution slot."""

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
            raise ValueError("a Worker-side NCCL batch must already contain compact rows")
        if batch.hidden_states.device != self._spec.device:
            raise ValueError("a Worker-side NCCL batch must use the execution device")
        if not 0 < batch.token_count <= self._spec.max_batch_tokens:
            raise ValueError("received batch token count exceeds the fixed slot")
        if batch.hidden_dim != self._spec.hidden_dim or batch.top_k != self._spec.top_k:
            raise ValueError("received batch shape does not match the fixed slot")
        if batch.hidden_states.dtype != self._spec.dtype:
            raise ValueError("received batch dtype does not match the fixed slot")
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
                "NCCL routing values do not match distinct_expert_ids metadata"
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
            raise ValueError("NCCL output requires its fixed receive-slot destination")
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


__all__ = [
    "NcclReceiveBufferPool",
    "NcclReceiveBuffers",
    "NcclWorkerBatchBuffers",
]
