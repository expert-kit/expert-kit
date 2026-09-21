"""Fixed Worker-side gRPC staging storage for one execution slot."""

from __future__ import annotations

import torch

from expertkit_transport.batches import WorkerBatch
from expertkit_transport.transports.base import BatchBufferConfig, WorkerBatchBuffers


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


class GrpcWorkerBatchBuffers(WorkerBatchBuffers):
    """Reuse pinned Host staging for accelerators and direct fixed tensors for CPU."""

    def __init__(self, spec: BatchBufferConfig) -> None:
        self._spec = spec
        self._uses_accelerator = spec.device.type in {"cuda", "npu"}
        self._closed = False
        if self._uses_accelerator:
            self._host_hidden_states = torch.empty(
                (spec.max_batch_tokens, spec.hidden_dim),
                dtype=spec.dtype,
                device="cpu",
                pin_memory=True,
            )
            self._host_expert_ids = torch.empty(
                (spec.max_batch_tokens, spec.top_k),
                dtype=torch.int32,
                device="cpu",
                pin_memory=True,
            )
            self._host_routing_weights = torch.empty(
                (spec.max_batch_tokens, spec.top_k),
                dtype=torch.float32,
                device="cpu",
                pin_memory=True,
            )
            self._host_partial_output = torch.empty(
                (spec.max_batch_tokens, spec.hidden_dim),
                dtype=spec.dtype,
                device="cpu",
                pin_memory=True,
            )
        else:
            self._host_hidden_states = None
            self._host_expert_ids = None
            self._host_routing_weights = None
            self._host_partial_output = None

    @property
    def host_staging_bytes(self) -> int:
        """Return the fixed pinned Host allocation for this execution slot."""

        if not self._uses_accelerator:
            return 0
        activation_bytes = torch.empty((), dtype=self._spec.dtype).element_size()
        return self._spec.max_batch_tokens * (
            2 * self._spec.hidden_dim * activation_bytes + 2 * self._spec.top_k * 4
        )

    def copy_input(
        self,
        batch: WorkerBatch,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> None:
        """Fill fixed Backend inputs without allocating request-sized Tensor storage."""

        self._require_open()
        if batch.token_indices is not None:
            raise ValueError("a Worker-side gRPC batch must already contain compact rows")
        if batch.hidden_states.device.type != "cpu":
            raise ValueError("a Worker-side gRPC batch must contain CPU tensors")
        if not 0 < batch.token_count <= self._spec.max_batch_tokens:
            raise ValueError("received batch token count exceeds the fixed slot")
        if batch.hidden_dim != self._spec.hidden_dim or batch.top_k != self._spec.top_k:
            raise ValueError("received batch shape does not match the fixed slot")
        if batch.hidden_states.dtype != self._spec.dtype:
            raise ValueError("received batch dtype does not match the fixed slot")
        for name, tensor in (
            ("received hidden_states", batch.hidden_states),
            ("received expert_ids", batch.expert_ids),
            ("received routing_weights", batch.routing_weights),
        ):
            if not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
        shape = (batch.token_count, self._spec.hidden_dim)
        routing_shape = (batch.token_count, self._spec.top_k)
        _require_tensor(
            "hidden_states",
            hidden_states,
            shape=shape,
            dtype=self._spec.dtype,
            device=self._spec.device,
        )
        _require_tensor(
            "expert_ids",
            expert_ids,
            shape=routing_shape,
            dtype=torch.int32,
            device=self._spec.device,
        )
        _require_tensor(
            "routing_weights",
            routing_weights,
            shape=routing_shape,
            dtype=torch.float32,
            device=self._spec.device,
        )

        if not self._uses_accelerator:
            hidden_states.copy_(batch.hidden_states)
            expert_ids.copy_(batch.expert_ids)
            routing_weights.copy_(batch.routing_weights)
            return

        if (
            batch.hidden_states.is_pinned()
            and batch.expert_ids.is_pinned()
            and batch.routing_weights.is_pinned()
        ):
            hidden_states.copy_(batch.hidden_states, non_blocking=True)
            expert_ids.copy_(batch.expert_ids, non_blocking=True)
            routing_weights.copy_(batch.routing_weights, non_blocking=True)
            return

        host_hidden = self._require_host(self._host_hidden_states)[: batch.token_count]
        host_expert_ids = self._require_host(self._host_expert_ids)[: batch.token_count]
        host_routing = self._require_host(self._host_routing_weights)[: batch.token_count]
        host_hidden.copy_(batch.hidden_states)
        host_expert_ids.copy_(batch.expert_ids)
        host_routing.copy_(batch.routing_weights)
        hidden_states.copy_(host_hidden, non_blocking=True)
        expert_ids.copy_(host_expert_ids, non_blocking=True)
        routing_weights.copy_(host_routing, non_blocking=True)

    def copy_output(
        self,
        partial_output: torch.Tensor,
        destination: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return or copy output into private staging or a Transport destination."""

        self._require_open()
        _require_tensor(
            "partial_output",
            partial_output,
            shape=(partial_output.shape[0], self._spec.hidden_dim),
            dtype=self._spec.dtype,
            device=self._spec.device,
        )
        if not 0 < partial_output.shape[0] <= self._spec.max_batch_tokens:
            raise ValueError("partial_output token count exceeds the fixed slot")
        if destination is not None:
            _require_tensor(
                "output destination",
                destination,
                shape=(partial_output.shape[0], partial_output.shape[1]),
                dtype=self._spec.dtype,
                device=torch.device("cpu"),
            )
            if self._uses_accelerator and not destination.is_pinned():
                raise ValueError("accelerator output destination must use pinned Host memory")
            destination.copy_(
                partial_output,
                non_blocking=self._uses_accelerator,
            )
            return destination
        if not self._uses_accelerator:
            return partial_output

        host_output = self._require_host(self._host_partial_output)[: partial_output.shape[0]]
        host_output.copy_(partial_output, non_blocking=True)
        return host_output

    def close(self) -> None:
        """Release pinned Host staging exactly once."""

        if self._closed:
            return
        self._closed = True
        self._host_hidden_states = None
        self._host_expert_ids = None
        self._host_routing_weights = None
        self._host_partial_output = None

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("Worker batch buffers are closed")

    @staticmethod
    def _require_host(tensor: torch.Tensor | None) -> torch.Tensor:
        if tensor is None:
            raise RuntimeError("pinned Host staging was not allocated")
        return tensor
