"""Worker-side copies between shared memory and one execution slot."""

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


class ShmWorkerBatchBuffers(WorkerBatchBuffers):
    """Copy through the session's fixed mapping without private Host staging."""

    def __init__(self, spec: BatchBufferConfig) -> None:
        if spec.device.type == "npu":
            raise ValueError("SHM transport does not support NPU devices")
        self._spec = spec
        self._closed = False

    @property
    def host_staging_bytes(self) -> int:
        """Return zero because the shared-memory session owns Host storage."""

        return 0

    def copy_input(
        self,
        batch: WorkerBatch,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> None:
        """Copy one claimed shared-memory slot into fixed Backend inputs."""

        self._require_open()
        if batch.token_indices is not None:
            raise ValueError("a Worker-side SHM batch must already contain compact rows")
        if batch.hidden_states.device.type != "cpu":
            raise ValueError("a Worker-side SHM batch must contain CPU tensors")
        if not 0 < batch.token_count <= self._spec.max_batch_tokens:
            raise ValueError("received batch token count exceeds the fixed slot")
        if batch.hidden_dim != self._spec.hidden_dim or batch.top_k != self._spec.top_k:
            raise ValueError("received batch shape does not match the fixed slot")
        if batch.hidden_states.dtype != self._spec.dtype:
            raise ValueError("received batch dtype does not match the fixed slot")

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
        for name, tensor in (
            ("received hidden_states", batch.hidden_states),
            ("received expert_ids", batch.expert_ids),
            ("received routing_weights", batch.routing_weights),
        ):
            if not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
            if self._spec.device.type == "cuda" and not tensor.is_pinned():
                raise ValueError(f"{name} must use CUDA-registered shared memory")

        non_blocking = self._spec.device.type == "cuda"
        hidden_states.copy_(batch.hidden_states, non_blocking=non_blocking)
        expert_ids.copy_(batch.expert_ids, non_blocking=non_blocking)
        routing_weights.copy_(batch.routing_weights, non_blocking=non_blocking)

    def copy_output(
        self,
        partial_output: torch.Tensor,
        destination: torch.Tensor | None,
    ) -> torch.Tensor:
        """Copy output directly into the claimed shared-memory slot."""

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
            raise ValueError("SHM output requires its claimed shared-memory destination")
        _require_tensor(
            "output destination",
            destination,
            shape=(partial_output.shape[0], partial_output.shape[1]),
            dtype=self._spec.dtype,
            device=torch.device("cpu"),
        )
        if self._spec.device.type == "cuda" and not destination.is_pinned():
            raise ValueError("CUDA output destination must use CUDA-registered shared memory")
        destination.copy_(
            partial_output,
            non_blocking=self._spec.device.type == "cuda",
        )
        return destination

    def close(self) -> None:
        """Mark the batch buffers closed; the session owns all Host mappings."""

        self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("Worker batch buffers are closed")
