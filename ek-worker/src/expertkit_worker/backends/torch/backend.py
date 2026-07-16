"""Torch eager implementation of one complete Worker-batch computation."""

from __future__ import annotations

import threading
from collections.abc import Callable

import torch
import torch.nn.functional as functional
from expertkit_transport.contracts import ACTIVATION_DTYPES

from expertkit_worker.backends.base import (
    BackendBatch,
    BackendCapabilities,
    BackendCompletion,
    BackendFatalError,
    BackendFatalReason,
    BackendResourceEstimate,
    BackendWeightUnavailable,
    ComputeBackend,
    InvalidBackendInput,
)
from expertkit_worker.backends.torch.weights import TorchExpertWeights
from expertkit_worker.weights import ReadyWeightLease, WeightsNotReady

type AcquireTorchWeights = Callable[
    [int, tuple[int, ...]],
    ReadyWeightLease[TorchExpertWeights],
]


class _TorchCompletion(BackendCompletion):
    """Retain ready weights and the submitting CUDA stream until result release."""

    def __init__(
        self,
        lease: ReadyWeightLease[TorchExpertWeights],
        stream: torch.cuda.Stream | None,
    ) -> None:
        self._lease = lease
        self._stream = stream
        self._lock = threading.Lock()
        self._waited = stream is None
        self._failed = False
        self._closed = False

    def wait_host(self) -> None:
        """Wait for submitted Torch CUDA work and surface asynchronous failures."""

        stream = self._stream
        if stream is None:
            return
        try:
            stream.synchronize()
        except BaseException:
            with self._lock:
                self._failed = True
            raise
        with self._lock:
            self._waited = True

    def close(self) -> None:
        """Release ready weights after submitted Torch work is complete."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
            stream = self._stream
            waited = self._waited
            failed = self._failed

        close_error: BaseException | None = None
        if stream is not None and not waited and not failed:
            try:
                stream.synchronize()
            except BaseException as error:
                close_error = error
        self._lease.close()
        if close_error is not None:
            raise close_error


class TorchBackend(ComputeBackend):
    """Execute gated experts with eager Torch operations and FP32 reduction."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        intermediate_dim: int,
        top_k: int,
        dtype: torch.dtype,
        device: torch.device | str,
        acquire_many: AcquireTorchWeights,
    ) -> None:
        for name, value in (
            ("hidden_dim", hidden_dim),
            ("intermediate_dim", intermediate_dim),
            ("top_k", top_k),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if dtype not in ACTIVATION_DTYPES:
            raise ValueError("Torch Backend dtype must be FP16, BF16, or FP32")
        resolved_device = torch.device(device)
        if resolved_device.type not in {"cpu", "cuda"}:
            raise ValueError("Torch Backend device must be CPU or CUDA")
        if resolved_device.type == "cuda" and resolved_device.index is None:
            raise ValueError("Torch Backend CUDA device must include an index")
        if not callable(acquire_many):
            raise TypeError("acquire_many must be callable")

        self._hidden_dim = hidden_dim
        self._intermediate_dim = intermediate_dim
        self._top_k = top_k
        self._dtype = dtype
        self._device = resolved_device
        self._acquire_many = acquire_many
        self._capabilities = BackendCapabilities(
            supports_dynamic_tokens=True,
            supports_concurrent_batches=True,
        )

    @property
    def capabilities(self) -> BackendCapabilities:
        """Return dynamic-token and bounded-concurrency support."""

        return self._capabilities

    def estimate_resources(self, max_batch_tokens: int) -> BackendResourceEstimate:
        """Estimate eager grouping, FFN intermediates, and FP32 accumulation."""

        if (
            isinstance(max_batch_tokens, bool)
            or not isinstance(max_batch_tokens, int)
            or max_batch_tokens <= 0
        ):
            raise ValueError("max_batch_tokens must be a positive integer")
        assignments = max_batch_tokens * self._top_k
        element_bytes = torch.empty((), dtype=self._dtype).element_size()
        index_bytes = assignments * 2 * 8
        mask_bytes = max_batch_tokens * self._top_k
        gathered_and_output = assignments * self._hidden_dim * element_bytes * 2
        intermediate = assignments * self._intermediate_dim * element_bytes * 3
        weighted_output = assignments * self._hidden_dim * 4
        accumulator = max_batch_tokens * self._hidden_dim * 4
        return BackendResourceEstimate(
            temporary_bytes_per_active_batch=(
                index_bytes
                + mask_bytes
                + gathered_and_output
                + intermediate
                + weighted_output
                + accumulator
            )
        )

    def submit(
        self,
        batch: BackendBatch,
        prepared_output: torch.Tensor,
    ) -> BackendCompletion:
        """Compute every local expert assignment into caller-owned output."""

        self._validate_batch(batch, prepared_output)
        lease = self._acquire(batch)

        try:
            self._validate_weights(lease, batch)
            with torch.inference_mode():
                self._compute(batch, prepared_output, lease)
            stream = (
                torch.cuda.current_stream(self._device) if self._device.type == "cuda" else None
            )
            return _TorchCompletion(lease, stream)
        except BaseException:
            self._finish_failed_submission(lease)
            raise

    def _validate_batch(
        self,
        batch: BackendBatch,
        prepared_output: torch.Tensor,
    ) -> None:
        if batch.hidden_dim != self._hidden_dim:
            raise InvalidBackendInput("batch hidden dimension does not match the Torch Backend")
        if batch.top_k != self._top_k:
            raise InvalidBackendInput("batch top_k does not match the Torch Backend")
        if batch.hidden_states.dtype != self._dtype:
            raise InvalidBackendInput("batch dtype does not match the Torch Backend")
        if batch.hidden_states.device != self._device:
            raise InvalidBackendInput("batch device does not match the Torch Backend")
        if prepared_output.shape != batch.hidden_states.shape:
            raise InvalidBackendInput("prepared output shape does not match hidden states")
        if prepared_output.dtype != self._dtype:
            raise InvalidBackendInput("prepared output dtype does not match the Torch Backend")
        if prepared_output.device != self._device:
            raise InvalidBackendInput("prepared output device does not match the Torch Backend")
        if not prepared_output.is_contiguous():
            raise InvalidBackendInput("prepared output must be contiguous")

    def _acquire(
        self,
        batch: BackendBatch,
    ) -> ReadyWeightLease[TorchExpertWeights]:
        try:
            return self._acquire_many(batch.layer_id, batch.distinct_expert_ids)
        except WeightsNotReady as error:
            raise BackendWeightUnavailable(error.expert_ids) from error

    def _validate_weights(
        self,
        lease: ReadyWeightLease[TorchExpertWeights],
        batch: BackendBatch,
    ) -> None:
        if lease.expert_ids != batch.distinct_expert_ids:
            raise RuntimeError("ready weight lookup returned different expert IDs")
        if len(lease.objects) != len(batch.distinct_expert_ids):
            raise RuntimeError("ready weight lookup returned the wrong object count")
        for weight in lease.objects:
            if not isinstance(weight, TorchExpertWeights):
                raise RuntimeError("ready weight lookup returned a non-Torch object")
            if (
                weight.hidden_dim != self._hidden_dim
                or weight.intermediate_dim != self._intermediate_dim
            ):
                raise BackendFatalError(
                    BackendFatalReason.UNEXPECTED,
                    "ready expert weight shape does not match the Torch Backend",
                )
            if weight.dtype != self._dtype:
                raise BackendFatalError(
                    BackendFatalReason.UNEXPECTED,
                    "ready expert weight dtype does not match the Torch Backend",
                )
            if weight.device != self._device:
                raise BackendFatalError(
                    BackendFatalReason.UNEXPECTED,
                    "ready expert weight device does not match the Torch Backend",
                )

    @staticmethod
    def _compute(
        batch: BackendBatch,
        prepared_output: torch.Tensor,
        lease: ReadyWeightLease[TorchExpertWeights],
    ) -> None:
        if not lease.objects:
            prepared_output.zero_()
            return

        accumulator = torch.zeros_like(batch.hidden_states, dtype=torch.float32)
        for expert_id, weight in zip(lease.expert_ids, lease.objects, strict=True):
            coordinates = torch.nonzero(batch.expert_ids == expert_id, as_tuple=False)
            if coordinates.shape[0] == 0:
                raise InvalidBackendInput(
                    "distinct expert metadata does not match the routing tensor"
                )
            token_indices = coordinates[:, 0]
            route_indices = coordinates[:, 1]
            expert_input = torch.index_select(batch.hidden_states, 0, token_indices)
            gate = functional.linear(expert_input, weight.gate_proj)
            up = functional.linear(expert_input, weight.up_proj)
            expert_output = functional.linear(functional.silu(gate) * up, weight.down_proj)
            routing = batch.routing_weights[token_indices, route_indices]
            accumulator.index_add_(
                0,
                token_indices,
                expert_output.to(torch.float32) * routing.unsqueeze(1),
            )
        prepared_output.copy_(accumulator.to(batch.hidden_states.dtype))

    def _finish_failed_submission(
        self,
        lease: ReadyWeightLease[TorchExpertWeights],
    ) -> None:
        synchronization_error: BaseException | None = None
        if self._device.type == "cuda":
            try:
                torch.cuda.current_stream(self._device).synchronize()
            except BaseException as error:
                synchronization_error = error
        lease.close()
        if synchronization_error is not None:
            raise BackendFatalError(
                BackendFatalReason.ASYNC_EXECUTION,
                str(synchronization_error),
            ) from synchronization_error
