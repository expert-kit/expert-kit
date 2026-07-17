"""Experimental fixed-slot Triton implementation of one Worker batch."""

from __future__ import annotations

import threading
from collections.abc import Callable

import torch

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
from expertkit_worker.backends.fused.kernels import FusedWorkspace, launch_fused_moe
from expertkit_worker.backends.fused.weights import (
    FusedExpertWeights,
    FusedWeightStorage,
)
from expertkit_worker.weights import ReadyWeightLease, WeightsNotReady

type AcquireFusedWeights = Callable[
    [int, tuple[int, ...]],
    ReadyWeightLease[FusedExpertWeights],
]

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16}


class _FusedCompletion(BackendCompletion):
    """Retain fixed slots and workspaces until submitted CUDA work is released."""

    def __init__(
        self,
        lease: ReadyWeightLease[FusedExpertWeights],
        workspace: FusedWorkspace | None,
        stream: torch.cuda.Stream,
    ) -> None:
        self._lease = lease
        self._workspace = workspace
        self._stream = stream
        self._lock = threading.Lock()
        self._waited = False
        self._failed = False
        self._closed = False

    def wait_host(self) -> None:
        """Wait for Triton work and classify asynchronous CUDA failures."""

        try:
            self._stream.synchronize()
        except BaseException as error:
            with self._lock:
                self._failed = True
            raise BackendFatalError(
                BackendFatalReason.ASYNC_EXECUTION,
                str(error),
            ) from error
        with self._lock:
            self._waited = True

    def close(self) -> None:
        """Release workspaces and fixed-slot usage exactly once."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
            waited = self._waited
            failed = self._failed

        close_error: BaseException | None = None
        if not waited and not failed:
            try:
                self._stream.synchronize()
            except BaseException as error:
                close_error = error
        self._workspace = None
        self._lease.close()
        if close_error is not None:
            raise BackendFatalError(
                BackendFatalReason.ASYNC_EXECUTION,
                str(close_error),
            ) from close_error


class FusedBackend(ComputeBackend):
    """Run the restricted unquantized SiLU MoE path with Triton kernels."""

    def __init__(
        self,
        *,
        num_layers: int,
        experts_per_layer: int,
        hidden_dim: int,
        intermediate_dim: int,
        top_k: int,
        dtype: torch.dtype,
        device: torch.device | str,
        acquire_many: AcquireFusedWeights,
    ) -> None:
        for name, value in (
            ("num_layers", num_layers),
            ("experts_per_layer", experts_per_layer),
            ("hidden_dim", hidden_dim),
            ("intermediate_dim", intermediate_dim),
            ("top_k", top_k),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if top_k > experts_per_layer:
            raise ValueError("top_k cannot exceed experts_per_layer")
        if dtype not in _SUPPORTED_DTYPES:
            raise ValueError("the fused Backend supports only FP16 or BF16")
        resolved_device = torch.device(device)
        if resolved_device.type != "cuda" or resolved_device.index is None:
            raise ValueError("the fused Backend requires one indexed NVIDIA CUDA device")
        if torch.version.hip is not None:
            raise ValueError("the fused Backend does not support ROCm")
        if not torch.cuda.is_available():
            raise RuntimeError("the fused Backend requires an available NVIDIA CUDA device")
        if not callable(acquire_many):
            raise TypeError("acquire_many must be callable")

        self._num_layers = num_layers
        self._experts_per_layer = experts_per_layer
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
        """Plan exact logical workspaces plus the persistent route-to-slot map."""

        if (
            isinstance(max_batch_tokens, bool)
            or not isinstance(max_batch_tokens, int)
            or max_batch_tokens <= 0
        ):
            raise ValueError("max_batch_tokens must be a positive integer")
        assignments = max_batch_tokens * self._top_k
        workspace_elements = assignments * (3 * self._intermediate_dim + self._hidden_dim)
        element_bytes = torch.empty((), dtype=self._dtype).element_size()
        return BackendResourceEstimate(
            temporary_bytes_per_active_batch=workspace_elements * element_bytes,
            shared_temporary_bytes=self._num_layers * self._experts_per_layer * 4,
        )

    def submit(
        self,
        batch: BackendBatch,
        prepared_output: torch.Tensor,
    ) -> BackendCompletion:
        """Launch fused MoE work directly into the caller's output Tensor."""

        self._validate_batch(batch, prepared_output)
        lease = self._acquire(batch)
        try:
            storage = self._validate_weights(lease, batch)
            with torch.inference_mode():
                workspace = (
                    None
                    if storage is None
                    else launch_fused_moe(
                        hidden_states=batch.hidden_states,
                        expert_ids=batch.expert_ids,
                        routing_weights=batch.routing_weights,
                        gate_up=storage.gate_up,
                        down=storage.down,
                        slot_mapping=storage.mapping_for_layer(batch.layer_id),
                        prepared_output=prepared_output,
                    )
                )
                if storage is None:
                    prepared_output.zero_()
            stream = torch.cuda.current_stream(self._device)
            return _FusedCompletion(lease, workspace, stream)
        except BaseException as error:
            self._finish_failed_submission(lease)
            if isinstance(error, BackendFatalError):
                raise
            if isinstance(error, torch.OutOfMemoryError):
                raise BackendFatalError(
                    BackendFatalReason.DEVICE_OOM,
                    str(error),
                ) from error
            raise BackendFatalError(
                BackendFatalReason.UNEXPECTED,
                str(error),
            ) from error

    def _validate_batch(
        self,
        batch: BackendBatch,
        prepared_output: torch.Tensor,
    ) -> None:
        if batch.layer_id >= self._num_layers:
            raise InvalidBackendInput("batch layer exceeds the fused Backend model shape")
        if batch.hidden_dim != self._hidden_dim:
            raise InvalidBackendInput("batch hidden dimension does not match the fused Backend")
        if batch.top_k != self._top_k:
            raise InvalidBackendInput("batch top_k does not match the fused Backend")
        if batch.hidden_states.dtype != self._dtype:
            raise InvalidBackendInput("batch dtype does not match the fused Backend")
        if batch.hidden_states.device != self._device:
            raise InvalidBackendInput("batch device does not match the fused Backend")
        if prepared_output.shape != batch.hidden_states.shape:
            raise InvalidBackendInput("prepared output shape does not match hidden states")
        if prepared_output.dtype != self._dtype:
            raise InvalidBackendInput("prepared output dtype does not match the fused Backend")
        if prepared_output.device != self._device:
            raise InvalidBackendInput("prepared output device does not match the fused Backend")
        if not prepared_output.is_contiguous():
            raise InvalidBackendInput("prepared output must be contiguous")

    def _acquire(
        self,
        batch: BackendBatch,
    ) -> ReadyWeightLease[FusedExpertWeights]:
        try:
            return self._acquire_many(batch.layer_id, batch.distinct_expert_ids)
        except WeightsNotReady as error:
            raise BackendWeightUnavailable(error.expert_ids) from error

    def _validate_weights(
        self,
        lease: ReadyWeightLease[FusedExpertWeights],
        batch: BackendBatch,
    ) -> FusedWeightStorage | None:
        if lease.expert_ids != batch.distinct_expert_ids:
            raise BackendFatalError(
                BackendFatalReason.UNEXPECTED,
                "ready weight lookup returned different expert IDs",
            )
        if len(lease.objects) != len(batch.distinct_expert_ids):
            raise BackendFatalError(
                BackendFatalReason.UNEXPECTED,
                "ready weight lookup returned the wrong object count",
            )
        storage: FusedWeightStorage | None = None
        for expert_id, weight in zip(lease.expert_ids, lease.objects, strict=True):
            if not isinstance(weight, FusedExpertWeights):
                raise BackendFatalError(
                    BackendFatalReason.UNEXPECTED,
                    "ready weight lookup returned a non-fused object",
                )
            if weight.layer_id != batch.layer_id or weight.expert_id != expert_id:
                raise BackendFatalError(
                    BackendFatalReason.UNEXPECTED,
                    "ready fused weight belongs to another model position",
                )
            if (
                weight.hidden_dim != self._hidden_dim
                or weight.intermediate_dim != self._intermediate_dim
                or weight.dtype != self._dtype
                or weight.device != self._device
            ):
                raise BackendFatalError(
                    BackendFatalReason.UNEXPECTED,
                    "ready fused weight does not match the Backend configuration",
                )
            if weight.storage.slot_for(batch.layer_id, expert_id) != weight.slot:
                raise BackendFatalError(
                    BackendFatalReason.UNEXPECTED,
                    "ready fused slot mapping changed before execution",
                )
            if storage is None:
                storage = weight.storage
            elif storage is not weight.storage:
                raise BackendFatalError(
                    BackendFatalReason.UNEXPECTED,
                    "ready fused weights do not share one fixed storage allocation",
                )
        return storage

    def _finish_failed_submission(
        self,
        lease: ReadyWeightLease[FusedExpertWeights],
    ) -> None:
        synchronization_error: BaseException | None = None
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
