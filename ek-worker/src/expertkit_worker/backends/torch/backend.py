"""Torch eager implementation of one complete Worker-batch computation."""

from __future__ import annotations

import math
import threading
from collections.abc import Callable

import torch
import torch.nn.functional as functional
from expertkit_transport.batches import ACTIVATION_DTYPES
from expertkit_transport.tracing import trace_span

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
from expertkit_worker.backends.torch.fp4 import fp8_activation_reference
from expertkit_worker.backends.torch.modelslim_w8a8 import (
    TorchModelSlimW8A8Weights,
    modelslim_w8a8_linear,
)
from expertkit_worker.backends.torch.w8a8 import TorchW8A8Weights, w8a8_linear
from expertkit_worker.backends.torch.weights import TorchExpertWeights
from expertkit_worker.device import CpuWorkerRuntime, CudaWorkerRuntime, WorkerDeviceRuntime
from expertkit_worker.device.runtime import DeviceWork
from expertkit_worker.weights import ReadyWeightLease, WeightsNotReady

type TorchReadyWeights = TorchExpertWeights | TorchW8A8Weights | TorchModelSlimW8A8Weights

type AcquireTorchWeights = Callable[
    [int, tuple[int, ...]],
    ReadyWeightLease[TorchReadyWeights],
]


class _TorchCompletion(BackendCompletion):
    """Retain ready weights and the submitting device stream until result release."""

    def __init__(
        self,
        lease: ReadyWeightLease[TorchReadyWeights],
        work: DeviceWork,
    ) -> None:
        self._lease = lease
        self._work = work
        self._lock = threading.Lock()
        self._waited = False
        self._failed = False
        self._closed = False

    def wait_host(self) -> None:
        """Wait for submitted device work and surface asynchronous failures."""
        try:
            self._work.wait_host()
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
            waited = self._waited
            failed = self._failed

        close_error: BaseException | None = None
        if not waited and not failed:
            try:
                self._work.wait_host()
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
        runtime: WorkerDeviceRuntime | None = None,
        device: torch.device | str | None = None,
        acquire_many: AcquireTorchWeights,
        expert_compute: str = "swiglu",
        swiglu_limit: float = 0.0,
        linear_compute: str = "float",
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
        if not callable(acquire_many):
            raise TypeError("acquire_many must be callable")
        if runtime is None:
            if device is None:
                raise TypeError("TorchBackend requires runtime or device")
            parsed_device = torch.device(device)
            if parsed_device.type == "cpu":
                runtime = CpuWorkerRuntime(parsed_device)
            elif parsed_device.type == "cuda":
                runtime = CudaWorkerRuntime(parsed_device)
            elif parsed_device.type == "npu":
                from expertkit_worker.device.ascend import AscendWorkerRuntime

                runtime = AscendWorkerRuntime(parsed_device)
            else:
                raise ValueError(f"unsupported Torch device: {parsed_device}")

        if (
            expert_compute not in {"swiglu", "deepseek_v4"}
            or not math.isfinite(swiglu_limit)
            or swiglu_limit < 0
        ):
            raise ValueError("unsupported expert computation policy")
        linear_implementations = {
            "float": self._float_linear,
            "fp8_reference": self._fp8_linear,
            "w8a8": self._w8a8_linear,
            "modelslim_w8a8_dynamic": self._modelslim_w8a8_linear,
        }
        if linear_compute not in linear_implementations:
            raise ValueError("unsupported linear computation recipe")
        if linear_compute == "fp8_reference" and (hidden_dim % 128 or intermediate_dim % 128):
            raise ValueError("V4 FP8 activation widths must be divisible by 128")
        self._expert_compute = expert_compute
        self._swiglu_limit = swiglu_limit
        self._linear = linear_implementations[linear_compute]
        self._weight_type = {
            "w8a8": TorchW8A8Weights,
            "modelslim_w8a8_dynamic": TorchModelSlimW8A8Weights,
        }.get(linear_compute, TorchExpertWeights)
        self._finish_expert = (
            self._v4_output if expert_compute == "deepseek_v4" else self._swiglu_output
        )
        self._quantized_linear = linear_compute != "float"
        self._hidden_dim = hidden_dim
        self._intermediate_dim = intermediate_dim
        self._top_k = top_k
        self._dtype = dtype
        self._acquire_many = acquire_many
        self._runtime = runtime
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
                # Bound quantization, INT32 products, FP32 activation work, and row padding.
                + (
                    (max_batch_tokens + 32) * (self._hidden_dim + self._intermediate_dim) * 64
                    if self._quantized_linear or self._expert_compute == "deepseek_v4"
                    else 0
                )
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

            work = self._runtime.capture_current_work()
            return _TorchCompletion(lease, work)
        except BaseException:
            self._finish_failed_submission(lease)
            raise

    def _validate_batch(
        self,
        batch: BackendBatch,
        prepared_output: torch.Tensor,
    ) -> None:
        device = self._runtime.device
        if batch.hidden_dim != self._hidden_dim:
            raise InvalidBackendInput("batch hidden dimension does not match the Torch Backend")
        if batch.top_k != self._top_k:
            raise InvalidBackendInput("batch top_k does not match the Torch Backend")
        if batch.hidden_states.dtype != self._dtype:
            raise InvalidBackendInput("batch dtype does not match the Torch Backend")
        if batch.hidden_states.device != device:
            raise InvalidBackendInput("batch device does not match the Torch Backend")
        if prepared_output.shape != batch.hidden_states.shape:
            raise InvalidBackendInput("prepared output shape does not match hidden states")
        if prepared_output.dtype != self._dtype:
            raise InvalidBackendInput("prepared output dtype does not match the Torch Backend")
        if prepared_output.device != device:
            raise InvalidBackendInput("prepared output device does not match the Torch Backend")
        if not prepared_output.is_contiguous():
            raise InvalidBackendInput("prepared output must be contiguous")

    def _acquire(
        self,
        batch: BackendBatch,
    ) -> ReadyWeightLease[TorchReadyWeights]:
        try:
            return self._acquire_many(batch.layer_id, batch.distinct_expert_ids)
        except WeightsNotReady as error:
            raise BackendWeightUnavailable(error.expert_ids) from error

    def _validate_weights(
        self,
        lease: ReadyWeightLease[TorchReadyWeights],
        batch: BackendBatch,
    ) -> None:
        device = self._runtime.device
        if lease.expert_ids != batch.distinct_expert_ids:
            raise RuntimeError("ready weight lookup returned different expert IDs")
        if len(lease.objects) != len(batch.distinct_expert_ids):
            raise RuntimeError("ready weight lookup returned the wrong object count")
        for weight in lease.objects:
            if not isinstance(
                weight, TorchExpertWeights | TorchW8A8Weights | TorchModelSlimW8A8Weights
            ):
                raise RuntimeError("ready weight lookup returned a non-Torch object")
            if not isinstance(weight, self._weight_type):
                raise RuntimeError("ready weight quantization differs from the Backend recipe")
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
            if weight.device != device:
                raise BackendFatalError(
                    BackendFatalReason.UNEXPECTED,
                    "ready expert weight device does not match the Torch Backend",
                )

    @staticmethod
    def _float_linear(x: torch.Tensor, weight: TorchExpertWeights, projection: int) -> torch.Tensor:
        return functional.linear(x, weight.tensors[projection])

    @staticmethod
    def _fp8_linear(x: torch.Tensor, weight: TorchExpertWeights, projection: int) -> torch.Tensor:
        return functional.linear(fp8_activation_reference(x), weight.tensors[projection])

    @staticmethod
    def _w8a8_linear(x: torch.Tensor, weight: TorchW8A8Weights, projection: int) -> torch.Tensor:
        return w8a8_linear(x, weight.matrices[projection], weight.scales[projection])

    @staticmethod
    def _modelslim_w8a8_linear(
        x: torch.Tensor, weight: TorchModelSlimW8A8Weights, projection: int
    ) -> torch.Tensor:
        return modelslim_w8a8_linear(x, weight, projection)

    def _swiglu_output(
        self, gate: torch.Tensor, up: torch.Tensor, routing: torch.Tensor, weight: TorchReadyWeights
    ) -> torch.Tensor:
        intermediate = functional.silu(gate) * up
        return self._linear(intermediate, weight, 2).float() * routing

    def _v4_output(
        self, gate: torch.Tensor, up: torch.Tensor, routing: torch.Tensor, weight: TorchReadyWeights
    ) -> torch.Tensor:
        dtype = gate.dtype
        gate, up = gate.float(), up.float()
        if self._swiglu_limit > 0:
            gate = gate.clamp(max=self._swiglu_limit)
            up = up.clamp(min=-self._swiglu_limit, max=self._swiglu_limit)
        intermediate = (functional.silu(gate) * up * routing).to(dtype)
        return self._linear(intermediate, weight, 2).float()

    def _compute(
        self,
        batch: BackendBatch,
        prepared_output: torch.Tensor,
        lease: ReadyWeightLease[TorchReadyWeights],
    ) -> None:
        if not lease.objects:
            prepared_output.zero_()
            return

        accumulator = torch.zeros_like(batch.hidden_states, dtype=torch.float32)
        for expert_id, weight in zip(lease.expert_ids, lease.objects, strict=True):
            with trace_span(
                "worker.expert.submit",
                attributes={
                    "expertkit.layer_id": batch.layer_id,
                    "expertkit.expert_id": expert_id,
                    "expertkit.timing": "host_submission",
                },
            ):
                coordinates = torch.nonzero(batch.expert_ids == expert_id, as_tuple=False)
                if coordinates.shape[0] == 0:
                    raise InvalidBackendInput(
                        "distinct expert metadata does not match the routing tensor"
                    )
                token_indices = coordinates[:, 0]
                route_indices = coordinates[:, 1]
                expert_input = torch.index_select(batch.hidden_states, 0, token_indices)
                gate = self._linear(expert_input, weight, 0)
                up = self._linear(expert_input, weight, 1)
                routing = batch.routing_weights[token_indices, route_indices].unsqueeze(1)
                weighted = self._finish_expert(gate, up, routing, weight)
                accumulator.index_add_(0, token_indices, weighted)
        prepared_output.copy_(accumulator.to(batch.hidden_states.dtype))

    def _finish_failed_submission(
        self,
        lease: ReadyWeightLease[TorchReadyWeights],
    ) -> None:
        synchronization_error: BaseException | None = None
        try:
            self._runtime.capture_current_work().wait_host()
        except BaseException as error:
            synchronization_error = error
        lease.close()
        if synchronization_error is not None:
            raise BackendFatalError(
                BackendFatalReason.ASYNC_EXECUTION,
                str(synchronization_error),
            ) from synchronization_error
