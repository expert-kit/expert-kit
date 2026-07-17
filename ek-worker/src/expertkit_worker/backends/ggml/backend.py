"""Experimental CPU-only GGML Worker-batch implementation."""

from __future__ import annotations

import ctypes
from collections.abc import Callable

import ggml
import torch
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
from expertkit_worker.backends.ggml.weights import _GGML_DTYPES, GgmlExpertWeights
from expertkit_worker.weights import ReadyWeightLease, WeightsNotReady

type AcquireGgmlWeights = Callable[
    [int, tuple[int, ...]],
    ReadyWeightLease[GgmlExpertWeights],
]

_GRAPH_NODE_CAPACITY = 32
_GRAPH_CONTEXT_BYTES = 32 * 1024


def _set_tensor_data(tensor: object, source: torch.Tensor) -> None:
    tensor.contents.data = ctypes.c_void_p(source.data_ptr())  # type: ignore[attr-defined]


def _allocate_tensor_data(tensor: object, retained: list[torch.Tensor]) -> torch.Tensor:
    storage = torch.empty(ggml.ggml_nbytes(tensor), dtype=torch.uint8)
    retained.append(storage)
    _set_tensor_data(tensor, storage)
    return storage


class _GgmlCompletion(BackendCompletion):
    """Retain ready weights until the synchronous result is released."""

    def __init__(self, lease: ReadyWeightLease[GgmlExpertWeights]) -> None:
        self._lease = lease
        self._closed = False

    def wait_host(self) -> None:
        """Return immediately because GGML computation completed in submit."""

    def close(self) -> None:
        """Release ready weights exactly once."""

        if self._closed:
            return
        self._closed = True
        self._lease.close()


class GgmlBackend(ComputeBackend):
    """Execute grouped single-expert FFNs through ggml-python on CPU."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        intermediate_dim: int,
        top_k: int,
        dtype: torch.dtype,
        cpu_threads: int,
        acquire_many: AcquireGgmlWeights,
    ) -> None:
        for name, value in (
            ("hidden_dim", hidden_dim),
            ("intermediate_dim", intermediate_dim),
            ("top_k", top_k),
            ("cpu_threads", cpu_threads),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if dtype not in ACTIVATION_DTYPES:
            raise ValueError("GGML Backend dtype must be FP16, BF16, or FP32")
        if not callable(acquire_many):
            raise TypeError("acquire_many must be callable")

        self._hidden_dim = hidden_dim
        self._intermediate_dim = intermediate_dim
        self._top_k = top_k
        self._dtype = dtype
        self._cpu_threads = cpu_threads
        self._acquire_many = acquire_many
        self._capabilities = BackendCapabilities(
            supports_dynamic_tokens=True,
            supports_concurrent_batches=False,
        )

    @property
    def capabilities(self) -> BackendCapabilities:
        """Return dynamic-token support with one active GGML batch."""

        return self._capabilities

    def estimate_resources(self, max_batch_tokens: int) -> BackendResourceEstimate:
        """Estimate grouped activations, graph buffers, and FP32 reduction."""

        if (
            isinstance(max_batch_tokens, bool)
            or not isinstance(max_batch_tokens, int)
            or max_batch_tokens <= 0
        ):
            raise ValueError("max_batch_tokens must be a positive integer")
        assignments = max_batch_tokens * self._top_k
        element_bytes = torch.empty((), dtype=self._dtype).element_size()
        index_bytes = assignments * 2 * 8
        gathered_and_output = assignments * self._hidden_dim * element_bytes * 2
        graph_intermediates = assignments * self._intermediate_dim * (4 * 3 + element_bytes)
        weighted_output = assignments * self._hidden_dim * 4
        accumulator = max_batch_tokens * self._hidden_dim * 4
        return BackendResourceEstimate(
            temporary_bytes_per_active_batch=(
                index_bytes
                + gathered_and_output
                + graph_intermediates
                + weighted_output
                + accumulator
                + _GRAPH_CONTEXT_BYTES
            )
        )

    def submit(
        self,
        batch: BackendBatch,
        prepared_output: torch.Tensor,
    ) -> BackendCompletion:
        """Compute every local assignment synchronously through GGML."""

        self._validate_batch(batch, prepared_output)
        lease = self._acquire(batch)
        try:
            self._validate_weights(lease, batch)
            with torch.inference_mode():
                self._compute(batch, prepared_output, lease)
            return _GgmlCompletion(lease)
        except BaseException:
            lease.close()
            raise

    def _validate_batch(self, batch: BackendBatch, prepared_output: torch.Tensor) -> None:
        if batch.hidden_dim != self._hidden_dim:
            raise InvalidBackendInput("batch hidden dimension does not match the GGML Backend")
        if batch.top_k != self._top_k:
            raise InvalidBackendInput("batch top_k does not match the GGML Backend")
        if batch.hidden_states.dtype != self._dtype:
            raise InvalidBackendInput("batch dtype does not match the GGML Backend")
        if batch.hidden_states.device.type != "cpu":
            raise InvalidBackendInput("the MVP GGML Backend requires CPU input")
        if prepared_output.shape != batch.hidden_states.shape:
            raise InvalidBackendInput("prepared output shape does not match hidden states")
        if prepared_output.dtype != self._dtype:
            raise InvalidBackendInput("prepared output dtype does not match the GGML Backend")
        if prepared_output.device.type != "cpu":
            raise InvalidBackendInput("the MVP GGML Backend requires CPU output")
        if not prepared_output.is_contiguous():
            raise InvalidBackendInput("prepared output must be contiguous")

    def _acquire(self, batch: BackendBatch) -> ReadyWeightLease[GgmlExpertWeights]:
        try:
            return self._acquire_many(batch.layer_id, batch.distinct_expert_ids)
        except WeightsNotReady as error:
            raise BackendWeightUnavailable(error.expert_ids) from error

    def _validate_weights(
        self,
        lease: ReadyWeightLease[GgmlExpertWeights],
        batch: BackendBatch,
    ) -> None:
        if lease.expert_ids != batch.distinct_expert_ids:
            raise RuntimeError("ready weight lookup returned different expert IDs")
        if len(lease.objects) != len(batch.distinct_expert_ids):
            raise RuntimeError("ready weight lookup returned the wrong object count")
        for weight in lease.objects:
            if not isinstance(weight, GgmlExpertWeights):
                raise RuntimeError("ready weight lookup returned a non-GGML object")
            if (
                weight.hidden_dim != self._hidden_dim
                or weight.intermediate_dim != self._intermediate_dim
                or weight.dtype != self._dtype
            ):
                raise BackendFatalError(
                    BackendFatalReason.UNEXPECTED,
                    "ready expert weight does not match the GGML Backend",
                )

    def _compute(
        self,
        batch: BackendBatch,
        prepared_output: torch.Tensor,
        lease: ReadyWeightLease[GgmlExpertWeights],
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
            expert_input = torch.index_select(batch.hidden_states, 0, token_indices).contiguous()
            expert_output = self._forward_expert(expert_input, weight)
            routing = batch.routing_weights[token_indices, route_indices]
            accumulator.index_add_(
                0,
                token_indices,
                expert_output.to(torch.float32) * routing.unsqueeze(1),
            )
        prepared_output.copy_(accumulator.to(batch.hidden_states.dtype))

    def _forward_expert(
        self,
        expert_input: torch.Tensor,
        weight: GgmlExpertWeights,
    ) -> torch.Tensor:
        context = ggml.ggml_init(
            ggml.ggml_init_params(
                mem_size=_GRAPH_CONTEXT_BYTES,
                mem_buffer=None,
                no_alloc=True,
            )
        )
        if not context:
            raise BackendFatalError(
                BackendFatalReason.DEVICE_OOM,
                "ggml failed to allocate graph metadata",
            )
        retained: list[torch.Tensor] = []
        try:
            ggml_dtype = _GGML_DTYPES[self._dtype]
            input_tensor = ggml.ggml_new_tensor_2d(
                context,
                ggml_dtype,
                self._hidden_dim,
                expert_input.shape[0],
            )
            _set_tensor_data(input_tensor, expert_input)

            up = ggml.ggml_mul_mat(context, weight.ggml_up, input_tensor)
            _allocate_tensor_data(up, retained)
            gate = ggml.ggml_mul_mat(context, weight.ggml_gate, input_tensor)
            _allocate_tensor_data(gate, retained)
            activated_gate = ggml.ggml_silu_inplace(context, gate)
            hidden_fp32 = ggml.ggml_mul_inplace(context, up, activated_gate)
            if self._dtype == torch.float32:
                hidden = hidden_fp32
            else:
                hidden = ggml.ggml_cast(context, hidden_fp32, ggml_dtype)
                _allocate_tensor_data(hidden, retained)

            output_fp32 = ggml.ggml_mul_mat(context, weight.ggml_down, hidden)
            output = torch.empty(
                (expert_input.shape[0], self._hidden_dim),
                dtype=self._dtype,
            )
            if self._dtype == torch.float32:
                _set_tensor_data(output_fp32, output)
                output_tensor = output_fp32
            else:
                _allocate_tensor_data(output_fp32, retained)
                output_tensor = ggml.ggml_cast(context, output_fp32, ggml_dtype)
                _set_tensor_data(output_tensor, output)

            graph = ggml.ggml_new_graph_custom(context, _GRAPH_NODE_CAPACITY, False)
            if not graph:
                raise MemoryError("ggml failed to allocate a computation graph")
            ggml.ggml_build_forward_expand(graph, output_tensor)
            plan = ggml.ggml_graph_plan(graph, self._cpu_threads, None)
            work = (ctypes.c_uint8 * plan.work_size)() if plan.work_size > 0 else None
            if work is not None:
                plan.work_data = ctypes.cast(work, ctypes.POINTER(ctypes.c_uint8))
            status = ggml.ggml_graph_compute(graph, ctypes.byref(plan))
            if status == ggml.GGML_STATUS_ALLOC_FAILED:
                raise BackendFatalError(
                    BackendFatalReason.DEVICE_OOM,
                    "ggml graph work allocation failed",
                )
            if status != ggml.GGML_STATUS_SUCCESS:
                raise BackendFatalError(
                    BackendFatalReason.UNEXPECTED,
                    f"ggml graph computation failed with status {status}",
                )
            return output
        except torch.OutOfMemoryError as error:
            raise BackendFatalError(BackendFatalReason.DEVICE_OOM, str(error)) from error
        except MemoryError as error:
            raise BackendFatalError(BackendFatalReason.DEVICE_OOM, str(error)) from error
        finally:
            ggml.ggml_free(context)
