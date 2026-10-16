import math
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
)
from expertkit_worker.device.runtime import DeviceWork, WorkerDeviceRuntime
from expertkit_worker.weights import ReadyWeightLease, WeightsNotReady

from .op import AscendFusedComputeType, run_ascend_fused_moe
from .weights import AscendFusedExpertWeights, AscendFusedModelSlimW8A8ExpertWeights

type AscendFusedReadyWeights = (
    AscendFusedExpertWeights | AscendFusedModelSlimW8A8ExpertWeights
)

type AcquireAscendFusedWeights = Callable[
    [int, tuple[int, ...]],
    ReadyWeightLease[AscendFusedReadyWeights],
]


class _AscendFusedCompletion(BackendCompletion):
    """Retain ready weights and submitted NPU resources until release."""

    def __init__(
        self,
        lease: ReadyWeightLease[AscendFusedReadyWeights],
        work: DeviceWork,
        retained_tensors: tuple[torch.Tensor, ...],
    ) -> None:
        self._lease = lease
        self._work = work
        self._retained_tensors = list(retained_tensors)
        self._lock = threading.Lock()
        self._waited = False
        self._failed = False
        self._closed = False

    def wait_host(self) -> None:
        """Wait for submitted device work and surface asynchronous failures."""
        try:
            self._work.wait_host()
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
        """Release ready weights after submitted NPU work is complete."""
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

        self._retained_tensors.clear()
        self._lease.close()

        if close_error is not None:
            raise BackendFatalError(
                BackendFatalReason.ASYNC_EXECUTION,
                str(close_error),
            ) from close_error


class AscendFusedBackend(ComputeBackend):
    """Execute gated experts with Ascend fused-MoE operations."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        intermediate_dim: int,
        top_k: int,
        dtype: torch.dtype,
        runtime: WorkerDeviceRuntime,
        acquire_many: AcquireAscendFusedWeights,
        compute_type: AscendFusedComputeType = AscendFusedComputeType.UNQUANTIZED,
        swiglu_limit: float = 0.0,
    ) -> None:
        for name, value in (
            ("hidden_dim", hidden_dim),
            ("intermediate_dim", intermediate_dim),
            ("top_k", top_k),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if dtype not in {torch.float16, torch.bfloat16}:
            raise ValueError("the Ascend fused Backend supports only FP16 or BF16")
        if runtime.device.type != "npu" or runtime.device.index is None:
            raise ValueError("the Ascend fused Backend requires one indexed NPU device")
        if not callable(acquire_many):
            raise TypeError("acquire_many must be callable")
        if not math.isfinite(swiglu_limit) or swiglu_limit < 0:
            raise ValueError("swiglu_limit must be finite and nonnegative")

        self._hidden_dim = hidden_dim
        self._intermediate_dim = intermediate_dim
        self._top_k = top_k
        self._dtype = dtype
        self._runtime = runtime
        self._acquire_many = acquire_many
        self._compute_type = AscendFusedComputeType(compute_type)
        self._swiglu_limit = swiglu_limit
        self._capabilities = BackendCapabilities(
            supports_dynamic_tokens=True,
            supports_concurrent_batches=True,
        )

    @property
    def capabilities(self) -> BackendCapabilities:
        """Return dynamic-token and bounded-concurrency support."""
        return self._capabilities

    def estimate_resources(self, max_batch_tokens: int) -> BackendResourceEstimate:
        """Return a conservative temporary-memory estimate for startup planning."""
        if (
            isinstance(max_batch_tokens, bool)
            or not isinstance(max_batch_tokens, int)
            or max_batch_tokens <= 0
        ):
            raise ValueError("max_batch_tokens must be a positive integer")
        topk_items = max_batch_tokens * self._top_k

        # Remapped IDs          A x 4
        expert_id_bytes = topk_items * torch.int32.itemsize

        # Sorted global IDs used by the device-side local-ID lookup, bounded by A.
        expert_lookup_bytes = topk_items * torch.int32.itemsize

        # Valid mask            A x 1
        mask_bytes = topk_items * torch.bool.itemsize

        # Routing probabilities cast from FP32 to the activation dtype.
        routing_weight_bytes = topk_items * self._dtype.itemsize

        # Dynamic routing quantizes the expanded activation and emits one FP32 scale per row.
        if self._compute_type is AscendFusedComputeType.DYNAMIC_W8A8:
            expanded_hidden_bytes = topk_items * self._hidden_dim * torch.int8.itemsize
            routing_scale_bytes = topk_items * torch.float32.itemsize
            activated_bytes = topk_items * self._intermediate_dim * torch.int8.itemsize
            activated_scale_bytes = topk_items * torch.float32.itemsize
        else:
            expanded_hidden_bytes = topk_items * self._hidden_dim * self._dtype.itemsize
            routing_scale_bytes = 0
            activated_bytes = topk_items * self._intermediate_dim * self._dtype.itemsize
            activated_scale_bytes = 0

        # Restoration indices plus the absolute-value view consumed by unpermute.
        expanded_row_index_bytes = topk_items * 2 * torch.int32.itemsize

        # Routing counts before and after conversion to GMM's int64 count form.
        group_list_bytes = topk_items * (torch.int32.itemsize + torch.int64.itemsize)

        # GMM1 output           A x 2I x B
        gmm1_output_bytes = topk_items * 2 * self._intermediate_dim * self._dtype.itemsize

        # GMM2 output           A x H x B
        gmm2_output_bytes = topk_items * self._hidden_dim * self._dtype.itemsize

        # restored output       T x H x B
        combined_output_bytes = max_batch_tokens * self._hidden_dim * self._dtype.itemsize

        return BackendResourceEstimate(
            temporary_bytes_per_active_batch=(
                expert_id_bytes
                + expert_lookup_bytes
                + mask_bytes
                + routing_weight_bytes
                + expanded_hidden_bytes
                + routing_scale_bytes
                + expanded_row_index_bytes
                + group_list_bytes
                + gmm1_output_bytes
                + activated_bytes
                + activated_scale_bytes
                + gmm2_output_bytes
                + combined_output_bytes
            )
        )

    def submit(
        self,
        batch: BackendBatch,
        prepared_output: torch.Tensor,
    ) -> BackendCompletion:
        """Launch one complete local MoE contribution into prepared_output."""
        lease = self._acquire(batch)
        try:
            self._validate_lease(lease, batch)

            with torch.inference_mode():
                retained_tensors = self._compute(batch, prepared_output, lease)

            work = self._runtime.capture_current_work()
            return _AscendFusedCompletion(lease, work, retained_tensors)
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

    def _acquire(
        self,
        batch: BackendBatch,
    ) -> ReadyWeightLease[AscendFusedReadyWeights]:
        try:
            return self._acquire_many(
                batch.layer_id,
                batch.distinct_expert_ids,
            )
        except WeightsNotReady as error:
            raise BackendWeightUnavailable(error.expert_ids) from error

    def _validate_lease(
        self,
        lease: ReadyWeightLease[AscendFusedReadyWeights],
        batch: BackendBatch,
    ) -> None:
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

    def _compute(
        self,
        batch: BackendBatch,
        prepared_output: torch.Tensor,
        lease: ReadyWeightLease[AscendFusedReadyWeights],
    ) -> tuple[torch.Tensor, ...]:
        if not lease.objects:
            prepared_output.zero_()
            return ()

        w13 = [w.w13 for w in lease.objects]
        w2 = [w.w2 for w in lease.objects]
        if self._compute_type is AscendFusedComputeType.DYNAMIC_W8A8:
            quantized_weights = tuple(
                weight
                for weight in lease.objects
                if isinstance(weight, AscendFusedModelSlimW8A8ExpertWeights)
            )
            w13_scale = [weight.w13_scale for weight in quantized_weights]
            w2_scale = [weight.w2_scale for weight in quantized_weights]
        else:
            w13_scale = None
            w2_scale = None

        inactive_mask = batch.expert_ids.eq(-1)
        # Remap global routing IDs to ordinal indices in the lease-ordered weight lists.
        local_expert_ids = _global_to_local(
            batch.expert_ids,
            lease.expert_ids,
            inactive_mask=inactive_mask,
        )
        routing_weights = batch.routing_weights.to(dtype=self._dtype)
        routing_weights.masked_fill_(inactive_mask, 0.0)

        output = run_ascend_fused_moe(
            hidden_states=batch.hidden_states,
            expert_ids=local_expert_ids,
            expert_weights=routing_weights,
            w13=w13,
            w2=w2,
            compute_type=self._compute_type,
            w13_scale=w13_scale,
            w2_scale=w2_scale,
            swiglu_limit=self._swiglu_limit,
        )

        prepared_output.copy_(output)
        return local_expert_ids, routing_weights, output

    def _finish_failed_submission(
        self,
        lease: ReadyWeightLease[AscendFusedReadyWeights],
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


def _global_to_local(
    expert_ids: torch.Tensor,
    distinct_expert_ids: tuple[int, ...],
    *,
    inactive_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Map sorted global IDs to lease ordinals without reading device data."""
    global_expert_ids = expert_ids.new_tensor(distinct_expert_ids)
    local_expert_ids = torch.searchsorted(
        global_expert_ids,
        expert_ids,
        out_int32=True,
    )
    if inactive_mask is None:
        inactive_mask = expert_ids.eq(-1)
    local_expert_ids.masked_fill_(inactive_mask, len(distinct_expert_ids))

    return local_expert_ids
