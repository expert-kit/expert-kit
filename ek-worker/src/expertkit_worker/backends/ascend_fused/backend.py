import threading
from abc import abstractmethod
from collections.abc import Callable
from copy import replace

import torch

from expertkit_worker.backends import BackendWeightUnavailable
from expertkit_worker.backends.base import (
    BackendBatch,
    BackendCapabilities,
    BackendCompletion,
    BackendResourceEstimate,
    ComputeBackend,
)
from expertkit_worker.device.runtime import DeviceWork, WorkerDeviceRuntime
from expertkit_worker.weights import ReadyWeightLease, WeightsNotReady

from .ops import ascend_fused_moe
from .weights import AscendFusedExpertWeights

type AcquireAscendFusedWeights = Callable[
    [int, tuple[int, ...]],
    ReadyWeightLease[AscendFusedExpertWeights],
]


class _AscendFusedCompletion(BackendCompletion):
    def __init__(
        self,
        lease: ReadyWeightLease[AscendFusedExpertWeights],
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


class AscendFusedBackend(ComputeBackend):
    """Execute gated experts with Fused MoE operations and FP32 reduction."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        intermediate_dim: int,
        top_k: int,
        dtype: torch.dtype,
        runtime: WorkerDeviceRuntime,
        acquire_many: AcquireAscendFusedWeights,
    ) -> None:
        self._hidden_dim = hidden_dim
        self._intermediate_dim = intermediate_dim
        self._top_k = top_k
        self._dtype = dtype
        self._runtime = runtime
        self._acquire_many = acquire_many
        self._capabilities = BackendCapabilities(
            supports_dynamic_tokens=True,
            supports_concurrent_batches=False,
        )

    @property
    def capabilities(self) -> BackendCapabilities:
        """Return dynamic-token and bounded-concurrency support."""
        return self._capabilities

    def estimate_resources(self, max_batch_tokens: int) -> BackendResourceEstimate:
        """Return a conservative temporary-memory estimate for startup planning."""
        topk_items = max_batch_tokens * self._top_k

        # Remapped IDs          A x 4
        expert_id_bytes = topk_items * torch.int32.itemsize

        # Valid mask            A x 1
        mask_bytes = topk_items * torch.bool.itemsize

        # Expanded hidden       A x H x B
        expanded_hidden_bytes = topk_items * self._hidden_dim * self._dtype.itemsize

        # Restoration indices   A x 8
        expanded_row_index_bytes = topk_items * torch.int32.itemsize

        # Group list            E x 8
        group_list_bytes = topk_items * torch.int64.itemsize

        # GMM1 output           A x 2I x B
        gmm1_output_bytes = topk_items * 2 * self._intermediate_dim * self._dtype.itemsize

        # SwiGLU output         A x I x B
        swiglu_output_bytes = topk_items * self._intermediate_dim * self._dtype.itemsize

        # GMM2 output           A x H x B
        gmm2_output_bytes = topk_items * self._hidden_dim * self._dtype.itemsize

        # restored output       T x H x B
        combined_output_bytes = max_batch_tokens * self._hidden_dim * self._dtype.itemsize

        return BackendResourceEstimate(
            temporary_bytes_per_active_batch=(
                expert_id_bytes
                + mask_bytes
                + expanded_hidden_bytes
                + expanded_row_index_bytes
                + group_list_bytes
                + gmm1_output_bytes
                + swiglu_output_bytes
                + gmm2_output_bytes
                + combined_output_bytes
            )
        )

    def submit(
        self,
        batch: BackendBatch,
        prepared_output: torch.Tensor,
    ) -> BackendCompletion:
        # TODO: validate batch
        lease = self._acquire(batch)
        try:
            # TODO: validate weights

            with torch.inference_mode():
                self._compute(batch, prepared_output, lease)

            work = self._runtime.capture_current_work()
            return _AscendFusedCompletion(lease, work)
        except BaseException:
            # TODO: handle failed submission
            raise

    def _acquire(
        self,
        batch: BackendBatch,
    ) -> ReadyWeightLease[AscendFusedExpertWeights]:
        try:
            return self._acquire_many(
                batch.layer_id,
                batch.distinct_expert_ids,
            )
        except WeightsNotReady as error:
            raise BackendWeightUnavailable(error.expert_ids) from error

    def _compute(
        self,
        batch: BackendBatch,
        prepared_output: torch.Tensor,
        lease: ReadyWeightLease[AscendFusedExpertWeights],
    ) -> None:
        w13 = tuple(w.w13 for w in lease.objects)
        w2 = tuple(w.w2 for w in lease.objects)

        local_expert_ids = _global_to_local(
            batch.expert_ids,
            lease.expert_ids,
        )

        output = ascend_fused_moe(
            hidden_states=batch.hidden_states,
            expert_ids=local_expert_ids,
            w13=w13,
            w2=w2,
        )

        prepared_output.copy_(output.output)


def _global_to_local(
    expert_ids: torch.Tensor,
    distinct_expert_ids: torch.Tensor,
) -> torch.Tensor:
    replace_id = max(expert_ids)
    local_expert_ids = torch.full_like(distinct_expert_ids, replace_id)
    local_expert_ids[expert_ids == -1] = replace_id

    return local_expert_ids
