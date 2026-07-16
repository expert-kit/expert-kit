"""Execute one Routed-MoE layer with bounded direct retry."""

from __future__ import annotations

import asyncio
import math
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable, Mapping

import torch

from expertkit_transport.buffers import OutputPool
from expertkit_transport.contracts import (
    RoutedLayerBatch,
    TransportError,
    TransportErrorCode,
)
from expertkit_transport.orchestration.dispatch import FailedWorkerBatch, dispatch_once
from expertkit_transport.orchestration.grouping import (
    WorkerBatchPlan,
    group_worker_batches,
)
from expertkit_transport.orchestration.selection import ReplicaSelector
from expertkit_transport.orchestration.topology import (
    TopologyProvider,
    WorkerIdentity,
)

_DEFAULT_SAME_WORKER_RETRY_DELAY_SECONDS = 0.001


def _deadline_error(diagnostic: str) -> TransportError:
    return TransportError(
        TransportErrorCode.DEADLINE_EXCEEDED,
        retryable=False,
        diagnostic=diagnostic,
    )


def _unfinished_batch(
    source: RoutedLayerBatch,
    failures: tuple[FailedWorkerBatch, ...],
) -> tuple[RoutedLayerBatch, Mapping[int, frozenset[WorkerIdentity]]]:
    expert_ids = torch.full_like(source.expert_ids, -1)
    routing_weights = torch.zeros_like(source.routing_weights)
    failed_workers: defaultdict[int, set[WorkerIdentity]] = defaultdict(set)

    for failure in failures:
        physical = failure.plan.batch
        token_indices = physical.token_indices
        if token_indices is None:
            token_indices = torch.arange(
                physical.token_count,
                dtype=torch.int64,
                device=source.hidden_states.device,
            )

        existing_ids = expert_ids.index_select(0, token_indices)
        existing_weights = routing_weights.index_select(0, token_indices)
        valid = physical.expert_ids >= 0
        expert_ids.index_copy_(
            0,
            token_indices,
            torch.where(valid, physical.expert_ids, existing_ids),
        )
        routing_weights.index_copy_(
            0,
            token_indices,
            torch.where(valid, physical.routing_weights, existing_weights),
        )
        for expert_id in physical.distinct_expert_ids:
            failed_workers[expert_id].add(failure.plan.target.identity)

    excluded = {
        expert_id: frozenset(identities) for expert_id, identities in failed_workers.items()
    }
    return (
        RoutedLayerBatch(
            instance_id=source.instance_id,
            layer_id=source.layer_id,
            hidden_states=source.hidden_states,
            expert_ids=expert_ids,
            routing_weights=routing_weights,
        ),
        excluded,
    )


def _uses_failed_process(
    plan: WorkerBatchPlan,
    failed_workers: Mapping[int, frozenset[WorkerIdentity]],
) -> bool:
    return any(
        plan.target.identity in failed_workers.get(expert_id, frozenset())
        for expert_id in plan.batch.distinct_expert_ids
    )


def _required_topology_version(
    failures: tuple[FailedWorkerBatch, ...],
) -> int | None:
    required = tuple(
        failure.error.min_topology_version
        for failure in failures
        if failure.error.min_topology_version is not None
    )
    return max(required, default=None)


def _unavailable_before_required_version(
    failures: tuple[FailedWorkerBatch, ...],
    required_version: int,
) -> TransportError:
    unavailable = tuple(
        sorted(
            {
                expert_id
                for failure in failures
                for expert_id in failure.plan.batch.distinct_expert_ids
            }
        )
    )
    return TransportError(
        TransportErrorCode.UNAVAILABLE,
        retryable=True,
        min_topology_version=required_version,
        unavailable_expert_ids=unavailable,
        diagnostic="the required replacement topology is not available",
    )


async def execute_routed_layer(
    batch: RoutedLayerBatch,
    topology: TopologyProvider,
    selector: ReplicaSelector,
    pools: Mapping[WorkerIdentity, OutputPool],
    *,
    monotonic_deadline: float,
    same_worker_retry_delay_seconds: float = _DEFAULT_SAME_WORKER_RETRY_DELAY_SECONDS,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> torch.Tensor:
    """Return one aggregated Routed-MoE layer result.

    Successful Worker partials are accumulated once in FP32. On a retryable
    Worker failure, only unfinished assignments are resolved again after one
    Topology refresh. The second attempt prefers a different process. Reusing a
    failed process is allowed only when no replacement exists and after the
    configured nonzero delay. The final result is cast once to the activation
    dtype.

    `pools` must already contain fixed output pools for every Worker that the
    Topology provider may publish. Installing a Topology and its pools is one
    control-plane operation; this function never allocates a pool on the hot
    path.

    Args:
        batch: Final assignments for one Routed-MoE layer.
        topology: Source of complete, atomically installed routing snapshots.
        selector: Ready-replica selection policy.
        pools: Preallocated output pools indexed by Worker process start.
        monotonic_deadline: Absolute deadline for grouping, refresh, and both
            direct attempts.
        same_worker_retry_delay_seconds: Delay before retrying a failed process.
        clock: Monotonic clock, replaceable for deterministic tests.
        sleep: Asynchronous delay function, replaceable for deterministic tests.

    Returns:
        Tensor shaped `[token_count, hidden_dim]` on the activation device and
        using the activation dtype.

    Raises:
        TransportError: Routing or a bounded direct attempt failed.
    """

    if not math.isfinite(same_worker_retry_delay_seconds) or same_worker_retry_delay_seconds <= 0:
        raise ValueError("same_worker_retry_delay_seconds must be finite and positive")
    if monotonic_deadline - clock() <= 0:
        raise _deadline_error("the Routed layer deadline expired before dispatch")

    snapshot = topology.current(batch.instance_id)
    plans = group_worker_batches(batch, snapshot, selector)
    accumulator = torch.zeros(
        (batch.token_count, batch.hidden_dim),
        dtype=torch.float32,
        device=batch.hidden_states.device,
    )
    if not plans:
        return accumulator.to(batch.hidden_states.dtype)

    failures = await dispatch_once(
        plans,
        pools,
        accumulator,
        monotonic_deadline=monotonic_deadline,
    )
    if not failures:
        return accumulator.to(batch.hidden_states.dtype)

    nonretryable = next(
        (failure.error for failure in failures if not failure.error.retryable),
        None,
    )
    if nonretryable is not None:
        raise nonretryable

    unfinished, failed_workers = _unfinished_batch(batch, failures)
    refreshed = await topology.refresh(
        batch.instance_id,
        observed_version=snapshot.version,
        monotonic_deadline=monotonic_deadline,
    )
    if monotonic_deadline - clock() <= 0:
        raise _deadline_error("the Routed layer deadline expired during Topology refresh")

    required_version = _required_topology_version(failures)
    if required_version is not None and refreshed.version < required_version:
        raise _unavailable_before_required_version(failures, required_version)

    retry_plans = group_worker_batches(
        unfinished,
        refreshed,
        selector,
        excluded_by_expert=failed_workers,
        fallback_to_excluded=True,
    )
    if any(_uses_failed_process(plan, failed_workers) for plan in retry_plans):
        remaining = monotonic_deadline - clock()
        if remaining <= same_worker_retry_delay_seconds:
            raise failures[-1].error
        await sleep(same_worker_retry_delay_seconds)
        if monotonic_deadline - clock() <= 0:
            raise _deadline_error("the Routed layer deadline expired before its retry")

    retry_failures = await dispatch_once(
        retry_plans,
        pools,
        accumulator,
        monotonic_deadline=monotonic_deadline,
    )
    if retry_failures:
        raise retry_failures[-1].error
    return accumulator.to(batch.hidden_states.dtype)
