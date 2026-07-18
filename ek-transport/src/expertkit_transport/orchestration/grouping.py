"""Resolve assignments, group them by Worker, and split physical batches."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from expertkit_transport.contracts import (
    RoutedLayerBatch,
    TransportError,
    TransportErrorCode,
    WorkerBatch,
)
from expertkit_transport.orchestration.selection import ReplicaSelector
from expertkit_transport.orchestration.topology import (
    TopologySnapshot,
    WorkerIdentity,
    WorkerTarget,
)


@dataclass(frozen=True, slots=True)
class WorkerBatchPlan:
    """Pair one physical batch with its selected Worker process."""

    target: WorkerTarget
    batch: WorkerBatch


def _invalid_request(diagnostic: str) -> TransportError:
    return TransportError(
        TransportErrorCode.INVALID_REQUEST,
        retryable=False,
        diagnostic=diagnostic,
    )


def _distinct_experts_by_chunk(
    expert_ids: torch.Tensor,
    chunk_size: int,
    distinct_expert_ids: tuple[int, ...],
) -> tuple[tuple[int, ...], ...]:
    """Return exact expert lists for every row chunk with one Host transfer."""

    chunk_count = (expert_ids.shape[0] + chunk_size - 1) // chunk_size
    if not distinct_expert_ids:
        return ((),) * chunk_count
    valid = expert_ids >= 0
    expert_width = distinct_expert_ids[-1] + 1
    rows = torch.arange(
        expert_ids.shape[0],
        dtype=torch.int64,
        device=expert_ids.device,
    )
    chunk_indices = torch.div(rows, chunk_size, rounding_mode="floor")
    chunk_indices = chunk_indices.unsqueeze(1).expand_as(expert_ids)
    safe_expert_ids = torch.where(valid, expert_ids, 0).to(dtype=torch.int64)
    flat_indices = chunk_indices * expert_width + safe_expert_ids
    counts = torch.zeros(
        chunk_count * expert_width,
        dtype=torch.int32,
        device=expert_ids.device,
    )
    counts.scatter_add_(
        0,
        flat_indices.flatten(),
        valid.flatten().to(dtype=torch.int32),
    )
    host_counts = counts.reshape(chunk_count, expert_width).detach().cpu().tolist()
    return tuple(
        tuple(expert_id for expert_id, count in enumerate(chunk) if count) for chunk in host_counts
    )


def _split_batch(target: WorkerTarget, batch: WorkerBatch) -> tuple[WorkerBatchPlan, ...]:
    if batch.token_count <= target.max_batch_tokens:
        return (WorkerBatchPlan(target=target, batch=batch),)

    plans: list[WorkerBatchPlan] = []
    distinct_by_chunk = _distinct_experts_by_chunk(
        batch.expert_ids,
        target.max_batch_tokens,
        batch.distinct_expert_ids,
    )
    token_indices = batch.token_indices
    if token_indices is None:
        token_indices = torch.arange(
            batch.hidden_states.shape[0],
            device=batch.hidden_states.device,
            dtype=torch.int64,
        )
    for chunk_index, start in enumerate(range(0, batch.token_count, target.max_batch_tokens)):
        stop = min(start + target.max_batch_tokens, batch.token_count)
        expert_ids = batch.expert_ids[start:stop]
        routing_weights = batch.routing_weights[start:stop]
        physical = WorkerBatch(
            instance_id=batch.instance_id,
            layer_id=batch.layer_id,
            topology_version=batch.topology_version,
            hidden_states=batch.hidden_states,
            token_indices=token_indices[start:stop],
            expert_ids=expert_ids,
            routing_weights=routing_weights,
            distinct_expert_ids=distinct_by_chunk[chunk_index],
        )
        plans.append(WorkerBatchPlan(target=target, batch=physical))
    return tuple(plans)


def group_worker_batches(
    batch: RoutedLayerBatch,
    topology: TopologySnapshot,
    selector: ReplicaSelector,
    *,
    excluded: frozenset[WorkerIdentity] = frozenset(),
    excluded_by_expert: Mapping[int, frozenset[WorkerIdentity]] | None = None,
    fallback_to_excluded: bool = False,
    reuse_complete_tensors: bool = True,
) -> tuple[WorkerBatchPlan, ...]:
    """Resolve and group one Routed layer batch against one Topology snapshot.

    Selection chooses one replica per expert for this grouping attempt. The
    Frontend-provided expert list avoids reading device routing values again.
    Multi-Worker masks and row compaction stay on the input device. Oversized
    batches copy one expert-count table for all physical chunks.

    Args:
        batch: Final model-router output.
        topology: Complete snapshot used for every assignment in this attempt.
        selector: Replica policy shared across Routed layer calls.
        excluded: Worker process starts that every expert should avoid.
        excluded_by_expert: Additional failed process starts to avoid for each
            expert independently.
        fallback_to_excluded: Use an excluded process only when that expert has
            no preferred replica.
        reuse_complete_tensors: Reuse the complete logical routing Tensors when
            every selected expert uses one Worker. Retry batches disable this so
            only unfinished token rows are sent again.

    Returns:
        Physical batches, each no larger than its selected Worker's published
        `max_batch_tokens`.

    Raises:
        TransportError: The input is invalid or a valid assignment has no
            eligible ready route.
    """

    if batch.instance_id != topology.instance_id:
        raise _invalid_request("Routed layer and Topology instance IDs differ")
    routes = topology.layer_routes(batch.layer_id)
    if not routes:
        missing = batch.distinct_expert_ids
        if not missing:
            return ()
        raise TransportError(
            TransportErrorCode.UNAVAILABLE,
            retryable=True,
            unavailable_expert_ids=missing,
            diagnostic="the layer has no ready expert routes",
        )

    requested_experts = batch.distinct_expert_ids
    if not requested_experts:
        return ()
    missing_experts = tuple(expert_id for expert_id in requested_experts if expert_id not in routes)
    if missing_experts:
        raise TransportError(
            TransportErrorCode.UNAVAILABLE,
            retryable=True,
            unavailable_expert_ids=missing_experts,
            diagnostic="one or more experts have no ready route",
        )

    selected_targets: dict[int, WorkerTarget] = {}
    targets_by_identity: dict[WorkerIdentity, WorkerTarget] = {}
    experts_by_identity: dict[WorkerIdentity, list[int]] = {}
    for expert_id in requested_experts:
        replicas = routes[expert_id]
        expert_excluded = excluded
        if excluded_by_expert is not None:
            expert_excluded = excluded | excluded_by_expert.get(expert_id, frozenset())
        target = selector.select(
            instance_id=batch.instance_id,
            layer_id=batch.layer_id,
            expert_id=expert_id,
            replicas=replicas,
            excluded=expert_excluded,
            fallback_to_excluded=fallback_to_excluded,
        )
        selected_targets[expert_id] = target
        targets_by_identity[target.identity] = target
        experts_by_identity.setdefault(target.identity, []).append(expert_id)

    if reuse_complete_tensors and len(targets_by_identity) == 1:
        target = next(iter(targets_by_identity.values()))
        logical = WorkerBatch(
            instance_id=batch.instance_id,
            layer_id=batch.layer_id,
            topology_version=topology.version,
            hidden_states=batch.hidden_states,
            token_indices=None,
            expert_ids=batch.expert_ids,
            routing_weights=batch.routing_weights,
            distinct_expert_ids=requested_experts,
        )
        return _split_batch(target, logical)

    max_expert_id = max(selected_targets)
    target_index_by_identity = {
        identity: index for index, identity in enumerate(sorted(targets_by_identity))
    }
    targets = tuple(targets_by_identity[identity] for identity in sorted(targets_by_identity))
    expert_target_values = [-1] * (max_expert_id + 1)
    for expert_id, target in selected_targets.items():
        expert_target_values[expert_id] = target_index_by_identity[target.identity]
    expert_target_indices = torch.tensor(
        expert_target_values,
        dtype=torch.int64,
        device=batch.expert_ids.device,
    )

    valid = batch.expert_ids >= 0
    safe_expert_ids = torch.where(valid, batch.expert_ids, 0).to(torch.int64)
    assignment_targets = expert_target_indices[safe_expert_ids]
    assignment_targets = torch.where(valid, assignment_targets, -1)

    plans: list[WorkerBatchPlan] = []
    invalid_expert = torch.full((), -1, dtype=torch.int32, device=batch.expert_ids.device)
    zero_weight = torch.zeros((), dtype=torch.float32, device=batch.routing_weights.device)
    for target_index, target in enumerate(targets):
        target_assignments = assignment_targets == target_index
        token_indices = torch.nonzero(
            torch.any(target_assignments, dim=1), as_tuple=False
        ).flatten()
        if token_indices.numel() == 0:
            continue
        selected_assignments = target_assignments[token_indices]
        selected_ids = torch.where(
            selected_assignments,
            batch.expert_ids[token_indices],
            invalid_expert,
        )
        selected_weights = torch.where(
            selected_assignments,
            batch.routing_weights[token_indices],
            zero_weight,
        )
        logical = WorkerBatch(
            instance_id=batch.instance_id,
            layer_id=batch.layer_id,
            topology_version=topology.version,
            hidden_states=batch.hidden_states,
            token_indices=token_indices,
            expert_ids=selected_ids,
            routing_weights=selected_weights,
            distinct_expert_ids=tuple(experts_by_identity[target.identity]),
        )
        plans.extend(_split_batch(target, logical))
    return tuple(plans)
