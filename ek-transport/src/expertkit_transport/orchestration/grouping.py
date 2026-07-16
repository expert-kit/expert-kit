"""Resolve assignments, group them by Worker, and split physical batches."""

from __future__ import annotations

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


def _validate_routing_values(batch: RoutedLayerBatch) -> None:
    invalid_ids = batch.expert_ids < -1
    if torch.any(invalid_ids).item():
        raise _invalid_request("expert IDs below -1 are invalid")

    invalid_positions = batch.expert_ids == -1
    invalid_weights = invalid_positions & (batch.routing_weights != 0)
    if torch.any(invalid_weights).item():
        raise _invalid_request("an invalid expert position must have zero routing weight")


def _distinct_experts(expert_ids: torch.Tensor) -> tuple[int, ...]:
    valid = expert_ids[expert_ids >= 0]
    if valid.numel() == 0:
        return ()
    return tuple(sorted(torch.unique(valid).detach().cpu().tolist()))


def _split_batch(target: WorkerTarget, batch: WorkerBatch) -> tuple[WorkerBatchPlan, ...]:
    if batch.token_count <= target.max_batch_tokens:
        return (WorkerBatchPlan(target=target, batch=batch),)

    plans: list[WorkerBatchPlan] = []
    token_indices = batch.token_indices
    if token_indices is None:
        token_indices = torch.arange(
            batch.hidden_states.shape[0],
            device=batch.hidden_states.device,
            dtype=torch.int64,
        )
    for start in range(0, batch.token_count, target.max_batch_tokens):
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
            distinct_expert_ids=_distinct_experts(expert_ids),
        )
        plans.append(WorkerBatchPlan(target=target, batch=physical))
    return tuple(plans)


def group_worker_batches(
    batch: RoutedLayerBatch,
    topology: TopologySnapshot,
    selector: ReplicaSelector,
    *,
    excluded: frozenset[WorkerIdentity] = frozenset(),
) -> tuple[WorkerBatchPlan, ...]:
    """Resolve and group one Routed layer batch against one Topology snapshot.

    Selection chooses one replica per expert for this grouping attempt. Tensor
    masks and row compaction stay on the input device. Only the small list of
    distinct expert numbers is copied to Host metadata.

    Args:
        batch: Final model-router output.
        topology: Complete snapshot used for every assignment in this attempt.
        selector: Replica policy shared across Routed layer calls.
        excluded: Worker process starts that a retry should avoid.

    Returns:
        Physical batches, each no larger than its selected Worker's published
        `max_batch_tokens`.

    Raises:
        TransportError: The input is invalid or a valid assignment has no
            eligible ready route.
    """

    if batch.instance_id != topology.instance_id:
        raise _invalid_request("Routed layer and Topology instance IDs differ")
    _validate_routing_values(batch)

    routes = topology.layer_routes(batch.layer_id)
    if not routes:
        valid = batch.expert_ids[batch.expert_ids >= 0]
        missing = _distinct_experts(valid)
        if not missing:
            return ()
        raise TransportError(
            TransportErrorCode.UNAVAILABLE,
            retryable=True,
            unavailable_expert_ids=missing,
            diagnostic="the layer has no ready expert routes",
        )

    requested_experts = _distinct_experts(batch.expert_ids)
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
    for expert_id in requested_experts:
        replicas = routes[expert_id]
        target = selector.select(
            instance_id=batch.instance_id,
            layer_id=batch.layer_id,
            expert_id=expert_id,
            replicas=replicas,
            excluded=excluded,
        )
        selected_targets[expert_id] = target
        targets_by_identity[target.identity] = target

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
            distinct_expert_ids=_distinct_experts(selected_ids),
        )
        plans.extend(_split_batch(target, logical))
    return tuple(plans)
