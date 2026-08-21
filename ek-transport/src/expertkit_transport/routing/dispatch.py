"""Bounded asynchronous dispatch and FP32 partial-output aggregation."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass

import torch

from expertkit_transport.buffers import OutputPool
from expertkit_transport.errors import TransportError
from expertkit_transport.routing.grouping import WorkerBatchPlan
from expertkit_transport.routing.topology import WorkerIdentity
from expertkit_transport.tracing import Tracer, trace_span


@dataclass(frozen=True, slots=True)
class FailedWorkerBatch:
    """Retain one unfinished physical contribution for retry policy."""

    plan: WorkerBatchPlan
    error: TransportError


def _validate_accumulator(
    plan: WorkerBatchPlan, pool: OutputPool, accumulator: torch.Tensor
) -> None:
    batch = plan.batch
    if accumulator.ndim != 2 or accumulator.shape[1] != batch.hidden_dim:
        raise ValueError("accumulator must have shape [layer_tokens, hidden_dim]")
    if accumulator.dtype != torch.float32:
        raise ValueError("accumulator must use FP32")
    if accumulator.device != batch.hidden_states.device:
        raise ValueError("accumulator and Worker batch must be on the same device")
    if pool.hidden_dim != batch.hidden_dim:
        raise ValueError("output pool hidden dimension does not match the Worker batch")
    if pool.dtype != batch.hidden_states.dtype:
        raise ValueError("output pool dtype does not match the activation dtype")
    if pool.device != accumulator.device:
        raise ValueError("output pool and accumulator must be on the same device")
    if pool.max_batch_tokens < batch.token_count:
        raise ValueError("output pool is too small for the physical Worker batch")


async def _dispatch_plan(
    plan: WorkerBatchPlan,
    pool: OutputPool,
    accumulator: torch.Tensor,
    monotonic_deadline: float,
    *,
    tracer: Tracer | None,
    attempt: int,
) -> FailedWorkerBatch | None:
    try:
        with trace_span(
            tracer,
            "frontend.worker_batch",
            attributes=_plan_attributes(plan, attempt),
        ):
            async with pool.lease(monotonic_deadline=monotonic_deadline) as lease:
                await plan.target.transport.execute(
                    plan.batch,
                    lease.tensor[: plan.batch.token_count],
                    monotonic_deadline=monotonic_deadline,
                )
                partial = lease.tensor[: plan.batch.token_count]
                token_indices = plan.batch.token_indices
                if token_indices is None:
                    token_indices = torch.arange(
                        plan.batch.token_count,
                        dtype=torch.int64,
                        device=accumulator.device,
                    )
                with trace_span(tracer, "frontend.merge"):
                    accumulator.index_add_(0, token_indices, partial.to(torch.float32))
                lease.mark_consumed()
    except TransportError as error:
        return FailedWorkerBatch(plan=plan, error=error)
    return None


async def dispatch_complete_plan(
    plan: WorkerBatchPlan,
    *,
    monotonic_deadline: float,
    tracer: Tracer | None = None,
    attempt: int = 1,
) -> tuple[torch.Tensor | None, FailedWorkerBatch | None]:
    """Return one complete Worker result without FP32 scatter aggregation."""

    batch = plan.batch
    if batch.token_indices is not None:
        raise ValueError("a complete Worker plan must select every source token")
    result = torch.empty_like(batch.hidden_states)
    try:
        with trace_span(
            tracer,
            "frontend.worker_batch",
            attributes=_plan_attributes(plan, attempt),
        ):
            await plan.target.transport.execute(
                batch,
                result,
                monotonic_deadline=monotonic_deadline,
            )
    except TransportError as error:
        return None, FailedWorkerBatch(plan=plan, error=error)
    return result, None


async def dispatch_once(
    plans: tuple[WorkerBatchPlan, ...],
    pools: Mapping[WorkerIdentity, OutputPool],
    accumulator: torch.Tensor,
    *,
    monotonic_deadline: float,
    tracer: Tracer | None = None,
    attempt: int = 1,
) -> tuple[FailedWorkerBatch, ...]:
    """Dispatch every physical batch once and add successes exactly once.

    Expected Transport failures are returned for retry policy. Unexpected
    exceptions cancel sibling submissions and propagate after their cleanup.

    Args:
        plans: Physical Worker contributions from one grouping attempt.
        pools: Preallocated output pools indexed by Worker process start.
        accumulator: Zero-initialized or partially completed FP32 layer output.
        monotonic_deadline: Absolute `time.monotonic()` deadline shared by all
            contributions.

    Returns:
        Failed contributions in input order. Successful contributions have
        already been scattered into `accumulator` once.
    """

    failures: list[FailedWorkerBatch | None] = [None] * len(plans)

    async def run(index: int, plan: WorkerBatchPlan) -> None:
        try:
            pool = pools[plan.target.identity]
        except KeyError as error:
            raise RuntimeError("no output pool for selected Worker process") from error
        _validate_accumulator(plan, pool, accumulator)
        failures[index] = await _dispatch_plan(
            plan,
            pool,
            accumulator,
            monotonic_deadline,
            tracer=tracer,
            attempt=attempt,
        )

    async with asyncio.TaskGroup() as tasks:
        for index, plan in enumerate(plans):
            tasks.create_task(run(index, plan))
    return tuple(failure for failure in failures if failure is not None)


def _plan_attributes(plan: WorkerBatchPlan, attempt: int) -> dict[str, str | int]:
    batch = plan.batch
    identity = plan.target.identity
    return {
        "expertkit.instance_id": batch.instance_id,
        "expertkit.layer_id": batch.layer_id,
        "expertkit.topology_version": batch.topology_version,
        "expertkit.token_count": batch.token_count,
        "expertkit.assignment_count": batch.token_count * batch.top_k,
        "expertkit.expert_ids": ",".join(str(expert_id) for expert_id in batch.distinct_expert_ids),
        "expertkit.worker_id": identity.worker_id,
        "expertkit.worker_start_id": identity.start_id,
        "expertkit.transport": plan.target.transport.transport_name,
        "expertkit.attempt": attempt,
    }
