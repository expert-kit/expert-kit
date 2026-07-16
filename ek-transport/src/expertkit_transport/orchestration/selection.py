"""Replica selection policy used before Worker grouping."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from threading import Lock

from expertkit_transport.contracts import TransportError, TransportErrorCode
from expertkit_transport.orchestration.topology import WorkerIdentity, WorkerTarget


class ReplicaSelector(ABC):
    """Select one eligible ready replica for an expert."""

    @abstractmethod
    def select(
        self,
        *,
        instance_id: int,
        layer_id: int,
        expert_id: int,
        replicas: tuple[WorkerTarget, ...],
        excluded: frozenset[WorkerIdentity] = frozenset(),
    ) -> WorkerTarget:
        """Return one replica or raise a retryable unavailable error."""


class RoundRobinSelector(ReplicaSelector):
    """Rotate each expert independently across its ready replicas."""

    def __init__(self) -> None:
        self._next: defaultdict[tuple[int, int, int], int] = defaultdict(int)
        self._lock = Lock()

    def select(
        self,
        *,
        instance_id: int,
        layer_id: int,
        expert_id: int,
        replicas: tuple[WorkerTarget, ...],
        excluded: frozenset[WorkerIdentity] = frozenset(),
    ) -> WorkerTarget:
        eligible = tuple(target for target in replicas if target.identity not in excluded)
        if not eligible:
            raise TransportError(
                TransportErrorCode.UNAVAILABLE,
                retryable=True,
                unavailable_expert_ids=(expert_id,),
                diagnostic="no eligible ready replica",
            )

        key = (instance_id, layer_id, expert_id)
        with self._lock:
            index = self._next[key]
            self._next[key] = index + 1
        return eligible[index % len(eligible)]
