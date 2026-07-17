"""Dependency-free metrics boundary used by Worker components."""

from __future__ import annotations

from typing import Protocol


class WorkerMetrics(Protocol):
    """Record bounded Worker metrics without exposing an exporter dependency."""

    def batch_started(self) -> None:
        """Record one batch entering active execution."""

    def batch_finished(self, *, fatal: bool, duration_seconds: float) -> None:
        """Record one active batch completion without synchronizing its device."""

    def batch_rejected(self, reason: str) -> None:
        """Record one rejection using a bounded reason value."""

    def pending_batches_changed(self, count: int) -> None:
        """Set the current Transport-owned waiting count."""

    def weight_source_result(self, source: str, *, success: bool) -> None:
        """Record one attempted weight-source lookup."""

    def expert_state_changed(self, state: str) -> None:
        """Record one reportable expert-state transition."""

    def device_weight_bytes_changed(self, byte_count: int) -> None:
        """Set accounted ready-weight bytes on this Worker's device."""


class NoopWorkerMetrics:
    """Implement the metrics boundary without allocating exporter state."""

    def batch_started(self) -> None:
        pass

    def batch_finished(self, *, fatal: bool, duration_seconds: float) -> None:
        pass

    def batch_rejected(self, reason: str) -> None:
        pass

    def pending_batches_changed(self, count: int) -> None:
        pass

    def weight_source_result(self, source: str, *, success: bool) -> None:
        pass

    def expert_state_changed(self, state: str) -> None:
        pass

    def device_weight_bytes_changed(self, byte_count: int) -> None:
        pass
