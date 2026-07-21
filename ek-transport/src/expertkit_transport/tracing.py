"""Dependency-free tracing boundary shared by Transport and Worker execution."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager
from typing import Protocol

type TraceAttribute = str | bool | int | float
type TraceContext = object


class TraceSpan(Protocol):
    """Expose the span operations used on the computation path."""

    def is_recording(self) -> bool:
        """Return whether this sampled span records attributes and events."""

    def set_attribute(self, key: str, value: TraceAttribute) -> None:
        """Attach one bounded scalar value to the span."""

    def end(self) -> None:
        """Finish a detached span exactly once."""


class Tracer(Protocol):
    """Create spans without exposing an OpenTelemetry package dependency."""

    def current_span_is_recording(self) -> bool:
        """Return whether the current request span was sampled."""

    def capture_context(self) -> TraceContext:
        """Capture the current parent context for another task or thread."""

    def start_span(
        self,
        name: str,
        *,
        context: TraceContext | None = None,
        attributes: Mapping[str, TraceAttribute] | None = None,
    ) -> TraceSpan:
        """Start a detached span that the caller will end."""

    def start_as_current_span(
        self,
        name: str,
        *,
        context: TraceContext | None = None,
        attributes: Mapping[str, TraceAttribute] | None = None,
    ) -> AbstractContextManager[TraceSpan]:
        """Start a span and make it current for the context-manager lifetime."""
