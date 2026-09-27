"""Dependency-free tracing boundary shared by Transport and Worker execution."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager, nullcontext
from contextvars import ContextVar
from functools import wraps
from inspect import iscoroutinefunction
from typing import Any, Literal, Protocol

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

    def set_current_attributes(self, attributes: Mapping[str, TraceAttribute]) -> None:
        """Attach scalar attributes to the current span."""

    def inject_context(self) -> tuple[tuple[str, str], ...]:
        """Return W3C context metadata for an outgoing RPC."""

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
        kind: Literal["internal", "client"] = "internal",
        attributes: Mapping[str, TraceAttribute] | None = None,
    ) -> AbstractContextManager[TraceSpan]:
        """Start a span and make it current for the context-manager lifetime."""


# A ContextVar follows task/thread context copies without a process-wide provider.
_active_tracer: ContextVar[Tracer | None] = ContextVar("expertkit_tracer", default=None)


@contextmanager
def use_tracer(tracer: Tracer | None) -> Iterator[None]:
    """Bind tracing to this operation and restore the previous scope on exit."""
    token = _active_tracer.set(tracer)
    try:
        yield
    finally:
        _active_tracer.reset(token)


def trace_span(
    name: str,
    *,
    attributes: Mapping[str, TraceAttribute] | None = None,
    kind: Literal["internal", "client"] = "internal",
) -> AbstractContextManager[TraceSpan | None]:
    """Time a host operation without importing optional dependencies when disabled."""
    tracer = _active_tracer.get()
    if tracer is None:
        return nullcontext()
    if kind == "client":
        return tracer.start_as_current_span(name, attributes=attributes, kind=kind)
    return tracer.start_as_current_span(name, attributes=attributes)


def traced(name: str) -> Callable:
    """Decorate a sync or async operation with a fresh span for each invocation."""

    def decorate(function: Callable) -> Callable:
        if iscoroutinefunction(function):

            @wraps(function)
            async def async_call(*args: Any, **kwargs: Any) -> Any:
                if _active_tracer.get() is None:
                    return await function(*args, **kwargs)
                with trace_span(name):
                    return await function(*args, **kwargs)

            return async_call

        @wraps(function)
        def sync_call(*args: Any, **kwargs: Any) -> Any:
            if _active_tracer.get() is None:
                return function(*args, **kwargs)
            with trace_span(name):
                return function(*args, **kwargs)

        return sync_call

    return decorate


def trace_attributes(attributes: Mapping[str, TraceAttribute]) -> None:
    """Set bounded attributes on the current span when tracing is activated."""
    tracer = _active_tracer.get()
    if tracer is not None:
        tracer.set_current_attributes(attributes)


def grpc_trace_metadata() -> tuple[tuple[str, str], ...]:
    """Propagate W3C context, including unsampled parents, to a Worker RPC."""
    tracer = _active_tracer.get()
    if tracer is None:
        return ()
    return tracer.inject_context()
