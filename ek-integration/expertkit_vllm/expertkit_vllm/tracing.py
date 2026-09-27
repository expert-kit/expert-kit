"""Opt-in model-forward tracing for the pinned vLLM forward context."""

from __future__ import annotations

import atexit
import os
from collections.abc import Iterator
from contextlib import contextmanager
from functools import wraps
from typing import Any

from expertkit_transport.observability import FrontendTracing
from expertkit_transport.tracing import trace_span, use_tracer


def install_forward_tracing() -> None:
    """Bracket full model forwards, with an exporter created in each engine process.

    Both vLLM and Ascend call the original module's override function. Wrapping
    it preserves already-imported aliases of set_forward_context. No model
    layer number or request ID is used to guess a batch boundary.
    """
    endpoint = os.getenv("EK_TRACE_ENDPOINT")
    if not endpoint:
        return
    import vllm.forward_context as context_module

    original = context_module.override_forward_context
    if getattr(original, "_expertkit_tracing", False):
        return
    sample_ratio = float(os.getenv("EK_TRACE_SAMPLE_RATIO", "1"))
    runtime: FrontendTracing | None = None
    process_id: int | None = None

    @wraps(original)
    @contextmanager
    def traced_forward(forward_context: Any) -> Iterator[None]:
        nonlocal runtime, process_id
        if runtime is None or process_id != os.getpid():
            runtime = FrontendTracing(endpoint, sample_ratio=sample_ratio)
            process_id = os.getpid()
            atexit.register(runtime.close)
        descriptor = getattr(forward_context, "batch_descriptor", None)
        num_tokens = getattr(descriptor, "num_tokens", None)
        attributes: dict[str, str | int] = {"expertkit.frontend": "vllm"}
        if isinstance(num_tokens, int):
            attributes["expertkit.token_count"] = num_tokens
        with (
            use_tracer(runtime.tracer),
            trace_span("frontend.model_forward", attributes=attributes),
            original(forward_context),
        ):
            yield

    traced_forward._expertkit_tracing = True
    context_module.override_forward_context = traced_forward
