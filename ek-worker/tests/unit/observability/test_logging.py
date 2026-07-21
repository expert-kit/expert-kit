"""Tests for process-wide structured logging configuration."""

import json
import logging
import re
from io import StringIO

import structlog

from expertkit_worker.config.models import LoggingConfig
from expertkit_worker.observability import configure_logging


def test_structlog_renders_stable_json_fields() -> None:
    output = StringIO()
    configure_logging(LoggingConfig(format="json"), stream=output)

    structlog.get_logger("worker.test").info(
        "worker_started",
        service="worker",
        worker_id="worker-0",
        backend="torch",
        device="cuda:0",
    )

    event = json.loads(output.getvalue())
    assert event["event"] == "worker_started"
    assert event["level"] == "info"
    assert event["logger"] == "worker.test"
    assert event["service"] == "worker"
    assert event["worker_id"] == "worker-0"
    assert event["backend"] == "torch"
    assert event["device"] == "cuda:0"
    assert "timestamp" in event


def test_filters_debug_events_at_default_level() -> None:
    output = StringIO()
    configure_logging(LoggingConfig(), stream=output)

    structlog.get_logger("worker.test").debug("batch_diagnostic", token_count=4)

    assert output.getvalue() == ""


def test_standard_library_logs_use_the_same_renderer() -> None:
    output = StringIO()
    configure_logging(LoggingConfig(format="json"), stream=output)

    logging.getLogger("external.library").warning("temporary failure")

    event = json.loads(output.getvalue())
    assert event["event"] == "temporary failure"
    assert event["level"] == "warning"
    assert event["logger"] == "external.library"


def test_default_console_renderer_matches_rust_service_layout() -> None:
    output = StringIO()
    configure_logging(LoggingConfig(), stream=output)

    structlog.get_logger("worker.test").info(
        "expert_loading_progress",
        worker_id="worker-0",
        ready_experts=64,
        assigned_experts=128,
    )

    rendered = output.getvalue().strip()
    assert re.fullmatch(
        r"<INFO>\([^)]*Z\) Expert loading progress "
        r"worker_id=worker-0 ready_experts=64 assigned_experts=128",
        rendered,
    )


def test_reconfiguration_does_not_duplicate_handlers() -> None:
    output = StringIO()
    config = LoggingConfig()
    configure_logging(config, stream=output)
    configure_logging(config, stream=output)

    structlog.get_logger("worker.test").info("worker_started")

    assert len(output.getvalue().splitlines()) == 1
