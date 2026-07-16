"""Configure structlog and standard logging once at the process entry point."""

from __future__ import annotations

import logging
import sys
from collections.abc import Sequence
from typing import IO, Any

import structlog

from expertkit_worker.config.models import LogFormat, LoggingConfig


def _shared_processors() -> Sequence[Any]:
    return (
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    )


def configure_logging(config: LoggingConfig, *, stream: IO[str] | None = None) -> None:
    """Install the process-wide standard logging and structlog configuration.

    Args:
        config: Validated threshold and renderer selection.
        stream: Optional destination used by tests and embedding processes.

    Note:
        Call this once from the process entry point before constructing service modules.
        Reconfiguration replaces root handlers so repeated test setup does not duplicate events.
    """

    renderer: Any
    if config.format is LogFormat.JSON:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=False)

    shared_processors = _shared_processors()
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=(
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ),
    )
    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(config.level.value)

    structlog.configure(
        processors=(
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
