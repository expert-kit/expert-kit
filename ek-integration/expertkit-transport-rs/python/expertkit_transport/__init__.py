"""
ExpertKit Transport Library
"""

import atexit
import logging

from ._lib import (
    PyExpertKitClient as ExpertKitClient,
    flush_tracing,
    shutdown_tracing,
)

logger = logging.getLogger(__name__)


def _shutdown_tracing_at_exit():
    try:
        shutdown_tracing()
    except Exception as exc:
        logger.debug("Failed to shutdown ExpertKit tracing at exit: %s", exc)


atexit.register(_shutdown_tracing_at_exit)

__version__ = "0.1.0"
__all__ = ["ExpertKitClient", "flush_tracing", "shutdown_tracing"]
