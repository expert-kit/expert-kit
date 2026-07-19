"""Command-line entry point for one Python Worker process."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from collections.abc import Sequence

import structlog
from pydantic import ValidationError

from expertkit_worker.config import ConfigFileError, WorkerConfig, load_config
from expertkit_worker.factory import build_worker_application
from expertkit_worker.observability import configure_logging

logger = structlog.get_logger(__name__)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ek-worker")
    parser.add_argument(
        "--config",
        help="Path to the Python Worker YAML configuration; overrides EK_CONFIG",
    )
    return parser


async def run_config(config: WorkerConfig) -> None:
    """Build and run one already validated Worker configuration."""

    application = await build_worker_application(config)
    remove_signal_handlers = application.install_signal_handlers()
    try:
        await application.run()
    finally:
        remove_signal_handlers()


def main(argv: Sequence[str] | None = None) -> int:
    """Parse configuration, configure logging once, and run the Worker."""

    arguments = _parser().parse_args(argv)
    config_path = arguments.config or os.environ.get("EK_CONFIG")
    if not config_path:
        _parser().error("--config or EK_CONFIG is required")
    try:
        config = load_config(config_path)
    except (ConfigFileError, ValidationError) as error:
        logging.basicConfig(level=logging.ERROR)
        logging.getLogger(__name__).error("invalid Worker configuration: %s", error)
        return 2

    configure_logging(config.logging)
    try:
        asyncio.run(run_config(config))
    except KeyboardInterrupt:
        return 130
    except BaseException:
        logger.critical("worker_process_failed", exc_info=True)
        return 1
    return 0
