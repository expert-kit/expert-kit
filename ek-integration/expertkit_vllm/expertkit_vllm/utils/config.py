"""Validated process configuration for the vLLM integration."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EkClientConfig:
    """Hold the Controller route and model instance used by this vLLM process."""

    controller_endpoint: str
    instance_id: int
    timeout_seconds: float


def collect_ek_client_config() -> EkClientConfig:
    """Read and validate the small configuration surface exposed to vLLM."""

    endpoint = os.getenv("EK_ADDR", "localhost:5002").strip()
    if not endpoint:
        raise ValueError("EK_ADDR must not be empty")
    try:
        instance_id = int(os.environ["EK_INSTANCE_ID"])
    except KeyError as error:
        raise ValueError("EK_INSTANCE_ID must be set") from error
    except ValueError as error:
        raise ValueError("EK_INSTANCE_ID must be an integer") from error
    if instance_id <= 0:
        raise ValueError("EK_INSTANCE_ID must be positive")
    try:
        timeout_seconds = float(os.getenv("EK_CLIENT_TIMEOUT", "6"))
    except ValueError as error:
        raise ValueError("EK_CLIENT_TIMEOUT must be numeric") from error
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("EK_CLIENT_TIMEOUT must be finite and positive")
    return EkClientConfig(endpoint, instance_id, timeout_seconds)
