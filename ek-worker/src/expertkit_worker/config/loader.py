"""Load the Worker YAML file before any runtime component is initialized."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from expertkit_worker.config.models import WorkerConfig


class ConfigFileError(ValueError):
    """Indicate that a Worker configuration file cannot be read as a YAML mapping."""


def load_config(path: str | Path) -> WorkerConfig:
    """Load and validate one Worker configuration file.

    Args:
        path: UTF-8 YAML file selected by the Worker ``--config`` argument.

    Returns:
        An immutable configuration model with all cross-section defaults resolved.

    Raises:
        ConfigFileError: The file cannot be read, parsed, or does not contain a mapping.
        pydantic.ValidationError: The parsed mapping violates the Worker schema.
    """

    config_path = Path(path)
    try:
        document: Any = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigFileError(f"cannot read Worker configuration {config_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigFileError(f"invalid YAML in Worker configuration {config_path}: {exc}") from exc

    if not isinstance(document, Mapping):
        raise ConfigFileError("Worker configuration must contain one YAML mapping")

    return WorkerConfig.model_validate(document)
