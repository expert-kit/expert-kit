"""Validated runtime configuration for Torch frontend benchmarks."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _kebab_case(name: str) -> str:
    return name.replace("_", "-")


class DevicePlatform(StrEnum):
    """Torch device prefix used to construct per-rank devices."""

    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    NPU = "npu"


class LauncherKind(StrEnum):
    """Process-launch strategy."""

    AUTO = "auto"
    INLINE = "inline"
    SPAWN = "spawn"


class BenchmarkMode(StrEnum):
    """Expert execution mode."""

    EXPERTKIT = "expertkit"
    LOCAL = "local"


class DatasetName(StrEnum):
    """Dataset adapters supported by the Torch ablation."""

    SHAREGPT = "sharegpt"


class BenchmarkDtype(StrEnum):
    """User-facing model dtype names."""

    AUTO = "auto"
    FP16 = "fp16"
    BF16 = "bf16"
    FP32 = "fp32"

    @property
    def torch_name(self) -> str:
        return {
            BenchmarkDtype.AUTO: "auto",
            BenchmarkDtype.FP16: "float16",
            BenchmarkDtype.BF16: "bfloat16",
            BenchmarkDtype.FP32: "float32",
        }[self]


class BenchmarkConfig(BaseModel):
    """Complete serde and launcher boundary for one benchmark invocation."""

    model_config = ConfigDict(
        alias_generator=_kebab_case,
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        str_strip_whitespace=True,
    )

    mode: BenchmarkMode = BenchmarkMode.EXPERTKIT
    controller_endpoint: str | None = None
    instance_id: int | None = Field(default=None, gt=0)
    launcher: LauncherKind = LauncherKind.AUTO
    device_platform: DevicePlatform
    device_ids: tuple[int, ...]
    model_path: Path
    dtype: BenchmarkDtype = BenchmarkDtype.AUTO
    dataset_name: DatasetName = DatasetName.SHAREGPT
    dataset_path: Path
    seed: int = 0
    num_prompts: int = Field(gt=0)
    max_concurrency: int = Field(gt=0)
    output_length: int = Field(gt=0)
    warmup_runs: int = Field(default=1, ge=0)
    json_output: Path | None = None

    @field_validator("device_ids")
    @classmethod
    def validate_device_ids(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value:
            raise ValueError("device_ids must not be empty")
        if any(isinstance(device_id, bool) or device_id < 0 for device_id in value):
            raise ValueError("device_ids must contain non-negative integers")
        if len(set(value)) != len(value):
            raise ValueError("device_ids must be unique")
        return value

    @model_validator(mode="after")
    def validate_execution_shape(self) -> Self:
        rank_count = len(self.device_ids)
        if self.num_prompts < rank_count:
            raise ValueError("num_prompts must be at least the number of ranks")
        if self.max_concurrency % rank_count != 0:
            raise ValueError("max_concurrency must be divisible by the number of ranks")
        if self.launcher is LauncherKind.INLINE and rank_count != 1:
            raise ValueError("inline launcher requires exactly one device")
        if self.launcher is LauncherKind.SPAWN and rank_count < 2:
            raise ValueError("spawn launcher requires at least two devices")
        if self.mode is BenchmarkMode.EXPERTKIT and not self.controller_endpoint:
            raise ValueError("expertkit mode requires controller_endpoint")
        return self

    @property
    def rank_count(self) -> int:
        return len(self.device_ids)

    @property
    def batch_size_per_rank(self) -> int:
        return self.max_concurrency // self.rank_count

    def device_name(self, device_id: int) -> str:
        return f"{self.device_platform.value}:{device_id}"

    @classmethod
    def from_yaml(cls, path: Path) -> Self:
        """Load a complete config and resolve its Host paths."""

        source = path.expanduser().resolve()
        data = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("benchmark config must contain a YAML mapping")
        config = cls.model_validate(data)
        return config.resolve_paths(source.parent)

    def resolve_paths(self, base: Path) -> Self:
        """Resolve Host paths against a config or process directory."""

        return self.model_copy(
            update={
                "model_path": _resolve_host_path(self.model_path, base),
                "dataset_path": _resolve_host_path(self.dataset_path, base),
                "json_output": (
                    _resolve_host_path(self.json_output, base)
                    if self.json_output is not None
                    else None
                ),
            }
        )


def _resolve_host_path(path: Path, base: Path) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = base / expanded
    return expanded.resolve()
