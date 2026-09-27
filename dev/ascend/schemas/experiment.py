"""Experiment, model, dataset, and serving configuration models."""

from __future__ import annotations
import yaml
from typing import Self
from pathlib import Path
from pydantic import AnyHttpUrl, Field, model_validator

from enum import StrEnum
from .config import ConfigModel


class DatasetType(StrEnum):
    """Dataset source supported by the benchmark generator."""

    RANDOM = "random"
    SHAREGPT = "sharegpt"
    CUSTOM = "custom"


class Dtype(StrEnum):
    """Activation or weight dtype label emitted in generated configs."""

    FP16 = "fp16"
    BF16 = "bf16"
    # FP4 and FP8 are accepted as configuration labels for generated benchmarks.
    FP8 = "fp8"
    FP4 = "fp4"


class TracingConfig(ConfigModel):
    """Optional tracing settings shared by the generated Frontend and Workers."""

    enabled: bool = False
    endpoint: AnyHttpUrl | None = None
    sample_ratio: float = Field(default=0.01, gt=0, le=1)

    @model_validator(mode="after")
    def validate_endpoint(self) -> TracingConfig:
        """Require a plaintext collector endpoint when tracing is enabled."""
        if self.enabled and self.endpoint is None:
            raise ValueError("tracing.endpoint is required when tracing is enabled")
        if (
            self.enabled
            and self.endpoint is not None
            and self.endpoint.scheme != "http"
        ):
            raise ValueError("tracing requires a plaintext HTTP endpoint")
        return self


class ExperimentConfig(ConfigModel):
    """Complete model, dataset, serving, and run configuration."""

    model: ModelConfig
    dataset: DatasetConfig
    serve: ServeConfig
    run: RunConfig
    tracing: TracingConfig = Field(default_factory=TracingConfig)

    @classmethod
    def from_yaml(cls, file: Path) -> Self:
        data = yaml.safe_load(file.read_text())
        return cls.model_validate(data)

    @model_validator(mode="after")
    def validate_dataset_run_options(self) -> Self:
        if self.dataset.type is DatasetType.RANDOM:
            if self.run.input_len is None:
                raise ValueError("random dataset requires run.input_len")
        elif self.run.input_len is not None:
            raise ValueError(
                f"{self.dataset.type} dataset must not define run.input_len"
            )

        return self


class ModelConfig(ConfigModel):
    """Model artifact and frontend execution settings."""

    name: str
    path_ref: str
    weight_version: str
    num_layers: int
    experts_per_layer: int
    hidden_dim: int
    intermediate_dim: int
    topk: int
    weight_dtype: Dtype
    activation_dtype: Dtype


class DatasetConfig(ConfigModel):
    """Dataset artifact and sampling settings."""

    type: DatasetType
    name: str
    path_ref: str | None = None
    mounted_path: Path | None = None
    file: str | None = None

    @model_validator(mode="after")
    def validate_path_fields(self) -> Self:
        path_fields = (
            self.path_ref,
            self.mounted_path,
            self.file,
        )

        if self.type is DatasetType.RANDOM:
            if any(value is not None for value in path_fields):
                raise ValueError(
                    "random dataset must not define path_ref, mounted_path, or file"
                )
        elif any(value is None for value in path_fields):
            raise ValueError(
                f"{self.type} dataset requires path_ref, mounted_path, and file"
            )

        return self


class ServeConfig(ConfigModel):
    """Serving frontend options for generated deployment files."""

    gpu_memory_utilization: float
    max_model_len: int
    enforce_eager: bool = False


class RunConfig(ConfigModel):
    """Benchmark repetition, concurrency, and output settings."""

    num_prompts: int
    max_concurrency: int
    input_len: int | None = None
    output_len: int
    num_warmups: int
    ignore_eos: bool
    temperature: float
    save_result: bool
