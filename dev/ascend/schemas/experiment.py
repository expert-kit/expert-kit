from __future__ import annotations
import yaml
from typing import Self
from pathlib import Path
from pydantic import model_validator

from enum import StrEnum
from .config import ConfigModel


class DatasetType(StrEnum):
    RANDOM = "random"
    SHAREGPT = "sharegpt"
    CUSTOM = "custom"


class Dtype(StrEnum):
    FP16 = "fp16"
    BF16 = "bf16"


class ExperimentConfig(ConfigModel):
    model: ModelConfig
    dataset: DatasetConfig
    serve: ServeConfig
    run: RunConfig

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
    name: str
    path_ref: str
    weight_version: str
    num_layers: int
    experts_per_layer: int
    hidden_dim: int
    intermediate_dim: int
    topk: int
    dtype: Dtype


class DatasetConfig(ConfigModel):
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
    gpu_memory_utilization: float
    max_model_len: int


class RunConfig(ConfigModel):
    num_prompts: int
    max_concurrency: int
    input_len: int | None = None
    output_len: int
    num_warmups: int
    ignore_eos: bool
    temperature: float
    save_result: bool
