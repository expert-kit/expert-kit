"""Create the selected Compute backend and its weight adapter."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from expertkit_worker.backends.base import ComputeBackend
from expertkit_worker.config import ActivationDType, BackendName, WorkerConfig
from expertkit_worker.weights.adapter import WeightAdapter

_TORCH_DTYPES = {
    ActivationDType.FP16: torch.float16,
    ActivationDType.BF16: torch.bfloat16,
    ActivationDType.FP32: torch.float32,
}


def torch_dtype(value: ActivationDType) -> torch.dtype:
    """Return the Torch dtype selected by one validated Worker configuration value."""

    return _TORCH_DTYPES[value]


def create_weight_adapter(
    config: WorkerConfig,
    *,
    source_dtype: torch.dtype,
    compute_dtype: torch.dtype,
    device: torch.device,
) -> WeightAdapter[Any, Any]:
    """Create the weight conversion and device-placement implementation for the Backend."""

    if config.worker.backend is BackendName.TORCH:
        from expertkit_worker.backends.torch import TorchWeightAdapter

        return TorchWeightAdapter(
            hidden_dim=config.model.hidden_dim,
            intermediate_dim=config.model.expert_intermediate_dim,
            source_dtype=source_dtype,
            compute_dtype=compute_dtype,
            device=device,
        )
    if config.worker.backend is BackendName.GGML:
        try:
            from expertkit_worker.backends.ggml import GgmlWeightAdapter
        except ModuleNotFoundError as error:
            if error.name == "ggml":
                raise RuntimeError("the GGML Backend requires the locked ggml extra") from error
            raise

        return GgmlWeightAdapter(
            hidden_dim=config.model.hidden_dim,
            intermediate_dim=config.model.expert_intermediate_dim,
            source_dtype=source_dtype,
            compute_dtype=compute_dtype,
        )
    try:
        from expertkit_worker.backends.fused import FusedWeightAdapter
    except ModuleNotFoundError as error:
        if error.name == "triton":
            raise RuntimeError("the fused Backend requires the locked fused extra") from error
        raise

    return FusedWeightAdapter(
        num_layers=config.model.num_layers,
        experts_per_layer=config.model.experts_per_layer,
        hidden_dim=config.model.hidden_dim,
        intermediate_dim=config.model.expert_intermediate_dim,
        source_dtype=source_dtype,
        compute_dtype=compute_dtype,
        device=device,
    )


def create_compute_backend(
    config: WorkerConfig,
    *,
    dtype: torch.dtype,
    device: torch.device,
    acquire_many: Callable[[int, tuple[int, ...]], Any],
) -> ComputeBackend:
    """Create the computation implementation selected by the Worker configuration."""

    if config.worker.backend is BackendName.TORCH:
        from expertkit_worker.backends.torch import TorchBackend

        return TorchBackend(
            hidden_dim=config.model.hidden_dim,
            intermediate_dim=config.model.expert_intermediate_dim,
            top_k=config.model.top_k,
            dtype=dtype,
            device=device,
            acquire_many=acquire_many,
        )
    if config.worker.backend is BackendName.GGML:
        from expertkit_worker.backends.ggml import GgmlBackend

        if config.worker.ggml is None:
            raise ValueError("worker.ggml configuration is missing after validation")
        return GgmlBackend(
            hidden_dim=config.model.hidden_dim,
            intermediate_dim=config.model.expert_intermediate_dim,
            top_k=config.model.top_k,
            dtype=dtype,
            cpu_threads=config.worker.ggml.cpu_threads,
            acquire_many=acquire_many,
        )
    try:
        from expertkit_worker.backends.fused import FusedBackend
    except ModuleNotFoundError as error:
        if error.name == "triton":
            raise RuntimeError("the fused Backend requires the locked fused extra") from error
        raise

    return FusedBackend(
        num_layers=config.model.num_layers,
        experts_per_layer=config.model.experts_per_layer,
        hidden_dim=config.model.hidden_dim,
        intermediate_dim=config.model.expert_intermediate_dim,
        top_k=config.model.top_k,
        dtype=dtype,
        device=device,
        acquire_many=acquire_many,
    )
