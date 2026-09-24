"""Create the selected Compute backend and its weight adapter."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from expertkit_worker.backends.base import ComputeBackend
from expertkit_worker.config import ActivationDType, BackendName, QuantizationType, WorkerConfig
from expertkit_worker.device import WorkerDeviceRuntime
from expertkit_worker.weights.adapter import WeightAdapter

_TORCH_DTYPES = {
    ActivationDType.FP16: torch.float16,
    ActivationDType.BF16: torch.bfloat16,
    ActivationDType.FP32: torch.float32,
}

_TORCH_LINEAR_COMPUTE = {
    QuantizationType.W8A8: "w8a8",
    QuantizationType.MODELSLIM_W8A8_DYNAMIC: "modelslim_w8a8_dynamic",
    QuantizationType.FP4: "fp8_reference",
    QuantizationType.GPTQ: "float",
}


def torch_dtype(value: ActivationDType) -> torch.dtype:
    """Return the Torch dtype selected by one validated Worker configuration value."""

    return _TORCH_DTYPES[value]


def _torch_linear_compute(quantization: QuantizationType | None) -> str:
    """Resolve the Torch compute recipe once for the selected quantization."""

    return "float" if quantization is None else _TORCH_LINEAR_COMPUTE[quantization]


def create_weight_adapter(
    config: WorkerConfig,
    *,
    source_dtype: torch.dtype,
    compute_dtype: torch.dtype,
    runtime: WorkerDeviceRuntime,
) -> WeightAdapter[Any, Any]:
    """Create the weight conversion and device-placement implementation for the Backend."""

    if config.worker.backend is BackendName.TORCH:
        from expertkit_worker.backends.torch import (
            TorchFP4WeightAdapter,
            TorchGPTQWeightAdapter,
            TorchW8A8WeightAdapter,
            TorchWeightAdapter,
        )

        if config.model.quantization is not None:
            if config.model.quantization.type is QuantizationType.W8A8:
                return TorchW8A8WeightAdapter(
                    hidden_dim=config.model.hidden_dim,
                    intermediate_dim=config.model.expert_intermediate_dim,
                    device=runtime.device,
                    compute_dtype=compute_dtype,
                )
            if config.model.quantization.type is QuantizationType.MODELSLIM_W8A8_DYNAMIC:
                if runtime.device.type != "npu":
                    raise ValueError("ModelSlim W8A8_DYNAMIC requires an indexed NPU device")
                from expertkit_worker.backends.torch import TorchModelSlimW8A8WeightAdapter

                return TorchModelSlimW8A8WeightAdapter(
                    hidden_dim=config.model.hidden_dim,
                    intermediate_dim=config.model.expert_intermediate_dim,
                    runtime=runtime,
                    compute_dtype=compute_dtype,
                )
            if config.model.quantization.type is QuantizationType.FP4:
                return TorchFP4WeightAdapter(
                    hidden_dim=config.model.hidden_dim,
                    intermediate_dim=config.model.expert_intermediate_dim,
                    device=runtime.device,
                    compute_dtype=compute_dtype,
                )
            if config.model.quantization.type is not QuantizationType.GPTQ:
                raise ValueError("unsupported Torch quantization type")
            return TorchGPTQWeightAdapter(
                hidden_dim=config.model.hidden_dim,
                intermediate_dim=config.model.expert_intermediate_dim,
                group_size=config.model.quantization.group_size,
                device=runtime.device,
                compute_dtype=compute_dtype,
            )

        return TorchWeightAdapter(
            hidden_dim=config.model.hidden_dim,
            intermediate_dim=config.model.expert_intermediate_dim,
            source_dtype=source_dtype,
            compute_dtype=compute_dtype,
            runtime=runtime,
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
    if config.worker.backend is BackendName.FUSED:
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
            device=runtime.device,
        )

    if config.worker.backend is BackendName.ASCEND_FUSED:
        from expertkit_worker.backends.ascend_fused import (
            AscendFusedModelSlimW8A8WeightAdapter,
            AscendFusedWeightAdapter,
        )

        if config.model.quantization is not None:
            if config.model.quantization.type is not QuantizationType.MODELSLIM_W8A8_DYNAMIC:
                raise ValueError("unsupported Ascend fused quantization type")
            return AscendFusedModelSlimW8A8WeightAdapter(
                hidden_dim=config.model.hidden_dim,
                intermediate_dim=config.model.expert_intermediate_dim,
                runtime=runtime,
                compute_dtype=compute_dtype,
            )

        return AscendFusedWeightAdapter(
            num_layers=config.model.num_layers,
            experts_per_layer=config.model.experts_per_layer,
            hidden_dim=config.model.hidden_dim,
            intermediate_dim=config.model.expert_intermediate_dim,
            source_dtype=source_dtype,
            compute_dtype=compute_dtype,
            runtime=runtime,
        )


def create_compute_backend(
    config: WorkerConfig,
    *,
    dtype: torch.dtype,
    runtime: WorkerDeviceRuntime,
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
            runtime=runtime,
            acquire_many=acquire_many,
            expert_compute=config.model.expert_compute,
            swiglu_limit=config.model.swiglu_limit,
            linear_compute=_torch_linear_compute(
                config.model.quantization.type if config.model.quantization is not None else None
            ),
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

    if config.worker.backend is BackendName.FUSED:
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
            device=runtime.device,
            acquire_many=acquire_many,
        )

    if config.worker.backend is BackendName.ASCEND_FUSED:
        from expertkit_worker.backends.ascend_fused import (
            AscendFusedBackend,
            AscendFusedComputeType,
        )

        compute_type = (
            AscendFusedComputeType.DYNAMIC_W8A8
            if config.model.quantization is not None
            else AscendFusedComputeType.UNQUANTIZED
        )

        return AscendFusedBackend(
            hidden_dim=config.model.hidden_dim,
            intermediate_dim=config.model.expert_intermediate_dim,
            top_k=config.model.top_k,
            dtype=dtype,
            runtime=runtime,
            acquire_many=acquire_many,
            compute_type=compute_type,
            swiglu_limit=config.model.swiglu_limit,
        )
