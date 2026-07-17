"""GGML conversion from SafeTensors regions to retained CPU views."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from expertkit_transport.contracts import ACTIVATION_DTYPES

from expertkit_worker.backends.ggml.weights import _WEIGHT_CONTEXT_BYTES, GgmlExpertWeights
from expertkit_worker.weights.adapter import WeightAdapter
from expertkit_worker.weights.format import (
    SafeTensorData,
    SafeTensorDType,
    SafeTensorRegion,
)

_TORCH_DTYPES = {
    SafeTensorDType.FP16: torch.float16,
    SafeTensorDType.BF16: torch.bfloat16,
    SafeTensorDType.FP32: torch.float32,
}


@dataclass(frozen=True, slots=True)
class GgmlCpuWeights:
    """Hold zero-copy Torch views over one parsed CPU expert file."""

    gate_proj: torch.Tensor
    up_proj: torch.Tensor
    down_proj: torch.Tensor

    @property
    def dtype(self) -> torch.dtype:
        """Return the parsed source dtype."""

        return self.gate_proj.dtype

    @property
    def tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return source views in gate, up, and down order."""

        return self.gate_proj, self.up_proj, self.down_proj


class GgmlWeightAdapter(WeightAdapter[GgmlCpuWeights, GgmlExpertWeights]):
    """Prepare CPU-only ggml weight views once during expert loading."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        intermediate_dim: int,
        source_dtype: torch.dtype,
        compute_dtype: torch.dtype,
    ) -> None:
        for name, value in (
            ("hidden_dim", hidden_dim),
            ("intermediate_dim", intermediate_dim),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if source_dtype not in ACTIVATION_DTYPES:
            raise ValueError("GGML source weight dtype must be FP16, BF16, or FP32")
        if compute_dtype not in ACTIVATION_DTYPES:
            raise ValueError("GGML compute weight dtype must be FP16, BF16, or FP32")

        self._hidden_dim = hidden_dim
        self._intermediate_dim = intermediate_dim
        self._source_dtype = source_dtype
        self._compute_dtype = compute_dtype

    @property
    def backend_name(self) -> str:
        """Return the built-in Backend name."""

        return "ggml"

    def make_cpu_weight(self, source: SafeTensorData) -> GgmlCpuWeights:
        """Create zero-copy CPU Tensor views over the parsed file."""

        gate = source.find_unique_suffix(("gate_proj.weight", "w1.weight"))
        up = source.find_unique_suffix(("up_proj.weight", "w3.weight"))
        down = source.find_unique_suffix(("down_proj.weight", "w2.weight"))
        return GgmlCpuWeights(
            gate_proj=self._view(gate, (self._intermediate_dim, self._hidden_dim)),
            up_proj=self._view(up, (self._intermediate_dim, self._hidden_dim)),
            down_proj=self._view(down, (self._hidden_dim, self._intermediate_dim)),
        )

    def make_ready_weight(
        self,
        cpu_weight: GgmlCpuWeights,
        *,
        layer_id: int,
        expert_id: int,
    ) -> GgmlExpertWeights:
        """Convert once to the configured dtype and create direct ggml views."""

        del layer_id, expert_id
        if not isinstance(cpu_weight, GgmlCpuWeights):
            raise TypeError("GGML cached weight has the wrong object type")
        if cpu_weight.dtype != self._source_dtype:
            raise ValueError("GGML cached weight dtype does not match the configured source")
        converted = tuple(
            tensor.to(
                dtype=self._compute_dtype,
                copy=self._compute_dtype != self._source_dtype,
            ).contiguous()
            for tensor in cpu_weight.tensors
        )
        return GgmlExpertWeights(
            gate_proj=converted[0],
            up_proj=converted[1],
            down_proj=converted[2],
        )

    def cpu_extra_bytes(self) -> int:
        """Return zero because parsed CPU Tensors view the retained source buffer."""

        return 0

    def source_tensor_bytes(self) -> int:
        """Return exact encoded projection bytes for one source expert."""

        elements = 3 * self._hidden_dim * self._intermediate_dim
        return elements * torch.empty((), dtype=self._source_dtype).element_size()

    def ready_weight_bytes(self) -> int:
        """Return ready weight storage plus the fixed ggml metadata context."""

        elements = 3 * self._hidden_dim * self._intermediate_dim
        tensor_bytes = elements * torch.empty((), dtype=self._compute_dtype).element_size()
        return tensor_bytes + _WEIGHT_CONTEXT_BYTES

    def conversion_temporary_bytes(self) -> int:
        """Return zero because conversion writes directly into final CPU Tensors."""

        return 0

    def _view(self, region: SafeTensorRegion, shape: tuple[int, int]) -> torch.Tensor:
        if _TORCH_DTYPES[region.dtype] != self._source_dtype:
            raise ValueError(f"weight Tensor {region.name!r} has an unexpected dtype")
        if region.shape != shape:
            raise ValueError(f"weight Tensor {region.name!r} has an unexpected shape")
        return torch.frombuffer(region.data, dtype=self._source_dtype).reshape(shape)
