"""Torch conversion from validated SafeTensors regions to ready expert weights."""

from __future__ import annotations

import torch
from expertkit_transport.contracts import ACTIVATION_DTYPES

from expertkit_worker.backends.torch.weights import TorchExpertWeights
from expertkit_worker.weights import (
    SafeTensorData,
    SafeTensorDType,
    SafeTensorRegion,
    WeightAdapter,
)

_TORCH_DTYPES = {
    SafeTensorDType.FP16: torch.float16,
    SafeTensorDType.BF16: torch.bfloat16,
    SafeTensorDType.FP32: torch.float32,
}


class TorchWeightAdapter(WeightAdapter[TorchExpertWeights, TorchExpertWeights]):
    """Build zero-copy CPU views and final-device Torch expert objects."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        intermediate_dim: int,
        source_dtype: torch.dtype,
        compute_dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        for name, value in (
            ("hidden_dim", hidden_dim),
            ("intermediate_dim", intermediate_dim),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if source_dtype not in ACTIVATION_DTYPES:
            raise ValueError("Torch source weight dtype must be FP16, BF16, or FP32")
        if compute_dtype not in ACTIVATION_DTYPES:
            raise ValueError("Torch compute weight dtype must be FP16, BF16, or FP32")
        resolved_device = torch.device(device)
        if resolved_device.type not in {"cpu", "cuda"}:
            raise ValueError("Torch weight device must be CPU or CUDA")
        if resolved_device.type == "cuda" and resolved_device.index is None:
            raise ValueError("Torch weight CUDA device must include an index")

        self._hidden_dim = hidden_dim
        self._intermediate_dim = intermediate_dim
        self._source_dtype = source_dtype
        self._compute_dtype = compute_dtype
        self._device = resolved_device

    @property
    def backend_name(self) -> str:
        """Return the built-in Backend name."""

        return "torch"

    def make_cpu_weight(self, source: SafeTensorData) -> TorchExpertWeights:
        """Create Torch CPU Tensor views without copying source weight bytes."""

        gate = source.find_unique_suffix(("gate_proj.weight", "w1.weight"))
        up = source.find_unique_suffix(("up_proj.weight", "w3.weight"))
        down = source.find_unique_suffix(("down_proj.weight", "w2.weight"))
        return TorchExpertWeights(
            gate_proj=self._view(gate, (self._intermediate_dim, self._hidden_dim)),
            up_proj=self._view(up, (self._intermediate_dim, self._hidden_dim)),
            down_proj=self._view(down, (self._hidden_dim, self._intermediate_dim)),
        )

    def make_ready_weight(self, cpu_weight: TorchExpertWeights) -> TorchExpertWeights:
        """Copy or convert CPU views directly into final Torch tensors."""

        if cpu_weight.device.type != "cpu":
            raise ValueError("Torch cached weight must be on CPU")
        if cpu_weight.dtype != self._source_dtype:
            raise ValueError("Torch cached weight dtype does not match the configured source")
        ready = TorchExpertWeights(
            gate_proj=cpu_weight.gate_proj.to(
                device=self._device,
                dtype=self._compute_dtype,
                copy=self._device.type != "cpu" or self._compute_dtype != self._source_dtype,
            ),
            up_proj=cpu_weight.up_proj.to(
                device=self._device,
                dtype=self._compute_dtype,
                copy=self._device.type != "cpu" or self._compute_dtype != self._source_dtype,
            ),
            down_proj=cpu_weight.down_proj.to(
                device=self._device,
                dtype=self._compute_dtype,
                copy=self._device.type != "cpu" or self._compute_dtype != self._source_dtype,
            ),
        )
        if self._device.type == "cuda":
            torch.cuda.current_stream(self._device).synchronize()
        return ready

    def cpu_extra_bytes(self) -> int:
        """Return zero because CPU Tensors view the retained source buffer."""

        return 0

    def ready_weight_bytes(self) -> int:
        """Return the exact logical size of the three final projection Tensors."""

        elements = 3 * self._hidden_dim * self._intermediate_dim
        return elements * torch.empty((), dtype=self._compute_dtype).element_size()

    def conversion_temporary_bytes(self) -> int:
        """Return zero because the MVP allocates no project-managed device staging."""

        return 0

    def _view(self, region: SafeTensorRegion, shape: tuple[int, int]) -> torch.Tensor:
        if _TORCH_DTYPES[region.dtype] != self._source_dtype:
            raise ValueError(f"weight Tensor {region.name!r} has an unexpected dtype")
        if region.shape != shape:
            raise ValueError(f"weight Tensor {region.name!r} has an unexpected shape")
        return torch.frombuffer(region.data, dtype=self._source_dtype).reshape(shape)
