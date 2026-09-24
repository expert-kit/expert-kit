from __future__ import annotations

from dataclasses import dataclass

import torch

from expertkit_worker.device import WorkerDeviceRuntime
from expertkit_worker.weights import WeightPlacementFatalError, WeightPlacementFatalReason
from expertkit_worker.weights.adapter import WeightAdapter
from expertkit_worker.weights.format import SafeTensorData, SafeTensorDType, SafeTensorRegion

_TORCH_DTYPES = {
    SafeTensorDType.FP16: torch.float16,
    SafeTensorDType.BF16: torch.bfloat16,
    SafeTensorDType.FP32: torch.float32,
}


@dataclass(frozen=True, slots=True)
class AscendFusedCpuWeights:
    gate_proj: torch.Tensor
    up_proj: torch.Tensor
    down_proj: torch.Tensor

    # TODO: add a validation check for __post_init__

    @torch.no_grad
    def copy(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> AscendFusedExpertWeights:
        intermediate_dim, hidden_dim = self.gate_proj.size()
        # w13 packs gate and up matrix
        w13 = torch.empty(
            (hidden_dim, 2 * intermediate_dim),
            device=device,
            dtype=dtype,
        )  # [H, 2 * I]
        w2 = torch.empty(
            (intermediate_dim, hidden_dim),
            device=device,
            dtype=dtype,
        )  # [I, H]

        # By convention, safetensors as well as gate/up/down tensors
        # are stored as (out_feature, in_feature), whereas w13 and w2
        # are stored as (in_feature, out_feature), thus a transpose is
        # needed
        w13[:, :intermediate_dim].copy_(self.gate_proj.T)
        w13[:, intermediate_dim:].copy_(self.up_proj.T)
        w2.copy_(self.down_proj.T)

        return AscendFusedExpertWeights(w13=w13, w2=w2)


@dataclass(frozen=True, slots=True)
class AscendFusedExpertWeights:
    w13: torch.Tensor
    w2: torch.Tensor


class AscendFusedWeightAdapter(WeightAdapter[AscendFusedCpuWeights, AscendFusedExpertWeights]):
    """Convert validated expert files into Ascend backend's ready objects."""

    def __init__(
        self,
        *,
        num_layers: int,
        experts_per_layer: int,
        hidden_dim: int,
        intermediate_dim: int,
        source_dtype: torch.dtype,
        compute_dtype: torch.dtype,
        runtime: WorkerDeviceRuntime,
    ) -> None:
        self._num_layers = num_layers
        self._experts_per_layer = experts_per_layer
        self._hidden_dim = hidden_dim
        self._intermediate_dim = intermediate_dim
        self._source_dtype = source_dtype
        self._compute_dtype = compute_dtype
        self._runtime = runtime

    @property
    def backend_name(self) -> str:
        """Return the configured built-in Backend name used for cache identity."""
        return "ascend_fused"

    def make_cpu_weight(self, source: SafeTensorData) -> AscendFusedCpuWeights:
        """Create the cached CPU object from validated source regions."""
        gate = source.find_unique_suffix(("gate_proj.weight", "w1.weight"))
        up = source.find_unique_suffix(("up_proj.weight", "w3.weight"))
        down = source.find_unique_suffix(("down_proj.weight", "w2.weight"))
        return AscendFusedCpuWeights(
            gate_proj=self._view(gate, (self._intermediate_dim, self._hidden_dim)),
            up_proj=self._view(up, (self._intermediate_dim, self._hidden_dim)),
            down_proj=self._view(down, (self._hidden_dim, self._intermediate_dim)),
        )

    def make_ready_weight(
        self,
        cpu_weight: AscendFusedCpuWeights,
        *,
        layer_id: int,
        expert_id: int,
    ) -> AscendFusedExpertWeights:
        """Create and synchronize the final object for one stable expert position."""
        runtime = self._runtime
        device = runtime.device
        try:
            ready = cpu_weight.copy(device=device, dtype=self._compute_dtype)
            runtime.capture_current_work().wait_host()
            return ready
        except torch.OutOfMemoryError as error:
            raise WeightPlacementFatalError(
                WeightPlacementFatalReason.DEVICE_OOM,
                str(error),
            ) from error
        except RuntimeError as error:
            if device.type != "cpu":
                raise WeightPlacementFatalError(
                    WeightPlacementFatalReason.DEVICE_FAILURE,
                    str(error),
                ) from error
            raise

    def cpu_extra_bytes(self) -> int:
        """Return bytes allocated beyond the retained SafeTensors buffer."""
        return 0

    def source_tensor_bytes(self) -> int:
        """Return the exact encoded bytes for one expert's Tensor data."""
        items = 3 * self._hidden_dim * self._intermediate_dim
        return items * self._source_dtype.itemsize

    def ready_weight_bytes(self) -> int:
        """Return final device bytes for one equal-shaped expert."""
        items = 3 * self._hidden_dim * self._intermediate_dim
        return items * self._compute_dtype.itemsize

    def conversion_temporary_bytes(self) -> int:
        """Return conservative temporary device bytes used during conversion."""
        return 0

    def _view(self, region: SafeTensorRegion, shape: tuple[int, int]) -> torch.Tensor:
        if _TORCH_DTYPES[region.dtype] != self._source_dtype:
            raise ValueError(f"weight Tensor {region.name!r} has an unexpected dtype")
        if region.shape != shape:
            raise ValueError(f"weight Tensor {region.name!r} has an unexpected shape")
        return torch.frombuffer(region.data, dtype=self._source_dtype).reshape(shape)
