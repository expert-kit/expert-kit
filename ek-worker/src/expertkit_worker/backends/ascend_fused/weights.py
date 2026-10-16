from __future__ import annotations

from dataclasses import dataclass

import torch
from expertkit_transport.batches import ACTIVATION_DTYPES

from expertkit_worker.device import WorkerDeviceRuntime
from expertkit_worker.weights import WeightPlacementFatalError, WeightPlacementFatalReason
from expertkit_worker.weights.adapter import WeightAdapter
from expertkit_worker.weights.format import SafeTensorData, SafeTensorDType, SafeTensorRegion

_TORCH_DTYPES = {
    SafeTensorDType.FP16: torch.float16,
    SafeTensorDType.BF16: torch.bfloat16,
    SafeTensorDType.FP32: torch.float32,
}
_SCALE_DTYPES = {
    SafeTensorDType.BF16: torch.bfloat16,
    SafeTensorDType.FP32: torch.float32,
}
_ROLES = (("gate_proj", "w1"), ("up_proj", "w3"), ("down_proj", "w2"))


def _dynamic_gmm_scale_dtype(output_dtype: torch.dtype) -> torch.dtype:
    """Return the weight-scale dtype required by per-token quantized GMM."""

    if output_dtype is torch.bfloat16:
        return torch.bfloat16
    if output_dtype is torch.float16:
        return torch.float32
    raise ValueError("dynamic quantized GMM output must be FP16 or BF16")


@dataclass(frozen=True, slots=True)
class AscendFusedCpuWeights:
    gate_proj: torch.Tensor
    up_proj: torch.Tensor
    down_proj: torch.Tensor

    def __post_init__(self) -> None:
        if self.gate_proj.ndim != 2 or self.up_proj.shape != self.gate_proj.shape:
            raise ValueError("Ascend fused gate and up projection shapes are inconsistent")
        if self.down_proj.shape != self.gate_proj.T.shape:
            raise ValueError("Ascend fused down projection shape is inconsistent")
        tensors = (self.gate_proj, self.up_proj, self.down_proj)
        if any(tensor.device.type != "cpu" for tensor in tensors):
            raise ValueError("Ascend fused cached weights must be on CPU")
        if any(tensor.dtype != self.gate_proj.dtype for tensor in tensors):
            raise ValueError("Ascend fused cached weights must share a dtype")
        if any(tensor.requires_grad for tensor in tensors):
            raise ValueError("Ascend fused cached weights must not require gradients")

    @torch.no_grad()
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

    def __post_init__(self) -> None:
        if self.w13.ndim != 2 or self.w2.ndim != 2:
            raise ValueError("Ascend fused weights must be matrices")
        hidden_dim, packed_intermediate_dim = self.w13.shape
        if packed_intermediate_dim % 2 or self.w2.shape != (
            packed_intermediate_dim // 2,
            hidden_dim,
        ):
            raise ValueError("Ascend fused weight shapes are inconsistent")
        if self.w13.dtype not in ACTIVATION_DTYPES or self.w2.dtype != self.w13.dtype:
            raise ValueError("Ascend fused weights must share a floating activation dtype")
        if self.w13.device != self.w2.device:
            raise ValueError("Ascend fused weights must share a device")
        if not self.w13.is_contiguous() or not self.w2.is_contiguous():
            raise ValueError("Ascend fused weights must be contiguous")
        if self.w13.requires_grad or self.w2.requires_grad:
            raise ValueError("Ascend fused weights must not require gradients")

    @property
    def hidden_dim(self) -> int:
        return self.w13.shape[0]

    @property
    def intermediate_dim(self) -> int:
        return self.w13.shape[1] // 2

    @property
    def device(self) -> torch.device:
        return self.w13.device

    @property
    def storage_bytes(self) -> int:
        return sum(tensor.numel() * tensor.element_size() for tensor in (self.w13, self.w2))


@dataclass(frozen=True, slots=True)
class AscendFusedModelSlimW8A8CpuWeights:
    """CPU views of ModelSlim I8 matrices and per-output-channel scales."""

    matrices: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    scales: tuple[torch.Tensor, torch.Tensor, torch.Tensor]

    def __post_init__(self) -> None:
        if len(self.matrices) != 3 or len(self.scales) != 3:
            raise ValueError("ModelSlim W8A8 requires three projections and scales")
        gate, up, down = self.matrices
        if gate.ndim != 2 or up.shape != gate.shape or down.shape != gate.T.shape:
            raise ValueError("ModelSlim projection shapes are inconsistent")
        for matrix, scale in zip(self.matrices, self.scales, strict=True):
            if matrix.device.type != "cpu" or scale.device.type != "cpu":
                raise ValueError("ModelSlim cached tensors must be on CPU")
            if matrix.dtype != torch.int8:
                raise ValueError("ModelSlim matrices must use I8")
            if scale.shape != (matrix.shape[0], 1) or scale.dtype not in {
                torch.bfloat16,
                torch.float32,
            }:
                raise ValueError("ModelSlim scales must have shape [output, 1]")
            if matrix.requires_grad or scale.requires_grad:
                raise ValueError("ModelSlim cached tensors must not require gradients")

    @torch.no_grad()
    def copy(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> AscendFusedModelSlimW8A8ExpertWeights:
        intermediate_dim, hidden_dim = self.matrices[0].shape
        gate, up, down = self.matrices
        gate_scale, up_scale, down_scale = self.scales
        scale_dtype = _dynamic_gmm_scale_dtype(dtype)

        w13 = torch.empty((hidden_dim, 2 * intermediate_dim), device=device, dtype=torch.int8)
        w2 = torch.empty((intermediate_dim, hidden_dim), device=device, dtype=torch.int8)
        w13_scale = torch.empty((2 * intermediate_dim,), device=device, dtype=scale_dtype)
        w2_scale = torch.empty((hidden_dim,), device=device, dtype=scale_dtype)

        w13[:, :intermediate_dim].copy_(gate.T)
        w13[:, intermediate_dim:].copy_(up.T)
        w2.copy_(down.T)
        w13_scale[:intermediate_dim].copy_(gate_scale.reshape(-1))
        w13_scale[intermediate_dim:].copy_(up_scale.reshape(-1))
        w2_scale.copy_(down_scale.reshape(-1))

        return AscendFusedModelSlimW8A8ExpertWeights(
            w13=w13,
            w2=w2,
            w13_scale=w13_scale,
            w2_scale=w2_scale,
            dtype=dtype,
        )


@dataclass(frozen=True, slots=True)
class AscendFusedModelSlimW8A8ExpertWeights:
    """Ascend GMM-ready ModelSlim weights and per-output-channel scales."""

    w13: torch.Tensor
    w2: torch.Tensor
    w13_scale: torch.Tensor
    w2_scale: torch.Tensor
    dtype: torch.dtype

    def __post_init__(self) -> None:
        if self.dtype not in ACTIVATION_DTYPES:
            raise ValueError("ModelSlim output dtype must be a floating activation dtype")
        if self.w13.ndim != 2 or self.w2.ndim != 2:
            raise ValueError("ModelSlim Ascend fused weights must be matrices")
        hidden_dim, packed_intermediate_dim = self.w13.shape
        if packed_intermediate_dim % 2 or self.w2.shape != (
            packed_intermediate_dim // 2,
            hidden_dim,
        ):
            raise ValueError("ModelSlim Ascend fused weight shapes are inconsistent")
        if self.w13_scale.shape != (packed_intermediate_dim,) or self.w2_scale.shape != (
            hidden_dim,
        ):
            raise ValueError("ModelSlim Ascend fused scale shapes are inconsistent")
        tensors = (self.w13, self.w2, self.w13_scale, self.w2_scale)
        if self.w13.dtype != torch.int8 or self.w2.dtype != torch.int8:
            raise ValueError("ModelSlim Ascend fused matrices must use I8")
        scale_dtype = _dynamic_gmm_scale_dtype(self.dtype)
        if self.w13_scale.dtype != scale_dtype or self.w2_scale.dtype != scale_dtype:
            raise ValueError("ModelSlim Ascend fused scale dtype must match GMM output policy")
        if any(tensor.device != self.w13.device for tensor in tensors):
            raise ValueError("ModelSlim Ascend fused tensors must share a device")
        if any(not tensor.is_contiguous() for tensor in tensors):
            raise ValueError("ModelSlim Ascend fused tensors must be contiguous")
        if any(tensor.requires_grad for tensor in tensors):
            raise ValueError("ModelSlim Ascend fused tensors must not require gradients")

    @property
    def hidden_dim(self) -> int:
        return self.w13.shape[0]

    @property
    def intermediate_dim(self) -> int:
        return self.w13.shape[1] // 2

    @property
    def device(self) -> torch.device:
        return self.w13.device

    @property
    def storage_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (self.w13, self.w2, self.w13_scale, self.w2_scale)
        )


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
        if not isinstance(cpu_weight, AscendFusedCpuWeights):
            raise TypeError("Ascend fused cached weight has the wrong object type")
        if cpu_weight.gate_proj.shape != (
            self._intermediate_dim,
            self._hidden_dim,
        ):
            raise ValueError("Ascend fused cached weight shape does not match the model")
        if cpu_weight.gate_proj.dtype != self._source_dtype:
            raise ValueError("Ascend fused cached weight dtype does not match the source")

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


class AscendFusedModelSlimW8A8WeightAdapter(
    WeightAdapter[
        AscendFusedModelSlimW8A8CpuWeights,
        AscendFusedModelSlimW8A8ExpertWeights,
    ]
):
    """Validate and pack ModelSlim dynamic-W8A8 weights for Ascend GMM."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        intermediate_dim: int,
        runtime: WorkerDeviceRuntime,
        compute_dtype: torch.dtype,
    ) -> None:
        if min(hidden_dim, intermediate_dim) <= 0:
            raise ValueError("ModelSlim dimensions must be positive")
        if compute_dtype not in {torch.float16, torch.bfloat16}:
            raise ValueError("Ascend fused ModelSlim supports only FP16 or BF16 computation")
        if runtime.device.type != "npu" or runtime.device.index is None:
            raise ValueError("Ascend fused ModelSlim requires an indexed NPU runtime")
        self._hidden_dim = hidden_dim
        self._intermediate_dim = intermediate_dim
        self._runtime = runtime
        self._compute_dtype = compute_dtype

    @property
    def backend_name(self) -> str:
        return "ascend-fused-modelslim-w8a8-dynamic"

    def make_cpu_weight(self, source: SafeTensorData) -> AscendFusedModelSlimW8A8CpuWeights:
        matrices: list[torch.Tensor] = []
        scales: list[torch.Tensor] = []
        consumed: set[str] = set()
        for roles, inputs, outputs in (
            (_ROLES[0], self._hidden_dim, self._intermediate_dim),
            (_ROLES[1], self._hidden_dim, self._intermediate_dim),
            (_ROLES[2], self._intermediate_dim, self._hidden_dim),
        ):
            weight = source.find_unique_suffix(tuple(f"{role}.weight" for role in roles))
            scale = source.tensors.get(weight.name.removesuffix("weight") + "weight_scale")
            offset = source.tensors.get(weight.name.removesuffix("weight") + "weight_offset")
            if weight.dtype is not SafeTensorDType.INT8 or weight.shape != (outputs, inputs):
                raise ValueError("unexpected ModelSlim W8A8 weight dtype or shape")
            if scale is None or scale.dtype not in _SCALE_DTYPES or scale.shape != (outputs, 1):
                raise ValueError("ModelSlim scales must be BF16 or F32 with shape [output, 1]")
            if offset is None or offset.dtype is not scale.dtype or offset.shape != scale.shape:
                raise ValueError("ModelSlim offsets must match scale dtype and shape")
            scale_tensor = torch.frombuffer(scale.data, dtype=_SCALE_DTYPES[scale.dtype]).reshape(
                scale.shape
            )
            offset_tensor = torch.frombuffer(
                offset.data,
                dtype=_SCALE_DTYPES[offset.dtype],
            ).reshape(offset.shape)
            if not torch.all(torch.isfinite(scale_tensor) & (scale_tensor > 0)):
                raise ValueError("ModelSlim scales must be finite and positive")
            if torch.any(offset_tensor != 0):
                raise ValueError("nonzero ModelSlim weight offsets are unsupported")
            matrices.append(torch.frombuffer(weight.data, dtype=torch.int8).reshape(weight.shape))
            scales.append(scale_tensor)
            consumed.update((weight.name, scale.name, offset.name))
        if consumed != set(source.tensors):
            raise ValueError("unsupported or ambiguous Tensor in ModelSlim expert bundle")
        return AscendFusedModelSlimW8A8CpuWeights(tuple(matrices), tuple(scales))

    def make_ready_weight(
        self,
        cpu_weight: AscendFusedModelSlimW8A8CpuWeights,
        *,
        layer_id: int,
        expert_id: int,
    ) -> AscendFusedModelSlimW8A8ExpertWeights:
        if not isinstance(cpu_weight, AscendFusedModelSlimW8A8CpuWeights):
            raise TypeError("ModelSlim cached weight has the wrong object type")
        if cpu_weight.matrices[0].shape != (
            self._intermediate_dim,
            self._hidden_dim,
        ):
            raise ValueError("ModelSlim cached weight shape does not match the model")

        del layer_id, expert_id
        device = self._runtime.device
        try:
            ready = cpu_weight.copy(device=device, dtype=self._compute_dtype)
            self._runtime.capture_current_work().wait_host()
            return ready
        except torch.OutOfMemoryError as error:
            raise WeightPlacementFatalError(
                WeightPlacementFatalReason.DEVICE_OOM,
                str(error),
            ) from error
        except RuntimeError as error:
            raise WeightPlacementFatalError(
                WeightPlacementFatalReason.DEVICE_FAILURE,
                str(error),
            ) from error

    def cpu_extra_bytes(self) -> int:
        return 0

    def source_tensor_bytes(self) -> int:
        matrix_items = 3 * self._hidden_dim * self._intermediate_dim
        scale_and_offset_items = 2 * (2 * self._intermediate_dim + self._hidden_dim)
        return matrix_items + scale_and_offset_items * torch.float32.itemsize

    def ready_weight_bytes(self) -> int:
        matrix_items = 3 * self._hidden_dim * self._intermediate_dim
        scale_items = 2 * self._intermediate_dim + self._hidden_dim
        scale_dtype = _dynamic_gmm_scale_dtype(self._compute_dtype)
        return matrix_items + scale_items * scale_dtype.itemsize

    def conversion_temporary_bytes(self) -> int:
        return 0
