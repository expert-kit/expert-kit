"""Fixed CUDA storage and ready objects for the experimental fused Backend."""

from __future__ import annotations

import threading
from dataclasses import dataclass

import torch

from expertkit_worker.weights.adapter import (
    WeightAdapter,
    WeightPlacementFatalError,
    WeightPlacementFatalReason,
)
from expertkit_worker.weights.format import (
    SafeTensorData,
    SafeTensorDType,
    SafeTensorRegion,
)

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16}
_TORCH_DTYPES = {
    SafeTensorDType.FP16: torch.float16,
    SafeTensorDType.BF16: torch.bfloat16,
    SafeTensorDType.FP32: torch.float32,
}


def _validate_positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate_nvidia_cuda_device(device: torch.device) -> None:
    if device.type != "cuda" or device.index is None:
        raise ValueError("the fused Backend requires one indexed NVIDIA CUDA device")
    if torch.version.hip is not None:
        raise ValueError("the fused Backend does not support ROCm")
    if not torch.cuda.is_available():
        raise RuntimeError("the fused Backend requires an available NVIDIA CUDA device")
    try:
        torch.cuda.get_device_properties(device)
    except RuntimeError as error:
        raise RuntimeError("the fused Backend cannot access its NVIDIA CUDA device") from error


@dataclass(frozen=True, slots=True)
class FusedCpuWeights:
    """Hold one parsed expert as CPU Tensor views."""

    gate_proj: torch.Tensor
    up_proj: torch.Tensor
    down_proj: torch.Tensor

    def __post_init__(self) -> None:
        tensors = self.tensors
        if any(tensor.ndim != 2 for tensor in tensors):
            raise ValueError("fused expert weights must be two-dimensional matrices")
        if any(tensor.device.type != "cpu" for tensor in tensors):
            raise ValueError("cached fused expert weights must remain on CPU")
        if any(tensor.dtype not in _SUPPORTED_DTYPES for tensor in tensors):
            raise ValueError("fused expert weights must use FP16 or BF16")
        if any(tensor.dtype != self.gate_proj.dtype for tensor in tensors[1:]):
            raise ValueError("fused expert weights must use one dtype")
        if any(not tensor.is_contiguous() for tensor in tensors):
            raise ValueError("fused expert weights must be contiguous")
        if any(tensor.requires_grad for tensor in tensors):
            raise ValueError("fused expert weights must not require gradients")

        intermediate_dim, hidden_dim = self.gate_proj.shape
        if min(intermediate_dim, hidden_dim) <= 0:
            raise ValueError("fused expert weight dimensions must be positive")
        if self.up_proj.shape != self.gate_proj.shape:
            raise ValueError("fused gate and up projection shapes must match")
        if self.down_proj.shape != (hidden_dim, intermediate_dim):
            raise ValueError("fused down projection shape must reverse gate and up dimensions")

    @property
    def dtype(self) -> torch.dtype:
        """Return the source weight dtype."""

        return self.gate_proj.dtype

    @property
    def tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the gate, up, and down matrices."""

        return self.gate_proj, self.up_proj, self.down_proj


class FusedWeightStorage:
    """Own process-lifetime packed weights and the stable expert-to-slot map."""

    def __init__(
        self,
        *,
        max_experts: int,
        num_layers: int,
        experts_per_layer: int,
        hidden_dim: int,
        intermediate_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        for name, value in (
            ("max_experts", max_experts),
            ("num_layers", num_layers),
            ("experts_per_layer", experts_per_layer),
            ("hidden_dim", hidden_dim),
            ("intermediate_dim", intermediate_dim),
        ):
            _validate_positive_integer(name, value)
        if dtype not in _SUPPORTED_DTYPES:
            raise ValueError("fixed fused storage must use FP16 or BF16")
        _validate_nvidia_cuda_device(device)

        self.max_experts = max_experts
        self.num_layers = num_layers
        self.experts_per_layer = experts_per_layer
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.dtype = dtype
        self.device = device
        self.gate_up = torch.empty(
            (max_experts, 2 * intermediate_dim, hidden_dim),
            dtype=dtype,
            device=device,
        )
        self.down = torch.empty(
            (max_experts, hidden_dim, intermediate_dim),
            dtype=dtype,
            device=device,
        )
        self.slot_mapping = torch.full(
            (num_layers, experts_per_layer),
            -1,
            dtype=torch.int32,
            device=device,
        )
        torch.cuda.current_stream(device).synchronize()
        self._owners: list[tuple[int, int] | None] = [None] * max_experts
        self._positions: list[list[int]] = [[-1] * experts_per_layer for _ in range(num_layers)]
        self._lock = threading.Lock()

    def claim(self, layer_id: int, expert_id: int) -> int:
        """Reserve one empty slot for a stable model position."""

        self._validate_position(layer_id, expert_id)
        with self._lock:
            if self._positions[layer_id][expert_id] >= 0:
                raise RuntimeError("fused expert position already owns a slot")
            try:
                slot = self._owners.index(None)
            except ValueError as error:
                raise RuntimeError("no free fused weight slot") from error
            self._owners[slot] = (layer_id, expert_id)
            self._positions[layer_id][expert_id] = slot
            return slot

    def publish_mapping(self, layer_id: int, expert_id: int, slot: int) -> None:
        """Publish one fully copied slot to device-side route mapping."""

        with self._lock:
            self._validate_owner(layer_id, expert_id, slot)
            self.slot_mapping[layer_id, expert_id].fill_(slot)

    def release(self, layer_id: int, expert_id: int, slot: int) -> None:
        """Clear device mapping before making a slot reusable."""

        with self._lock:
            self._validate_owner(layer_id, expert_id, slot)
            self.slot_mapping[layer_id, expert_id].fill_(-1)
            torch.cuda.current_stream(self.device).synchronize()
            self._positions[layer_id][expert_id] = -1
            self._owners[slot] = None

    def abandon(self, layer_id: int, expert_id: int, slot: int) -> None:
        """Return a claimed slot whose weight conversion did not finish."""

        with self._lock:
            self._validate_owner(layer_id, expert_id, slot)
            self.slot_mapping[layer_id, expert_id].fill_(-1)
            torch.cuda.current_stream(self.device).synchronize()
            self._positions[layer_id][expert_id] = -1
            self._owners[slot] = None

    def slot_for(self, layer_id: int, expert_id: int) -> int:
        """Return the Host-side slot mapping without a device synchronization."""

        self._validate_position(layer_id, expert_id)
        with self._lock:
            return self._positions[layer_id][expert_id]

    def mapping_for_layer(self, layer_id: int) -> torch.Tensor:
        """Return the persistent device mapping row for one routed layer."""

        if isinstance(layer_id, bool) or not isinstance(layer_id, int):
            raise ValueError("layer_id must be an integer")
        if not 0 <= layer_id < self.num_layers:
            raise ValueError("layer_id exceeds fused weight storage")
        return self.slot_mapping[layer_id]

    def _validate_position(self, layer_id: int, expert_id: int) -> None:
        if (
            isinstance(layer_id, bool)
            or not isinstance(layer_id, int)
            or not 0 <= layer_id < self.num_layers
        ):
            raise ValueError("layer_id exceeds fused weight storage")
        if (
            isinstance(expert_id, bool)
            or not isinstance(expert_id, int)
            or not 0 <= expert_id < self.experts_per_layer
        ):
            raise ValueError("expert_id exceeds fused weight storage")

    def _validate_owner(self, layer_id: int, expert_id: int, slot: int) -> None:
        self._validate_position(layer_id, expert_id)
        if isinstance(slot, bool) or not isinstance(slot, int):
            raise ValueError("slot must be an integer")
        if not 0 <= slot < self.max_experts:
            raise ValueError("slot exceeds fused weight storage")
        if self._owners[slot] != (layer_id, expert_id):
            raise RuntimeError("fused slot owner changed unexpectedly")
        if self._positions[layer_id][expert_id] != slot:
            raise RuntimeError("fused position mapping changed unexpectedly")


@dataclass(frozen=True, slots=True)
class FusedExpertWeights:
    """Reference one occupied slot in process-lifetime fused storage."""

    storage: FusedWeightStorage
    layer_id: int
    expert_id: int
    slot: int

    def __post_init__(self) -> None:
        if self.storage.slot_for(self.layer_id, self.expert_id) != self.slot:
            raise ValueError("fused ready weight does not match its fixed slot")

    @property
    def hidden_dim(self) -> int:
        """Return the FFN input and output width."""

        return self.storage.hidden_dim

    @property
    def intermediate_dim(self) -> int:
        """Return the gated FFN intermediate width."""

        return self.storage.intermediate_dim

    @property
    def dtype(self) -> torch.dtype:
        """Return the computation dtype."""

        return self.storage.dtype

    @property
    def device(self) -> torch.device:
        """Return the selected CUDA device."""

        return self.storage.device


class FusedWeightAdapter(WeightAdapter[FusedCpuWeights, FusedExpertWeights]):
    """Parse experts and copy them directly into preallocated fused slots."""

    def __init__(
        self,
        *,
        num_layers: int,
        experts_per_layer: int,
        hidden_dim: int,
        intermediate_dim: int,
        source_dtype: torch.dtype,
        compute_dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        for name, value in (
            ("num_layers", num_layers),
            ("experts_per_layer", experts_per_layer),
            ("hidden_dim", hidden_dim),
            ("intermediate_dim", intermediate_dim),
        ):
            _validate_positive_integer(name, value)
        if source_dtype not in _SUPPORTED_DTYPES or compute_dtype not in _SUPPORTED_DTYPES:
            raise ValueError("the fused Backend supports only FP16 or BF16 weights")
        resolved_device = torch.device(device)
        _validate_nvidia_cuda_device(resolved_device)

        self._num_layers = num_layers
        self._experts_per_layer = experts_per_layer
        self._hidden_dim = hidden_dim
        self._intermediate_dim = intermediate_dim
        self._source_dtype = source_dtype
        self._compute_dtype = compute_dtype
        self._device = resolved_device
        self._storage: FusedWeightStorage | None = None

    @property
    def backend_name(self) -> str:
        """Return the built-in Backend name."""

        return "fused"

    def make_cpu_weight(self, source: SafeTensorData) -> FusedCpuWeights:
        """Create zero-copy CPU views over validated SafeTensors regions."""

        gate = source.find_unique_suffix(("gate_proj.weight", "w1.weight"))
        up = source.find_unique_suffix(("up_proj.weight", "w3.weight"))
        down = source.find_unique_suffix(("down_proj.weight", "w2.weight"))
        return FusedCpuWeights(
            gate_proj=self._view(gate, (self._intermediate_dim, self._hidden_dim)),
            up_proj=self._view(up, (self._intermediate_dim, self._hidden_dim)),
            down_proj=self._view(down, (self._hidden_dim, self._intermediate_dim)),
        )

    def make_ready_weight(
        self,
        cpu_weight: FusedCpuWeights,
        *,
        layer_id: int,
        expert_id: int,
    ) -> FusedExpertWeights:
        """Copy one CPU expert directly into an empty final CUDA slot."""

        if not isinstance(cpu_weight, FusedCpuWeights):
            raise TypeError("fused cached weight has the wrong object type")
        if cpu_weight.dtype != self._source_dtype:
            raise ValueError("fused cached weight dtype does not match the configured source")
        if cpu_weight.gate_proj.shape != (
            self._intermediate_dim,
            self._hidden_dim,
        ):
            raise ValueError("fused cached weight shape does not match the configured model")
        storage = self._require_storage()
        slot = storage.claim(layer_id, expert_id)
        try:
            storage.gate_up[slot, : self._intermediate_dim].copy_(cpu_weight.gate_proj)
            storage.gate_up[slot, self._intermediate_dim :].copy_(cpu_weight.up_proj)
            storage.down[slot].copy_(cpu_weight.down_proj)
            storage.publish_mapping(layer_id, expert_id, slot)
            torch.cuda.current_stream(self._device).synchronize()
            return FusedExpertWeights(storage, layer_id, expert_id, slot)
        except torch.OutOfMemoryError as error:
            storage.abandon(layer_id, expert_id, slot)
            raise WeightPlacementFatalError(
                WeightPlacementFatalReason.DEVICE_OOM,
                str(error),
            ) from error
        except RuntimeError as error:
            storage.abandon(layer_id, expert_id, slot)
            raise WeightPlacementFatalError(
                WeightPlacementFatalReason.DEVICE_FAILURE,
                str(error),
            ) from error
        except BaseException:
            storage.abandon(layer_id, expert_id, slot)
            raise

    def initialize_ready_storage(self, max_experts: int) -> None:
        """Allocate all fixed weight slots and mapping metadata at startup."""

        super().initialize_ready_storage(max_experts)
        if self._storage is not None:
            if self._storage.max_experts != max_experts:
                raise RuntimeError("fused weight storage was initialized with another capacity")
            return
        try:
            self._storage = FusedWeightStorage(
                max_experts=max_experts,
                num_layers=self._num_layers,
                experts_per_layer=self._experts_per_layer,
                hidden_dim=self._hidden_dim,
                intermediate_dim=self._intermediate_dim,
                dtype=self._compute_dtype,
                device=self._device,
            )
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

    def release_ready_weight(self, ready_weight: FusedExpertWeights) -> None:
        """Clear one mapping entry and make its fixed slot reusable."""

        if not isinstance(ready_weight, FusedExpertWeights):
            raise TypeError("fused ready weight has the wrong object type")
        storage = self._require_storage()
        if ready_weight.storage is not storage:
            raise RuntimeError("fused ready weight belongs to another storage allocation")
        storage.release(
            ready_weight.layer_id,
            ready_weight.expert_id,
            ready_weight.slot,
        )

    def cpu_extra_bytes(self) -> int:
        """Return zero because parsed CPU Tensors view retained source bytes."""

        return 0

    def source_tensor_bytes(self) -> int:
        """Return encoded bytes for the three unquantized source matrices."""

        elements = 3 * self._hidden_dim * self._intermediate_dim
        return elements * torch.empty((), dtype=self._source_dtype).element_size()

    def ready_weight_bytes(self) -> int:
        """Return fixed storage bytes consumed by one expert slot."""

        elements = 3 * self._hidden_dim * self._intermediate_dim
        return elements * torch.empty((), dtype=self._compute_dtype).element_size()

    def conversion_temporary_bytes(self) -> int:
        """Return zero because conversion copies directly into the final slot."""

        return 0

    def _require_storage(self) -> FusedWeightStorage:
        storage = self._storage
        if storage is None:
            raise RuntimeError("fused weight storage has not been initialized")
        return storage

    def _view(self, region: SafeTensorRegion, shape: tuple[int, int]) -> torch.Tensor:
        if _TORCH_DTYPES[region.dtype] != self._source_dtype:
            raise ValueError(f"weight Tensor {region.name!r} has an unexpected dtype")
        if region.shape != shape:
            raise ValueError(f"weight Tensor {region.name!r} has an unexpected shape")
        return torch.frombuffer(region.data, dtype=self._source_dtype).reshape(shape)
