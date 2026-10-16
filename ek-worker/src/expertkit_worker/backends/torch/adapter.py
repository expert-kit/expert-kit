"""Torch conversion from validated SafeTensors regions to ready expert weights."""

from __future__ import annotations

import torch
from expertkit_transport.batches import ACTIVATION_DTYPES

from expertkit_worker.backends.torch.weights import (
    TorchExpertWeights,
    TorchGPTQCpuWeight,
    dequantize_gptq,
)
from expertkit_worker.device import CpuWorkerRuntime, CudaWorkerRuntime, WorkerDeviceRuntime
from expertkit_worker.weights.adapter import (
    WeightAdapter,
    WeightPlacementFatalError,
    WeightPlacementFatalReason,
)
from expertkit_worker.weights.format import SafeTensorData, SafeTensorDType, SafeTensorRegion

_TORCH_DTYPES = {
    SafeTensorDType.FP16: torch.float16,
    SafeTensorDType.BF16: torch.bfloat16,
    SafeTensorDType.FP32: torch.float32,
}


def _runtime_for_device(device: torch.device) -> WorkerDeviceRuntime:
    """Create a compatibility runtime for direct adapter construction."""

    if device.type == "cpu":
        return CpuWorkerRuntime(device)
    if device.type == "cuda":
        return CudaWorkerRuntime(device)
    if device.type == "npu":
        from expertkit_worker.device.ascend import AscendWorkerRuntime

        return AscendWorkerRuntime(device)
    raise ValueError(f"unsupported Torch device: {device}")


class TorchGPTQWeightAdapter(WeightAdapter[TorchGPTQCpuWeight, TorchExpertWeights]):
    """Load symmetric AutoGPTQ v1 INT4 weights into a floating-point ready cache.

    This compatibility path accepts FP16 scales and canonical groups only. It
    dequantizes on CPU during placement; the request path uses ordinary GEMMs.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        intermediate_dim: int,
        group_size: int,
        device: torch.device | str,
        compute_dtype: torch.dtype = torch.float16,
    ) -> None:
        for value in (hidden_dim, intermediate_dim, group_size):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("GPTQ dimensions and group_size must be positive integers")
        if any(width % 8 or width % group_size for width in (hidden_dim, intermediate_dim)):
            raise ValueError("GPTQ widths must be divisible by 8 and group_size")
        self._float_adapter = TorchWeightAdapter(
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            source_dtype=torch.float16,
            compute_dtype=compute_dtype,
            device=device,
        )
        self._hidden_dim = hidden_dim
        self._intermediate_dim = intermediate_dim
        self._group_size = group_size
        self._compute_dtype = compute_dtype

    @property
    def backend_name(self) -> str:
        """Return the adapter name used for startup diagnostics."""
        return "torch-gptq-dequantize"

    def _matrix(
        self,
        source: SafeTensorData,
        roles: tuple[str, str],
        inputs: int,
        outputs: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        values = []
        groups = inputs // self._group_size
        for name, dtype, shape in (
            ("qweight", SafeTensorDType.INT32, (inputs // 8, outputs)),
            ("qzeros", SafeTensorDType.INT32, (groups, outputs // 8)),
            ("scales", SafeTensorDType.FP16, (groups, outputs)),
        ):
            region = source.find_unique_suffix(tuple(f"{role}.{name}" for role in roles))
            if region.dtype is not dtype or region.shape != shape:
                raise ValueError(f"unexpected GPTQ {name} dtype or shape")
            tdtype = torch.float16 if dtype is SafeTensorDType.FP16 else torch.int32
            values.append(torch.frombuffer(region.data, dtype=tdtype).reshape(shape))
        if not torch.all(values[1] == 0x77777777):
            raise ValueError("GPTQ symmetric v1 zero points must decode to 8")
        if not torch.all(torch.isfinite(values[2]) & (values[2] > 0)):
            raise ValueError("GPTQ scales must be finite and positive")
        indices = [
            r
            for r in source.tensors.values()
            if any(r.name.endswith(f"{role}.g_idx") for role in roles)
        ]
        if len(indices) > 1:
            raise ValueError("ambiguous GPTQ g_idx")
        if indices:
            region = indices[0]
            if region.dtype is not SafeTensorDType.INT32 or region.shape != (inputs,):
                raise ValueError("unexpected GPTQ g_idx dtype or shape")
            index = torch.frombuffer(region.data, dtype=torch.int32)
            if not torch.equal(index, torch.arange(inputs, dtype=torch.int32) // self._group_size):
                raise ValueError("GPTQ activation-order groups are unsupported")
        return tuple(values)

    def make_cpu_weight(self, source: SafeTensorData) -> TorchGPTQCpuWeight:
        """Validate packed CPU views, including layout and optional group indices."""
        roles = (("gate_proj", "w1"), ("up_proj", "w3"), ("down_proj", "w2"))
        allowed = tuple(
            f"{role}.{field}"
            for pair in roles
            for role in pair
            for field in ("qweight", "qzeros", "scales", "g_idx")
        )
        if any(not name.endswith(allowed) for name in source.tensors):
            raise ValueError("unsupported Tensor in GPTQ expert bundle")
        hidden, inter = self._hidden_dim, self._intermediate_dim
        return TorchGPTQCpuWeight(
            gate=self._matrix(source, roles[0], hidden, inter),
            up=self._matrix(source, roles[1], hidden, inter),
            down=self._matrix(source, roles[2], inter, hidden),
        )

    def make_ready_weight(
        self,
        cpu_weight: TorchGPTQCpuWeight,
        *,
        layer_id: int,
        expert_id: int,
    ) -> TorchExpertWeights:
        """Dequantize on CPU and use the ordinary adapter's synchronized placement."""
        matrices = [
            dequantize_gptq(
                *matrix,
                in_features=self._hidden_dim if i != 2 else self._intermediate_dim,
                out_features=self._intermediate_dim if i != 2 else self._hidden_dim,
                group_size=self._group_size,
            )
            for i, matrix in enumerate((cpu_weight.gate, cpu_weight.up, cpu_weight.down))
        ]
        decoded = TorchExpertWeights(*matrices)
        return self._float_adapter.make_ready_weight(
            decoded, layer_id=layer_id, expert_id=expert_id
        )

    def cpu_extra_bytes(self) -> int:
        """Return additional resident CPU bytes beyond the serialized source."""
        return 0

    def source_tensor_bytes(self) -> int:
        """Bound INT4 words, FP16 scales, and optional INT32 group indices."""
        hidden, inter = self._hidden_dim, self._intermediate_dim
        return sum(
            inputs * outputs // 2
            + (inputs // self._group_size) * outputs // 2
            + (inputs // self._group_size) * outputs * 2
            + inputs * 4
            for inputs, outputs in ((hidden, inter), (hidden, inter), (inter, hidden))
        )

    def ready_weight_bytes(self) -> int:
        """Return the floating-point device allocation size."""
        return self._float_adapter.ready_weight_bytes()

    def conversion_temporary_bytes(self) -> int:
        """Use the ordinary adapter's device conversion reservation."""
        return self._float_adapter.conversion_temporary_bytes()

    def host_conversion_temporary_bytes(self) -> int:
        """Bound retained FP16 matrices, integer expansion, scales, and group indices."""
        elements = self._hidden_dim * self._intermediate_dim
        return (
            3 * elements * 2
            + 64 * elements
            + 16 * max(self._hidden_dim, self._intermediate_dim)
            + 128
        )


class TorchWeightAdapter(WeightAdapter[TorchExpertWeights, TorchExpertWeights]):
    """Build zero-copy CPU views and final-device Torch expert objects."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        intermediate_dim: int,
        source_dtype: torch.dtype,
        compute_dtype: torch.dtype,
        runtime: WorkerDeviceRuntime | None = None,
        device: torch.device | str | None = None,
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

        self._hidden_dim = hidden_dim
        self._intermediate_dim = intermediate_dim
        self._source_dtype = source_dtype
        self._compute_dtype = compute_dtype
        if runtime is None:
            if device is None:
                raise TypeError("TorchWeightAdapter requires runtime or device")
            runtime = _runtime_for_device(torch.device(device))
        self._runtime = runtime

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

    def make_ready_weight(
        self,
        cpu_weight: TorchExpertWeights,
        *,
        layer_id: int,
        expert_id: int,
    ) -> TorchExpertWeights:
        """Copy or convert CPU views directly into final Torch tensors."""

        del layer_id, expert_id
        if cpu_weight.device.type != "cpu":
            raise ValueError("Torch cached weight must be on CPU")
        if cpu_weight.dtype != self._source_dtype:
            raise ValueError("Torch cached weight dtype does not match the configured source")

        runtime = self._runtime
        device = runtime.device
        try:
            # H2D with copy, or get a view if device is CPU
            copy = device.type != "cpu" or self._compute_dtype != self._source_dtype
            ready = cpu_weight.to(
                device=device,
                dtype=self._compute_dtype,
                copy=copy,
            )

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
        """Return zero because CPU Tensors view the retained source buffer."""

        return 0

    def source_tensor_bytes(self) -> int:
        """Return the exact encoded bytes of the three source projection Tensors."""

        elements = 3 * self._hidden_dim * self._intermediate_dim
        return elements * torch.empty((), dtype=self._source_dtype).element_size()

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
