"""Computation-ready Torch expert weight objects."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from expertkit_transport.batches import ACTIVATION_DTYPES


@dataclass(frozen=True, slots=True)
class TorchExpertWeights:
    """Hold one gated FFN's projection Torch tensors on its computation device.

    Instances may represent zero-copy CPU cache views or final computation
    weights. Readiness is established by TorchWeightAdapter completing placement
    and WeightManager publishing the object into ReadyWeightTable.

    Attributes:
        gate_proj: Contiguous matrix shaped ``[intermediate_dim, hidden_dim]``.
        up_proj: Contiguous matrix shaped ``[intermediate_dim, hidden_dim]``.
        down_proj: Contiguous matrix shaped ``[hidden_dim, intermediate_dim]``.

    Note:
        These tensors are already converted to the compute dtype and device. The
        Backend never moves or converts them on the request path.
    """

    gate_proj: torch.Tensor
    up_proj: torch.Tensor
    down_proj: torch.Tensor

    def __post_init__(self) -> None:
        tensors = (self.gate_proj, self.up_proj, self.down_proj)
        if any(tensor.ndim != 2 for tensor in tensors):
            raise ValueError("Torch expert weights must be two-dimensional matrices")
        if any(tensor.dtype not in ACTIVATION_DTYPES for tensor in tensors):
            raise ValueError("Torch expert weights must use FP16, BF16, or FP32")
        if any(tensor.device != self.gate_proj.device for tensor in tensors[1:]):
            raise ValueError("Torch expert weights must be on one device")
        if any(tensor.dtype != self.gate_proj.dtype for tensor in tensors[1:]):
            raise ValueError("Torch expert weights must use one dtype")
        if any(not tensor.is_contiguous() for tensor in tensors):
            raise ValueError("Torch expert weights must be contiguous")
        if any(tensor.requires_grad for tensor in tensors):
            raise ValueError("Torch expert weights must not require gradients")

        intermediate_dim, hidden_dim = self.gate_proj.shape
        if min(intermediate_dim, hidden_dim) <= 0:
            raise ValueError("Torch expert weight dimensions must be positive")
        if self.up_proj.shape != self.gate_proj.shape:
            raise ValueError("gate and up projection shapes must match")
        if self.down_proj.shape != (hidden_dim, intermediate_dim):
            raise ValueError("down projection shape must reverse gate and up dimensions")

    def to(
        self,
        device: str | torch.device | int | None = None,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
        copy: bool = False,
        *,
        memory_format: torch.memory_format | None = None,
    ) -> TorchExpertWeights:
        return TorchExpertWeights(
            gate_proj=self.gate_proj.to(
                device=device,
                dtype=dtype,
                non_blocking=non_blocking,
                copy=copy,
                memory_format=memory_format,
            ),
            up_proj=self.up_proj.to(
                device=device,
                dtype=dtype,
                non_blocking=non_blocking,
                copy=copy,
                memory_format=memory_format,
            ),
            down_proj=self.down_proj.to(
                device=device,
                dtype=dtype,
                non_blocking=non_blocking,
                copy=copy,
                memory_format=memory_format,
            ),
        )

    @property
    def hidden_dim(self) -> int:
        """Return the FFN input and output width."""

        return self.gate_proj.shape[1]

    @property
    def intermediate_dim(self) -> int:
        """Return the gated FFN intermediate width."""

        return self.gate_proj.shape[0]

    @property
    def dtype(self) -> torch.dtype:
        """Return the already prepared computation dtype."""

        return self.gate_proj.dtype

    @property
    def device(self) -> torch.device:
        """Return the already prepared computation device."""

        return self.gate_proj.device

    @property
    def storage_bytes(self) -> int:
        """Return logical Tensor storage bytes for device-capacity accounting."""

        return sum(tensor.numel() * tensor.element_size() for tensor in self.tensors)

    @property
    def tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the projection tensors in gate, up, and down order."""

        return self.gate_proj, self.up_proj, self.down_proj
