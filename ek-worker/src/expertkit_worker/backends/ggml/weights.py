"""Computation-ready CPU weight objects backed by ggml tensor metadata."""

from __future__ import annotations

import ctypes
from contextlib import suppress

import ggml
import torch
from expertkit_transport.batches import ACTIVATION_DTYPES

_WEIGHT_CONTEXT_BYTES = 8 * 1024
_GGML_DTYPES = {
    torch.float16: ggml.GGML_TYPE_F16,
    torch.bfloat16: ggml.GGML_TYPE_BF16,
    torch.float32: ggml.GGML_TYPE_F32,
}


def _set_tensor_data(tensor: object, source: torch.Tensor) -> None:
    tensor.contents.data = ctypes.c_void_p(source.data_ptr())  # type: ignore[attr-defined]


class GgmlExpertWeights:
    """Retain CPU weight storage and ggml views for one gated FFN.

    The ggml context owns only tensor metadata. The Torch tensors own the
    immutable weight bytes, and their addresses remain stable for this object's
    lifetime.
    """

    def __init__(
        self,
        *,
        gate_proj: torch.Tensor,
        up_proj: torch.Tensor,
        down_proj: torch.Tensor,
    ) -> None:
        tensors = (gate_proj, up_proj, down_proj)
        if any(tensor.ndim != 2 for tensor in tensors):
            raise ValueError("GGML expert weights must be two-dimensional matrices")
        if any(tensor.device.type != "cpu" for tensor in tensors):
            raise ValueError("the MVP GGML backend requires CPU weights")
        if any(tensor.dtype not in ACTIVATION_DTYPES for tensor in tensors):
            raise ValueError("GGML expert weights must use FP16, BF16, or FP32")
        if any(tensor.dtype != gate_proj.dtype for tensor in tensors[1:]):
            raise ValueError("GGML expert weights must use one dtype")
        if any(not tensor.is_contiguous() for tensor in tensors):
            raise ValueError("GGML expert weights must be contiguous")
        if any(tensor.requires_grad for tensor in tensors):
            raise ValueError("GGML expert weights must not require gradients")

        intermediate_dim, hidden_dim = gate_proj.shape
        if min(intermediate_dim, hidden_dim) <= 0:
            raise ValueError("GGML expert weight dimensions must be positive")
        if up_proj.shape != gate_proj.shape:
            raise ValueError("gate and up projection shapes must match")
        if down_proj.shape != (hidden_dim, intermediate_dim):
            raise ValueError("down projection shape must reverse gate and up dimensions")

        context = ggml.ggml_init(
            ggml.ggml_init_params(
                mem_size=_WEIGHT_CONTEXT_BYTES,
                mem_buffer=None,
                no_alloc=True,
            )
        )
        if not context:
            raise MemoryError("ggml failed to allocate expert-weight metadata")
        try:
            ggml_dtype = _GGML_DTYPES[gate_proj.dtype]
            ggml_gate = self._make_view(context, gate_proj, ggml_dtype)
            ggml_up = self._make_view(context, up_proj, ggml_dtype)
            ggml_down = self._make_view(context, down_proj, ggml_dtype)
        except BaseException:
            ggml.ggml_free(context)
            raise

        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj
        self.ggml_gate = ggml_gate
        self.ggml_up = ggml_up
        self.ggml_down = ggml_down
        self._context = context

    @staticmethod
    def _make_view(context: object, tensor: torch.Tensor, ggml_dtype: int) -> object:
        view = ggml.ggml_new_tensor_2d(
            context,
            ggml_dtype,
            tensor.shape[1],
            tensor.shape[0],
        )
        if not view:
            raise MemoryError("ggml expert-weight metadata context is exhausted")
        _set_tensor_data(view, tensor)
        return view

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
        """Return the computation dtype represented by the ggml views."""

        return self.gate_proj.dtype

    @property
    def storage_bytes(self) -> int:
        """Return logical weight bytes, excluding small ggml metadata."""

        return sum(tensor.numel() * tensor.element_size() for tensor in self.tensors)

    @property
    def tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the retained projection storage in gate, up, and down order."""

        return self.gate_proj, self.up_proj, self.down_proj

    def __del__(self) -> None:
        with suppress(BaseException):
            context = self._context
            self._context = None
            if context:
                ggml.ggml_free(context)


__all__ = ["_GGML_DTYPES", "_WEIGHT_CONTEXT_BYTES", "GgmlExpertWeights"]
