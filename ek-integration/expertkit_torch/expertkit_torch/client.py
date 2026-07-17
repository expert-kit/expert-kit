"""Torch model-facing client for one complete routed MoE layer."""

from __future__ import annotations

import math
import threading

import torch
from expertkit_transport.adapters.grpc import BlockingGrpcRoutedMoEClient


class RoutedMoEClient:
    """Adapt final Torch router outputs to Expert Kit Transport middleware.

    The first call fixes the Frontend device and activation dtype and starts one
    process-lifetime Transport event loop. Expert Kit returns the already
    weighted and cross-Worker aggregated `[tokens, hidden_dim]` result.
    """

    def __init__(
        self,
        controller_endpoint: str,
        *,
        instance_id: int,
        num_layers: int,
        experts_per_layer: int,
        hidden_dim: int,
        top_k: int,
        timeout_seconds: float = 6.0,
    ) -> None:
        if not controller_endpoint:
            raise ValueError("controller_endpoint must not be empty")
        for name, value in (
            ("instance_id", instance_id),
            ("num_layers", num_layers),
            ("experts_per_layer", experts_per_layer),
            ("hidden_dim", hidden_dim),
            ("top_k", top_k),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if top_k > experts_per_layer:
            raise ValueError("top_k must not exceed experts_per_layer")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")

        self._controller_endpoint = controller_endpoint
        self._instance_id = instance_id
        self._num_layers = num_layers
        self._experts_per_layer = experts_per_layer
        self._hidden_dim = hidden_dim
        self._top_k = top_k
        self._timeout_seconds = timeout_seconds
        self._transport: BlockingGrpcRoutedMoEClient | None = None
        self._device: torch.device | None = None
        self._dtype: torch.dtype | None = None
        self._lock = threading.Lock()
        self._closed = False

    def start(self, *, device: torch.device | str, dtype: torch.dtype) -> None:
        """Create fixed Frontend buffers and install the first topology snapshot."""

        resolved_device = torch.device(device)
        if dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("activation dtype must be FP16, BF16, or FP32")
        with self._lock:
            if self._closed:
                raise RuntimeError("Routed-MoE client is closed")
            if self._transport is not None:
                if resolved_device != self._device or dtype != self._dtype:
                    raise ValueError("Frontend device and activation dtype changed after startup")
                return
            transport = BlockingGrpcRoutedMoEClient(
                self._controller_endpoint,
                instance_id=self._instance_id,
                num_layers=self._num_layers,
                experts_per_layer=self._experts_per_layer,
                hidden_dim=self._hidden_dim,
                top_k=self._top_k,
                dtype=dtype,
                device=resolved_device,
            )
            try:
                transport.start(timeout_seconds=self._timeout_seconds)
            except BaseException:
                transport.close()
                raise
            self._transport = transport
            self._device = resolved_device
            self._dtype = dtype

    def forward_layer(
        self,
        *,
        layer_id: int,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Execute final assignments for one model layer.

        Args:
            layer_id: Zero-based routed layer number.
            hidden_states: Tensor shaped `[token_count, hidden_dim]`.
            expert_ids: Final assignments shaped `[token_count, top_k]`.
            routing_weights: Final routing weights shaped `[token_count, top_k]`.

        Returns:
            Weighted and aggregated activation Tensor shaped
            `[token_count, hidden_dim]`.
        """

        if hidden_states.ndim != 2 or hidden_states.shape[1] != self._hidden_dim:
            raise ValueError("hidden_states must match the configured hidden dimension")
        if expert_ids.shape != (hidden_states.shape[0], self._top_k):
            raise ValueError("expert_ids must have shape [token_count, top_k]")
        if routing_weights.shape != expert_ids.shape:
            raise ValueError("routing_weights must match expert_ids")
        if (
            expert_ids.device != hidden_states.device
            or routing_weights.device != hidden_states.device
        ):
            raise ValueError("all Routed-MoE tensors must use the activation device")

        self.start(device=hidden_states.device, dtype=hidden_states.dtype)
        transport = self._transport
        assert transport is not None
        encoded_experts = expert_ids.to(dtype=torch.int32)
        fp32_weights = routing_weights.to(dtype=torch.float32)
        return transport.execute(
            layer_id=layer_id,
            hidden_states=hidden_states,
            expert_ids=encoded_experts,
            routing_weights=fp32_weights,
            timeout_seconds=self._timeout_seconds,
        )

    def close(self) -> None:
        """Close Transport resources; repeated calls are safe."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
            transport = self._transport
            self._transport = None
        if transport is not None:
            transport.close()
