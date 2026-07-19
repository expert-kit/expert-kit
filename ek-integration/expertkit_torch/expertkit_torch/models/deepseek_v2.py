"""DeepSeek-V2 routed MoE block for the pinned Transformers implementation."""

from __future__ import annotations

import torch
from torch import nn
from transformers.models.deepseek_v2 import modeling_deepseek_v2

from expertkit_torch.client import RoutedMoEClient
from expertkit_torch.models._common import RoutedLayerIds


def create_routed_moe_class(
    client: RoutedMoEClient,
    layer_ids: RoutedLayerIds,
) -> type[nn.Module]:
    """Create a DeepSeek-V2 MoE class bound to one model and client."""

    class RoutedDeepseekV2MoE(nn.Module):
        """Keep the native router and shared expert while routing other experts."""

        def __init__(self, config) -> None:
            super().__init__()
            self.layer_id = layer_ids.take()
            self.gate = modeling_deepseek_v2.DeepseekV2MoEGate(config)
            shared_expert_count = config.n_shared_experts
            if shared_expert_count is None:
                self.shared_experts = None
            else:
                self.shared_experts = modeling_deepseek_v2.DeepseekV2MLP(
                    config=config,
                    intermediate_size=config.moe_intermediate_size * shared_expert_count,
                )

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            """Execute routed experts remotely and the shared expert locally."""

            residual = hidden_states
            original_shape = hidden_states.shape
            expert_ids, routing_weights = self.gate(hidden_states)
            flattened = hidden_states.reshape(-1, hidden_states.shape[-1])
            routed = client.forward_layer(
                layer_id=self.layer_id,
                hidden_states=flattened,
                expert_ids=expert_ids,
                routing_weights=routing_weights,
            ).reshape(original_shape)
            if self.shared_experts is not None:
                routed = routed + self.shared_experts(residual)
            return routed

    return RoutedDeepseekV2MoE
