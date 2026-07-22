"""DeepSeek-V3 routed MoE block for the pinned Transformers implementation."""

from __future__ import annotations

import torch
from torch import nn
from transformers.models.deepseek_v3 import modeling_deepseek_v3

from expertkit_torch.client import RoutedMoEClient
from expertkit_torch.models._common import RoutedLayerIds


def create_routed_moe_class(
    client: RoutedMoEClient,
    layer_ids: RoutedLayerIds,
) -> type[nn.Module]:
    """Create a DeepSeek-V3 MoE class bound to one model and client."""

    class RoutedDeepseekV3MoE(modeling_deepseek_v3.DeepseekV3MoE):
        """Keep the native router and shared expert while routing other experts."""

        def __init__(self, config) -> None:
            nn.Module.__init__(self)
            self.layer_id = layer_ids.take()
            self.config = config
            self.gate = modeling_deepseek_v3.DeepseekV3TopkRouter(config)
            self.shared_experts = modeling_deepseek_v3.DeepseekV3MLP(
                config=config,
                intermediate_size=config.moe_intermediate_size * config.n_shared_experts,
            )
            self.n_routed_experts = config.n_routed_experts
            self.n_group = config.n_group
            self.topk_group = config.topk_group
            self.norm_topk_prob = config.norm_topk_prob
            self.routed_scaling_factor = config.routed_scaling_factor
            self.top_k = config.num_experts_per_tok

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            """Execute routed experts remotely and the shared expert locally."""

            residual = hidden_states
            original_shape = hidden_states.shape
            router_logits = self.gate(hidden_states)
            expert_ids, routing_weights = self.route_tokens_to_experts(router_logits)
            flattened = hidden_states.reshape(-1, hidden_states.shape[-1])
            routed = client.forward_layer(
                layer_id=self.layer_id,
                hidden_states=flattened,
                expert_ids=expert_ids,
                routing_weights=routing_weights,
            ).reshape(original_shape)
            return routed + self.shared_experts(residual)

    return RoutedDeepseekV3MoE
