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

    class RoutedDeepseekV2Moe(modeling_deepseek_v2.DeepseekV2Moe):
        """Keep the native router and shared expert while routing other experts."""

        def __init__(self, config) -> None:
            nn.Module.__init__(self)
            self.layer_id = layer_ids.take()
            self.config = config
            self.gate = nn.Linear(config.hidden_size, config.n_routed_experts, bias=False)
            shared_expert_count = config.n_shared_experts
            if shared_expert_count is None:
                self.shared_experts = None
            else:
                self.shared_experts = modeling_deepseek_v2.DeepseekV2MLP(
                    config=config,
                    intermediate_size=config.moe_intermediate_size * shared_expert_count,
                )
            self.routed_scaling_factor = config.routed_scaling_factor
            self.topk_method = config.topk_method
            self.num_group = config.n_group
            self.num_experts = config.n_routed_experts
            self.top_k = config.num_experts_per_tok
            self.topk_group = config.topk_group

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            """Execute routed experts remotely and the shared expert locally."""

            residual = hidden_states
            original_shape = hidden_states.shape
            router_logits = nn.functional.linear(
                hidden_states.type(torch.float32),
                self.gate.weight.type(torch.float32),
            )
            expert_ids, routing_weights = self.route_tokens_to_experts(router_logits)
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

    return RoutedDeepseekV2Moe
