"""Mixtral routed MoE block for the pinned Transformers implementation."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from expertkit_torch.client import RoutedMoEClient
from expertkit_torch.models._common import RoutedLayerIds


def create_routed_moe_class(
    client: RoutedMoEClient,
    layer_ids: RoutedLayerIds,
) -> type[nn.Module]:
    """Create a Mixtral MoE class bound to one model and client."""

    class RoutedMixtralSparseMoeBlock(nn.Module):
        """Preserve Mixtral routing and return the remote aggregated result."""

        def __init__(self, config) -> None:
            super().__init__()
            self.layer_id = layer_ids.take()
            self.hidden_dim = config.hidden_size
            self.num_experts = config.num_local_experts
            self.top_k = config.num_experts_per_tok
            self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
            self.jitter_noise = config.router_jitter_noise

        def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """Route final top-k assignments through Expert Kit."""

            batch_size, sequence_length, hidden_dim = hidden_states.shape
            if self.training and self.jitter_noise > 0:
                hidden_states *= torch.empty_like(hidden_states).uniform_(
                    1.0 - self.jitter_noise,
                    1.0 + self.jitter_noise,
                )
            flattened = hidden_states.reshape(-1, hidden_dim)
            router_logits = self.gate(flattened)
            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float32)
            routing_weights, expert_ids = torch.topk(
                routing_weights,
                self.top_k,
                dim=-1,
            )
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
            routed = client.forward_layer(
                layer_id=self.layer_id,
                hidden_states=flattened,
                expert_ids=expert_ids,
                routing_weights=routing_weights,
            )
            return routed.reshape(batch_size, sequence_length, hidden_dim), router_logits

    return RoutedMixtralSparseMoeBlock
