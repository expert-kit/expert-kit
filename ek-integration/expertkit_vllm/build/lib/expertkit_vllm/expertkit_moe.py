import torch
from torch import nn
from typing import Optional, List
from transformers import PretrainedConfig
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce
)
from vllm.model_executor.layers.linear import ReplicatedLinear
from .grpc_client import ExpertKitClient


class ExpertKitMoE(nn.Module):
    """ExpertKit MoE layer that replaces DeepseekV2MoE with remote expert computation.

    This layer has the same interface as DeepseekV2MoE but routes expert computation
    to a remote ExpertKit service via gRPC. It fails fast on any errors and uses
    raw tensor transfer without compression.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_shared_experts = config.n_shared_experts

        # Extract layer ID from prefix for gRPC call
        try:
            self.layer_id = int(prefix.split(".")[-2])
        except (IndexError, ValueError):
            self.layer_id = 0

        # Check for required configuration
        if not hasattr(config, "expertkit_addr"):
            raise RuntimeError("Missing expertkit_addr in config")

        # Initialize gRPC client
        timeout_sec = getattr(config, "expertkit_timeout_sec", 2.0)
        self.client = ExpertKitClient(config.expertkit_addr, timeout_sec)

        # We still need the gate for routing
        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.n_routed_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate"
        )

        if config.topk_method == "noaux_tc":
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(config.n_routed_experts)
            )
        else:
            self.gate.e_score_correction_bias = None

        # Store config values
        self.hidden_size = config.hidden_size
        self.n_routed_experts = config.n_routed_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass using remote expert computation.

        Args:
            hidden_states: Input tensor [num_tokens, hidden_dim]

        Returns:
            Output tensor after expert computation

        Raises:
            RuntimeError: On any remote computation failure
        """
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # Compute routing (same as original)
        router_logits, _ = self.gate(hidden_states)

        # Determine top-k experts for each token
        top_k = min(self.num_experts_per_tok, self.n_routed_experts)
        routing_weights, routing_indices = torch.topk(
            router_logits, top_k, dim=-1)
        
        print("routing indices: ", routing_indices)

        # Normalize weights (if required)
        if self.norm_topk_prob:
            routing_weights = torch.softmax(routing_weights, dim=-1)
        else:
            routing_weights = torch.sigmoid(routing_weights)

        # For each token, compute weighted sum of expert outputs
        # Initialize output with zeros
        output = torch.zeros_like(hidden_states)

        # For each expert, collect tokens that route to it
        for expert_idx in range(self.n_routed_experts):
            # Find which tokens route to this expert and their positions
            mask = routing_indices == expert_idx
            if not mask.any():
                continue  # Skip if no tokens route to this expert

            # Get positions and weights for this expert
            positions = torch.nonzero(mask)
            token_indices = positions[:, 0]  # Which tokens
            weight_indices = positions[:, 1]  # Which position in the top-k

            # Get weights for these positions
            weights = routing_weights[token_indices,
                                      weight_indices].unsqueeze(1)

            # Get input for this expert
            expert_input = hidden_states[token_indices]

            # Forward through the remote expert
            expert_output = self.client.forward_expert(
                layer=self.layer_id,
                idx=expert_idx,
                hidden_state=expert_input
            )

            # Add weighted expert output to the result
            output[token_indices] += weights * expert_output

        # Apply tensor model parallel reduction if needed
        if self.tp_size > 1:
            output = tensor_model_parallel_all_reduce(output)

        return output.view(num_tokens, hidden_dim)
