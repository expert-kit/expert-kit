import torch
import logging
import os
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

logger = logging.getLogger(__name__)

class ExpertKitMoE(nn.Module):
    """ExpertKit MoE layer that replaces DeepseekV2MoE with remote expert computation.

    This layer has the same interface as DeepseekV2MoE but routes expert computation
    to a remote ExpertKit service via gRPC. It fails fast on any errors and uses
    raw tensor transfer without compression.
    """

    # sharing grpc client
    client: Optional[ExpertKitClient] = None

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

        self.debug_mode = os.getenv("EXPERTKIT_DEBUG_MODE", "0") == "1"

        # Extract layer ID from prefix for gRPC call
        try:
            # TODO: now hard code for Deepseek Model, layer_id may in different position
            self.layer_idx = int(prefix.split(".")[-2])
        except (IndexError, ValueError):
            self.layer_idx = 0

        # Check for required configuration
        if not hasattr(config, "expertkit_addr"):
            raise RuntimeError("Missing expertkit_addr in config")

        # Initialize gRPC client
        timeout_sec = getattr(config, "expertkit_timeout_sec", 2.0)

        if ExpertKitMoE.client is None:
            print(f"🚀ExpertKitMoE {prefix} creating new gRPC client, with addr: {config.expertkit_addr}")
            ExpertKitMoE.client = ExpertKitClient(config.expertkit_addr, timeout_sec)

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
        batch_size, hidden_dim = hidden_states.shape
        if self.debug_mode:
            print(f"🚀shape: {hidden_states.shape}, batch_size: {batch_size}, hidden_dim: {hidden_dim}")
            print(f"🚀remove duplicate shape: {torch.unique(hidden_states, dim=0).shape}")

        hidden_states = hidden_states.view(-1, hidden_dim)

        # Compute routing (same as original)
        router_logits, _ = self.gate(hidden_states)

        # Determine top-k experts for each token
        top_k = min(self.num_experts_per_tok, self.n_routed_experts)
        routing_weights, routing_indices = torch.topk(
            router_logits, top_k, dim=-1)

        # Normalize weights (if required)
        if self.norm_topk_prob:
            routing_weights = torch.softmax(routing_weights, dim=-1)
        else:
            routing_weights = torch.sigmoid(routing_weights)

        # ------------------ Optimization: Deduplicate hidden states ------------------
        # Doing this cause vllm always pad hidden_states to max_num_batched_tokens
        # Use torch.unique with return_inverse=True to get unique vectors and mapping
        hidden_flattened = hidden_states.view(batch_size, -1)
        unique_hidden, inverse_indices = torch.unique(
            hidden_flattened, dim=0, return_inverse=True)
        
        unique_batch_size = unique_hidden.shape[0]
        
        # Convert expert indices to string IDs as required by forward_expert
        expert_ids = []
        
        # If there's no significant reduction, process normally
        if unique_batch_size > batch_size * 0.8:  # Less than 80% duplication
            # Convert expert indices to string IDs
            expert_ids = []
            for seq_idx in range(batch_size):
                # Get expert indices for current token
                token_expert_indices = routing_indices[seq_idx].tolist()
                # Convert indices to string IDs
                token_expert_ids = [
                    f"{self.layer_idx}_{expert_idx}" for expert_idx in token_expert_indices]
                expert_ids.append(token_expert_ids)
            
            # Call remote expert service with all tokens
            expert_outputs = self.client.forward_expert(
                expert_ids=expert_ids,
                hidden_state=hidden_states
            )

            expert_outputs = expert_outputs.to(device=hidden_states.device, dtype=hidden_states.dtype)
        else:
            # Create a mapping from unique hidden states to all original tokens
            # that share this hidden state
            unique_to_original = [[] for _ in range(unique_batch_size)]
            for i in range(batch_size):
                unique_idx = inverse_indices[i].item()
                unique_to_original[unique_idx].append(i)
            
            # Build expert IDs for unique hidden states, but we need to consider
            # that tokens with identical hidden states might have different expert assignments
            # due to the routing decision from the gate
            unique_expert_ids = []
            
            # For each unique hidden state, gather all expert assignments from original tokens
            for unique_idx in range(unique_batch_size):
                # Get all original token indices that map to this unique hidden state
                original_indices = unique_to_original[unique_idx]
                
                # Get a representative token (just use the first one for simplicity)
                # Note: This assumes tokens with identical hidden states get identical routing decisions
                # This is likely true since the gate is deterministic on identical inputs
                rep_idx = original_indices[0]
                
                # Get expert assignments for this representative token
                token_expert_indices = routing_indices[rep_idx].tolist()
                token_expert_ids = [
                    f"{self.layer_idx}_{expert_idx}" for expert_idx in token_expert_indices]
                unique_expert_ids.append(token_expert_ids)
            
            # Call remote expert service with just the unique tokens
            unique_expert_outputs = self.client.forward_expert(
                expert_ids=unique_expert_ids,
                hidden_state=unique_hidden
            )
            
            # Map the unique outputs back to the original batch
            # Shape: [unique_batch_size, top_k, hidden_dim] -> [batch_size, top_k, hidden_dim]
            #TODO: This matrix could be created only once and reused, to enhance performance
            expert_outputs = torch.zeros(
                (batch_size, top_k, hidden_dim), 
                device=hidden_states.device, 
                dtype=hidden_states.dtype
            )
            
            for i in range(batch_size):
                unique_idx = inverse_indices[i].item()
                expert_outputs[i] = unique_expert_outputs[unique_idx]
        
        # Calculate weighted sum (same as original)
        # Expand routing_weights to [num_tokens, top_k, 1] for broadcast calculation
        expanded_weights = routing_weights.unsqueeze(-1)

        # Compute weighted sum across expert dimension: [num_tokens, hidden_dim]
        if self.debug_mode:
            print(f"🚀 {expanded_weights.device}, {expert_outputs.device}, {hidden_states.device} ")
            print(f"🚀 {expanded_weights.dtype}, {expert_outputs.dtype}, {hidden_states.dtype} ")
        output = torch.sum(expanded_weights * expert_outputs, dim=1)

        return output.view(batch_size, hidden_dim)