import torch
import logging
from torch import nn
from typing import Optional, List, Callable
from transformers import PretrainedConfig
from vllm.model_executor.layers.quantization import QuantizationConfig
from expertkit_vllm.grpc_client import ExpertKitClient

logger = logging.getLogger(__name__)

class GrpcExpert(nn.Module):
    """GrpcExpert Expert layer that handles remote expert computation.
    
    This layer handles the remote expert computation via gRPC and is designed
    to be used independently or within the MoE architecture.
    """

    # sharing grpc client across instances
    client: Optional[ExpertKitClient] = None

    def __init__(
        self,
        num_experts: int,  # Global number of experts
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        prefix: str = "",
        *args,

        expertkit_addr: Optional[str] = None,
        expertkit_timeout_sec: float = 2.0,
        debug_mode: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        reduce_results: bool = False,
        renormalize: bool = True,
        use_grouped_topk: bool = False,
        num_expert_group: Optional[int] = None,
        topk_group: Optional[int] = None,
        quant_config: Optional[QuantizationConfig] = None,
        tp_size: Optional[int] = None,
        ep_size: Optional[int] = None,
        dp_size: Optional[int] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        activation: str = "silu",
        **kwargs,
    ):
        """Initialize GrpcExpert.
        
        Args:
            num_experts: Global number of experts
            top_k: Number of experts to route each token to
            hidden_size: Size of the hidden dimension
            intermediate_size: Size of the intermediate dimension
            params_dtype: Optional dtype for parameters
            reduce_results: Whether to reduce results across TP ranks
            renormalize: Whether to renormalize routing weights
            use_grouped_topk: Whether to use grouped topk
            num_expert_group: Number of expert groups
            topk_group: Top-k group
            quant_config: Quantization configuration
            tp_size: Tensor parallel size
            ep_size: Expert parallel size
            dp_size: Data parallel size
            prefix: Prefix for naming
            custom_routing_function: Custom routing function
            scoring_func: Scoring function
            e_score_correction_bias: Expert score correction bias
            activation: Activation function
            expertkit_addr: Address of the ExpertKit service
            expertkit_timeout_sec: Timeout for gRPC calls
            debug_mode: Whether to enable debug logging
        """
        super().__init__()
        
        # Store necessary parameters
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.prefix = prefix
        self.debug_mode = debug_mode
        
        # Extract layer ID from prefix for gRPC call
        try:
            # Extract layer index from the prefix
            # Assuming format like "model.layers.12.mlp.experts"
            self.layer_idx = int(prefix.split(".")[-3])
        except (IndexError, ValueError):
            self.layer_idx = 0
            logger.warning(f"Could not extract layer index from prefix '{prefix}', using default 0")
        
        # Initialize gRPC client if not already initialized
        if expertkit_addr is None:
            raise RuntimeError("Missing expertkit_addr in config")
            
        if GrpcExpert.client is None:
            logger.info(f"🚀 GrpcExpert {prefix} creating new gRPC client, with addr: {expertkit_addr}")
            GrpcExpert.client = ExpertKitClient(expertkit_addr, expertkit_timeout_sec)
    
    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor) -> torch.Tensor:
        """Forward pass using remote expert computation.

        Args:
            hidden_states: Input tensor [num_tokens, hidden_dim]
            router_logits: Router logits [num_tokens, num_experts]

        Returns:
            Output tensor after expert computation [num_tokens, hidden_dim]

        Raises:
            RuntimeError: On any remote computation failure
        """
        batch_size, hidden_dim = hidden_states.shape
        if self.debug_mode:
            logger.debug(f"🚀 Hidden states shape: {hidden_states.shape}, batch_size: {batch_size}, hidden_dim: {hidden_dim}")
            logger.debug(f"🚀 Router logits shape: {router_logits.shape}")
            
        # Get top-k experts for each token based on router_logits
        routing_weights, routing_indices = torch.topk(
            router_logits, self.top_k, dim=-1)
            
        # ------------------ Optimization: Deduplicate hidden states ------------------
        # Doing this because vllm always pads hidden_states to max_num_batched_tokens
        hidden_flattened = hidden_states.view(batch_size, -1)
        unique_hidden, inverse_indices = torch.unique(
            hidden_flattened, dim=0, return_inverse=True)
        
        unique_batch_size = unique_hidden.shape[0]
        if self.debug_mode:
            logger.debug(f"🚀 unique_batch_size: {unique_batch_size}, batch_size: {batch_size}")
        
        # Convert expert indices to string IDs as required by forward_expert
        expert_ids = []
        
        # If there's no significant reduction, process normally
        if unique_batch_size > 0.8 * batch_size:  # Unique more than 80%(dominant), send directly
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
            # Optimization for cases with many duplicate hidden states
            
            # Create a mapping from unique hidden states to all original tokens
            # that share this hidden state
            unique_to_original = [[] for _ in range(unique_batch_size)]
            for i in range(batch_size):
                unique_idx = inverse_indices[i].item()
                unique_to_original[unique_idx].append(i)
            
            # Build expert IDs for unique hidden states
            unique_expert_ids = []
            
            # For each unique hidden state, gather all expert assignments from original tokens
            for unique_idx in range(unique_batch_size):
                # Get all original token indices that map to this unique hidden state
                original_indices = unique_to_original[unique_idx]
                
                # Get a representative token (just use the first one)
                # This assumes tokens with identical hidden states get identical routing decisions
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
            expert_outputs = torch.zeros(
                (batch_size, self.top_k, hidden_dim), 
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