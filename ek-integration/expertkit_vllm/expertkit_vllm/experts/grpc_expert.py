import torch
import logging
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter
from typing import Optional, List, Callable
from vllm.model_executor.layers.quantization import QuantizationConfig
from expertkit_vllm.grpc_client import ExpertKitClient
from expertkit_vllm.utils.config import collect_ek_client_cfg
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.models.utils import PPMissingLayer

logger = logging.getLogger(__name__)

class GrpcExpert(PPMissingLayer):
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
        # collect ek config
        ek_cfg = collect_ek_client_cfg()

        # Store necessary parameters
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.prefix = prefix
        self.ek_model_name = ek_cfg.ek_model_name
        self.debug_mode = ek_cfg.ek_debug_mode

        # essential params for expert_select
        self.renormalize=renormalize
        self.topk_group=topk_group
        self.num_expert_group=num_expert_group
        self.custom_routing_function = custom_routing_function
        self.scoring_func = scoring_func
        self.e_score_correction_bias = e_score_correction_bias

        if self.debug_mode:
            logger.setLevel(logging.DEBUG)
            print(f"🚀 GrpcExpert initialized with prefix: {prefix}, num_experts: {num_experts}, top_k: {top_k}, hidden_size: {hidden_size}")
        
        # Extract layer ID from prefix for gRPC call
        try:
            # Extract layer index from the prefix
            # Assuming format like "model.layers.12.mlp.experts"
            self.layer_idx = int(prefix.split(".")[-3])
        except (IndexError, ValueError):
            self.layer_idx = 0
            logger.warning(f"Could not extract layer index from prefix '{prefix}', using default 0")
            
        if GrpcExpert.client is None:
            logger.info(f"🚀 GrpcExpert {prefix} creating new gRPC client, with addr: {ek_cfg.ek_addr}")
            GrpcExpert.client = ExpertKitClient(ek_cfg.ek_addr, ek_cfg.ek_client_timeout)

        # self._create_mock_parameters()

    def _create_mock_parameters(self):
        """create mock parameters for the expert layer."""

        with torch.device('meta'):
            # Create parameters on meta device (does not occupy actual memory)
            self.w13_weight = Parameter(torch.empty(
                    self.num_experts, 
                    2 * self.intermediate_size, 
                    self.hidden_size
                ),
                requires_grad=False
            )
            self.w2_weight = Parameter(torch.empty(
                    self.num_experts, 
                    self.hidden_size, 
                    self.intermediate_size,
                ),
                requires_grad=False
            )

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
            
        # Apply softmax first, then take topk
        router_probs = F.softmax(router_logits, dim=-1)
        routing_weights, routing_indices = torch.topk(router_probs, self.top_k, dim=-1)

        # TODO: use vllm original select_experts
        # routing_weights, routing_indices = self.select_experts(
        #     hidden_states=hidden_states,
        #     router_logits=router_logits,
        #     use_grouped_topk=False,
        #     top_k=self.top_k,
        #     renormalize=self.renormalize,
        #     topk_group=self.topk_group,
        #     num_expert_group=self.num_expert_group,
        #     custom_routing_function=self.custom_routing_function,
        #     scoring_func=self.scoring_func,
        #     e_score_correction_bias=self.e_score_correction_bias
        # )
        
        # Renormalize topk weights to ensure they sum to 1
        if self.renormalize:
            routing_weights = routing_weights / (routing_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Use reasonable threshold for similarity detection
        should_optimize = False
        unique_batch_size = batch_size
        inverse_indices = None
        
        if batch_size > 32:  # Only consider optimization for larger batches
            # Use hash or reduced precision to detect duplicates
            hidden_hash = torch.round(hidden_states * 1000).int()  # Reduce precision
            unique_hash, inverse_indices = torch.unique(
                hidden_hash, dim=0, return_inverse=True
            )
            unique_batch_size = unique_hash.shape[0]
            should_optimize = unique_batch_size < 0.7 * batch_size  # Adjust threshold
            
            if self.debug_mode:
                logger.debug(f"🚀 unique_batch_size: {unique_batch_size}, batch_size: {batch_size}, should_optimize: {should_optimize}")
        
        if not should_optimize:
            # Standard path: process all tokens directly
            expert_ids = []
            for seq_idx in range(batch_size):
                token_expert_indices = routing_indices[seq_idx].tolist()
                token_expert_ids = [
                    f"{self.ek_model_name}/l{self.layer_idx}-e{expert_idx}" 
                    for expert_idx in token_expert_indices
                ]
                expert_ids.append(token_expert_ids)
            
            # Call remote expert service
            expert_outputs = self.client.forward_expert(
                expert_ids=expert_ids,
                hidden_state=hidden_states
            )
            
        else:
            unique_hidden = hidden_states[torch.unique(inverse_indices)]
            
            # Build expert IDs for unique tokens
            unique_expert_ids = []
            unique_routing_weights = []
            unique_routing_indices = []
            
            processed_unique = set()
            for i in range(batch_size):
                unique_idx = inverse_indices[i].item()
                if unique_idx not in processed_unique:
                    processed_unique.add(unique_idx)
                    
                    token_expert_indices = routing_indices[i].tolist()
                    token_expert_ids = [
                        f"{self.ek_model_name}/l{self.layer_idx}-e{expert_idx}" 
                        for expert_idx in token_expert_indices
                    ]
                    unique_expert_ids.append(token_expert_ids)
                    unique_routing_weights.append(routing_weights[i])
                    unique_routing_indices.append(routing_indices[i])
            
            # Call remote expert service
            unique_expert_outputs = self.client.forward_expert(
                expert_ids=unique_expert_ids,
                hidden_state=unique_hidden
            )
            
            # Map back to original batch
            expert_outputs = torch.zeros(
                (batch_size, self.top_k, hidden_dim), 
                device=hidden_states.device, 
                dtype=hidden_states.dtype
            )
            
            unique_idx_map = {}
            unique_counter = 0
            for i in range(batch_size):
                orig_unique_idx = inverse_indices[i].item()
                if orig_unique_idx not in unique_idx_map:
                    unique_idx_map[orig_unique_idx] = unique_counter
                    unique_counter += 1
                
                mapped_idx = unique_idx_map[orig_unique_idx]
                expert_outputs[i] = unique_expert_outputs[mapped_idx]
        
        expert_outputs = expert_outputs.to(
            device=hidden_states.device, 
            dtype=hidden_states.dtype
        )
        
        expected_shape = (batch_size, self.top_k, hidden_dim)
        if expert_outputs.shape != expected_shape:
            raise RuntimeError(
                f"Expert outputs shape mismatch: expected {expected_shape}, "
                f"got {expert_outputs.shape}"
            )
        
        # Calculate weighted sum
        expanded_weights = routing_weights.unsqueeze(-1)  # [batch_size, top_k, 1]
        
        if self.debug_mode:
            logger.debug(f"🚀 Weights device: {expanded_weights.device}, Expert outputs device: {expert_outputs.device}")
            logger.debug(f"🚀 Weights dtype: {expanded_weights.dtype}, Expert outputs dtype: {expert_outputs.dtype}")
            logger.debug(f"🚀 Weights shape: {expanded_weights.shape}, Expert outputs shape: {expert_outputs.shape}")
        
        # Compute weighted sum: [batch_size, hidden_dim]
        output = torch.sum(expanded_weights * expert_outputs, dim=1)
        
        return output

    
    @staticmethod
    def select_experts(
        *args,
        **kwargs
    ):
        return FusedMoE.select_experts(
            *args,
            **kwargs
        )
    
    @classmethod
    def make_expert_params_mapping(
        cls,
        *args,
        **kwargs
    ):
        return []
    