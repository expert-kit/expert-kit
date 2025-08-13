import time
from expertkit_torch.grpc_client import ExpertKitClient
from torch import nn
import torch.nn.functional as F
from transformers.models.mixtral.modeling_mixtral import MixtralBlockSparseTop2MLP
from transformers.models.mixtral import modeling_mixtral as mixtral
import torch


def intercept_moe(
    enable_ek=True,
    ek_addr: str = "localhost:5002",
    ek_model_name: str = "mixtral_8x7b",
):

    class InterceptedMOE(nn.Module):

        client: ExpertKitClient = None

        def __init__(self, config):
            super().__init__()
            if enable_ek and InterceptedMOE.client is None:
                InterceptedMOE.client = ExpertKitClient(ek_addr, DEFAULT_TIMEOUT_INTVAL)
            self.hidden_dim = config.hidden_size
            self.ffn_dim = config.intermediate_size
            self.num_experts = config.num_local_experts
            self.top_k = config.num_experts_per_tok

            # gating
            self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)

            if not enable_ek:
                self.experts = nn.ModuleList(
                    [MixtralBlockSparseTop2MLP(config) for _ in range(self.num_experts)]
                )

            # Jitter parameters
            self.jitter_noise = config.router_jitter_noise

        def ek_forward(
            self,
            *,
            hidden_states: torch.Tensor,
            routing_weights: torch.Tensor,
            selected_experts: torch.Tensor,
            batch_size: int,
            sequence_length: int,
            hidden_dim: int,
        ):
            start_time = time.time()

            expert_ids = []
            total_seq_len, _ = hidden_states.shape
            for seq_idx in range(total_seq_len):
                eids = selected_experts[seq_idx].tolist()
                ids = [
                    f"{ek_model_name}/l{self.layer_id}-e{expert_idx}"
                    for expert_idx in eids
                ]
                expert_ids.append(ids)

            outputs = self.client.forward_expert(
                expert_ids=expert_ids, hidden_state=hidden_states
            )
            outputs = outputs.to(device=hidden_states.device, dtype=hidden_states.dtype)
            expanded_weights = routing_weights.unsqueeze(-1)
            output = torch.sum(expanded_weights * outputs, dim=1)

            final_hidden_states = output.reshape(
                batch_size, sequence_length, hidden_dim
            )

            # Record expert computation time if profiler is available
            end_time = time.time()

            return final_hidden_states

        def normal_forward(
            self,
            hidden_states: torch.Tensor,
            expert_mask: torch.Tensor,
            hidden_dim: int,
            routing_weights: torch.Tensor,
            final_hidden_states: torch.Tensor,
        ):
            for expert_idx in range(self.num_experts):
                expert_layer = self.experts[expert_idx]
                idx, top_x = torch.where(expert_mask[expert_idx])

                # Index the correct hidden states and compute the expert hidden state for
                # the current expert. We need to make sure to multiply the output hidden
                # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
                current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
                current_hidden_states = (
                    expert_layer(current_state) * routing_weights[top_x, idx, None]
                )

                # However `index_add_` only support torch tensors for indexing so we'll use
                # the `top_x` tensor here.
                final_hidden_states.index_add_(
                    0, top_x, current_hidden_states.to(hidden_states.dtype)
                )
            pass

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            """ """
            batch_size, sequence_length, hidden_dim = hidden_states.shape
            if self.training and self.jitter_noise > 0:
                hidden_states *= torch.empty_like(hidden_states).uniform_(
                    1.0 - self.jitter_noise, 1.0 + self.jitter_noise
                )
            hidden_states = hidden_states.view(-1, hidden_dim)
            # router_logits: (batch * sequence_length, n_experts)
            router_logits = self.gate(hidden_states)

            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
            routing_weights, selected_experts = torch.topk(
                routing_weights, self.top_k, dim=-1
            )
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
            # we cast back to the input dtype
            routing_weights = routing_weights.to(hidden_states.dtype)

            final_hidden_states = torch.zeros(
                (batch_size * sequence_length, hidden_dim),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            # One hot encode the selected experts to create an expert mask
            # this will be used to easily index which expert is going to be sollicitated
            expert_mask = torch.nn.functional.one_hot(
                selected_experts, num_classes=self.num_experts
            ).permute(2, 1, 0)

            if not enable_ek:
                final = self.normal_forward(
                    hidden_states=hidden_states,
                    expert_mask=expert_mask,
                    hidden_dim=hidden_dim,
                    routing_weights=routing_weights,
                    final_hidden_states=final_hidden_states,
                )

                final = final_hidden_states.reshape(
                    batch_size, sequence_length, hidden_dim
                )
            else:
                final = self.ek_forward(
                    hidden_states=hidden_states,
                    routing_weights=routing_weights,
                    selected_experts=selected_experts,
                    batch_size=batch_size,
                    sequence_length=sequence_length,
                    hidden_dim=hidden_dim,
                )

            return final, router_logits

    delattr(mixtral, "MixtralSparseMoeBlock")
    setattr(mixtral, "MixtralSparseMoeBlock", InterceptedMOE)
