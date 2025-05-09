from transformers.models.deepseek_v3 import modeling_deepseek_v3 as ds_v3
from transformers.models.deepseek_v3 import configuration_deepseek_v3 as ds_v3_config
from transformers import AutoTokenizer
from torch import nn
import torch
import argparse
from expertkit_torch.grpc_client import ExpertKitClient

layer_idx = 0


def intercept_moe(with_ek: bool):
    class InterceptedDeepseekV3MoE(nn.Module):
        """
        A mixed expert module containing shared experts.
        """

        client: ExpertKitClient = None

        def __init__(self, config):
            super().__init__()
            global layer_idx
            self.layer_id = layer_idx
            layer_idx += 1
            self.config = config
            if not with_ek:
                self.experts = nn.ModuleList(
                    [
                        ds_v3.DeepseekV3MLP(
                            config, intermediate_size=config.moe_intermediate_size
                        )
                        for _ in range(config.n_routed_experts)
                    ]
                )
            self.gate = ds_v3.DeepseekV3TopkRouter(config)
            self.shared_experts = ds_v3.DeepseekV3MLP(
                config=config,
                intermediate_size=config.moe_intermediate_size
                * config.n_shared_experts,
            )

        def ek_moe(
            self,
            hidden_states: torch.Tensor,
            topk_indices: torch.Tensor,
            topk_weights: torch.Tensor,
        ):
            expert_ids = []
            total_seq_len, _ = hidden_states.shape
            for seq_idx in range(total_seq_len):
                eids = topk_indices[seq_idx].tolist()
                ids = [
                    f"model-layer{self.layer_id}-expert{expert_idx}.safetensors"
                    for expert_idx in eids
                ]
                expert_ids.append(ids)

            for seq_idx in range(total_seq_len):
                eids = topk_indices[seq_idx].tolist()

            outputs = self.client.forward_expert(
                expert_ids=expert_ids, hidden_state=hidden_states
            )
            outputs = outputs.to(device=hidden_states.device, dtype=hidden_states.dtype)
            expanded_weights = topk_weights.unsqueeze(-1)
            output = torch.sum(expanded_weights * outputs, dim=1)

            final_hidden_states = output.type(hidden_states.dtype).reshape(
                hidden_states.shape
            )
            return final_hidden_states

        def moe(
            self,
            hidden_states: torch.Tensor,
            topk_indices: torch.Tensor,
            topk_weights: torch.Tensor,
        ):
            r"""
            CALL FOR CONTRIBUTION! I don't have time to optimise this right now, but expert weights need to be fused
            to not have to do a loop here (deepseek has 256 experts soooo yeah).
            """
            final_hidden_states = torch.zeros_like(
                hidden_states, dtype=topk_weights.dtype
            )
            expert_mask = torch.nn.functional.one_hot(
                topk_indices, num_classes=len(self.experts)
            )
            expert_mask = expert_mask.permute(2, 0, 1)

            for expert_idx in range(len(self.experts)):
                expert = self.experts[expert_idx]
                mask = expert_mask[expert_idx]
                token_indices, weight_indices = torch.where(mask)

                if token_indices.numel() > 0:
                    expert_weights = topk_weights[token_indices, weight_indices]
                    expert_input = hidden_states[token_indices]
                    expert_output = expert(expert_input)
                    weighted_output = expert_output * expert_weights.unsqueeze(-1)
                    final_hidden_states.index_add_(0, token_indices, weighted_output)

            # in original deepseek, the output of the experts are gathered once we leave this module
            # thus the moe module is itelsf an IsolatedParallel module
            # and all expert are "local" meaning we shard but we don't gather
            return final_hidden_states.type(hidden_states.dtype)

        def forward(self, hidden_states):
            residuals = hidden_states
            orig_shape = hidden_states.shape
            topk_indices, topk_weights = self.gate(hidden_states)
            hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
            if with_ek:
                hidden_states = self.ek_moe(
                    hidden_states, topk_indices, topk_weights
                ).view(*orig_shape)
            else:
                hidden_states = self.moe(
                    hidden_states, topk_indices, topk_weights
                ).view(*orig_shape)
            hidden_states = hidden_states + self.shared_experts(residuals)
            return hidden_states

    delattr(ds_v3, "DeepseekV3MoE")
    setattr(ds_v3, "DeepseekV3MoE", InterceptedDeepseekV3MoE)


def evaluate(model_path=str, enable_ek=True):
    tokenizer = AutoTokenizer.from_pretrained("deepseek-v3")
    chat = [
        {"role": "user", "content": "Hello, how are you?"},
    ]
    model_config = ds_v3_config.DeepseekV3Config.from_pretrained(model_path)
    model = ds_v3.DeepseekV3ForCausalLM.from_pretrained(model_path, config=model_config)
    inputs = tokenizer.apply_chat_template(
        chat, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    import time

    start = time.time()
    outputs = model.generate(inputs, max_new_tokens=50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model directory.",
    )
    parser.add_argument(
        "--enable_ek",
        type=bool,
        action=argparse.BooleanOptionalAction,
        help="Enable ExpertKit.",
    )
    args = parser.parse_args()
    evaluate(model_path=args.model_path, enable_ek=args.enable_ek)
