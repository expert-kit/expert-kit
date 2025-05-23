from transformers.models.deepseek_v3 import modeling_deepseek_v3 as ds_v3
from transformers.models.deepseek_v3 import configuration_deepseek_v3 as ds_v3_config
from transformers import AutoTokenizer
from torch import nn
import torch
import argparse
from expertkit_torch.grpc_client import ExpertKitClient
from transformers import modeling_utils as mu

layer_idx = 3
device = "mps"


def intercept_missing():
    """
    Intercept the missing function in the DeepseekV3 model.
    """

    def missing_function(self, *args, **kwargs):
        return ([], [])

    # Intercept the missing function in the DeepseekV3 model
    delattr(mu, "_find_mismatched_keys")
    delattr(mu, "_find_missing_and_unexpected_keys")
    setattr(mu, "_find_mismatched_keys", missing_function)
    setattr(mu, "_find_missing_and_unexpected_keys", missing_function)


def intercept_moe(ek_addr: str | None, model_name: str):
    with_ek = ek_addr is not None

    class InterceptedDeepseekV3MoE(nn.Module):
        """
        A mixed expert module containing shared experts.
        """

        client: ExpertKitClient | None = None

        def __init__(self, config):
            super().__init__()
            global layer_idx
            self.layer_id = layer_idx
            layer_idx += 1
            self.config = config
            if with_ek and InterceptedDeepseekV3MoE.client is None:
                InterceptedDeepseekV3MoE.client = ExpertKitClient(ek_addr, 240)
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
                    f"{model_name}/l{self.layer_id}-e{expert_idx}"
                    for expert_idx in eids
                ]
                expert_ids.append(ids)

            for seq_idx in range(total_seq_len):
                eids = topk_indices[seq_idx].tolist()

            if self.client is None:
                raise SystemError("client is None, please check the address")

            import time

            print(f"send at {self.layer_id} {hidden_states.shape=}")
            start = time.time()
            hidden_states = hidden_states.to(torch.bfloat16)

            outputs = self.client.forward_expert(
                expert_ids=expert_ids, hidden_state=hidden_states
            )
            end = time.time()
            outputs = outputs.to(device=hidden_states.device, dtype=hidden_states.dtype)
            expanded_weights = topk_weights.unsqueeze(-1)
            print(
                f"send at {self.layer_id} elapsed= {end - start},{outputs.shape}  {expanded_weights.shape=}"
            )
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


@torch.no_grad()
@torch.inference_mode()
def evaluate(model_path=str, ek_addr: str | None = None, model_name: str = "", batch=1):

    intercept_missing()
    intercept_moe(ek_addr=ek_addr, model_name=model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    chat = [
        {"role": "user", "content": "Hello, how are you?"},
    ]
    test_prompts = [
        "What is MoE Model?",
    ] * 512

    model_config = ds_v3_config.DeepseekV3Config.from_pretrained(model_path)

    batch_messages = []
    prompts = test_prompts[:batch]
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        batch_messages.append(text)

    model = ds_v3.DeepseekV3ForCausalLM.from_pretrained(
        model_path,
        config=model_config,
        local_files_only=True,
        device_map=device,
    )
    inputs = tokenizer.apply_chat_template(
        chat, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    import time

    start = time.time()

    model_inputs = tokenizer(
        batch_messages, return_tensors="pt", padding=True, truncation=True
    ).to(model.device)
    print(f"123, {model_inputs}")
    generated_ids = model.generate(
        **model_inputs, max_new_tokens=1, pad_token_id=tokenizer.eos_token_id
    )
    now = time.time()
    output_ids = generated_ids[0].tolist()

    generation_time = now - start

    total_input_tokens = model_inputs.input_ids.numel()
    total_output_tokens = generated_ids.numel() - total_input_tokens
    total_tps = (total_input_tokens + total_output_tokens) / generation_time
    output_tps = total_output_tokens / generation_time

    elasped = now - start
    print(f"\n--- Test Size {len(prompts)} ---")
    print(
        f"Batch inference elapsed time: {generation_time:.2f}s for {len(prompts)} prompts"
    )
    print(
        f"Total tokens processed: {total_input_tokens + total_output_tokens} ({total_input_tokens} input + {total_output_tokens} output)"
    )
    print(f"Total TPS (input+output): {total_tps:.2f} tokens/second")
    print(f"Output TPS (generation only): {output_tps:.2f} tokens/second")
    print(f"Average TPS per prompt: {output_tps/len(prompts):.2f} tokens/second")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model directory.",
    )
    parser.add_argument(
        "--ek_addr",
        type=str,
        help="ExpertKit address. like http://localhost:5002",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="The name of the model used in ExpertKit. like qwen3",
    )
    parser.add_argument(
        "--seq",
        type=int,
        required=False,
        help="Batch size for evaluation.",
        default=1,
    )
    args = parser.parse_args()
    evaluate(
        model_path=args.model_path,
        ek_addr=args.ek_addr,
        model_name=args.model_name,
        batch=args.seq,
    )
