import argparse
import logging
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from transformers.utils.logging import set_verbosity_error
from transformers.models.qwen3_moe import modeling_qwen3_moe as qwen3_moe
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from expertkit_torch.grpc_client import ExpertKitClient

set_verbosity_error()

layer_idx = 0


def intercept_moe():
    class Intercepted(nn.Module):
        client: ExpertKitClient = None

        def __init__(self, config):
            super().__init__()
            global layer_idx
            self.layer_id = layer_idx
            layer_idx += 1
            self.num_experts = config.num_experts
            self.top_k = config.num_experts_per_tok
            self.norm_topk_prob = config.norm_topk_prob

            self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
            # self.experts = ([idx for idx in range(self.num_experts)])

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            """ """
            batch_size, sequence_length, hidden_dim = hidden_states.shape
            hidden_states = hidden_states.view(-1, hidden_dim)
            # router_logits: (batch * sequence_length, n_experts)
            router_logits = self.gate(hidden_states)

            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
            routing_weights, selected_experts = torch.topk(
                routing_weights, self.top_k, dim=-1
            )
            if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
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

            # Loop over all available experts in the model and perform the computation on each expert
            # dim = (seq,selected idx)
            expert_ids = []
            total_seq_len, _ = hidden_states.shape
            for seq_idx in range(total_seq_len):
                eids = selected_experts[seq_idx].tolist()
                ids = [
                    f"model-layer{self.layer_id}-expert{expert_idx}.safetensors"
                    for expert_idx in eids
                ]
                expert_ids.append(ids)
            # TODO
            # outputs = self.client.forward_expert(
            #     expert_ids=expert_ids, hidden_state=hidden_states
            # )
            outputs = torch.randn(total_seq_len, self.top_k, hidden_dim)
            outputs = outputs.to(device=hidden_states.device, dtype=hidden_states.dtype)
            # rw(dim=seq*n_active)
            # outputs(dim=seq*n_active*hidden)
            expanded_weights = routing_weights.unsqueeze(-1)
            output = torch.sum(expanded_weights * outputs, dim=1)

            final_hidden_states = output.reshape(
                batch_size, sequence_length, hidden_dim
            )
            return final_hidden_states, router_logits

    delattr(qwen3_moe, "Qwen3MoeSparseMoeBlock")
    setattr(qwen3_moe, "Qwen3MoeSparseMoeBlock", Intercepted)


def evaluate(*, model_path="./", prompt="What is MoE Model?", ek_enable=True):
    if ek_enable:
        intercept_moe()
    # load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_path,
    )
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_path,
        torch_dtype="auto",
    )

    # prepare the model input
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,  # Switches between thinking and non-thinking modes. Default is True.
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # conduct text completion
    generated_ids = model.generate(**model_inputs, max_new_tokens=20)
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()

    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        print("value error")
        index = 0

    thinking_content = tokenizer.decode(
        output_ids[:index], skip_special_tokens=True
    ).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    print("thinking content:", thinking_content)
    print("content:", content)
    return {
        "thinking_content": thinking_content,
        "content": content,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model directory.",
    )
    args = parser.parse_args()
    evaluate(model_path=args.model_path)
