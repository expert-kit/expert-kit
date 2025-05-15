# coding=utf-8
# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# Modifications Copyright (c) 2025 expertkit-torch.
#
# This file is based on code from the Qwen3 project (originally licensed under Apache 2.0)
# and has been modified.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import time
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from transformers.utils.logging import set_verbosity_error
from transformers.models.qwen3_moe import modeling_qwen3_moe as qwen3_moe
import torch
from torch import nn
import torch.nn.functional as F
from expertkit_torch.grpc_client import ExpertKitClient

set_verbosity_error()

# default timeout interval for ek client, in seconds
DEFAULT_TIMEOUT_INTVAL = 100
layer_idx = 0


def intercept_moe(
    enable_ek: bool = True,
    ek_addr: str = "localhost:50051",
    ek_model_name: str = "qwen3",    
):
    class Intercepted(nn.Module):
        client: ExpertKitClient = None

        def __init__(self, config):
            super().__init__()
            global layer_idx
            if enable_ek and Intercepted.client is None:
                Intercepted.client = ExpertKitClient(ek_addr, DEFAULT_TIMEOUT_INTVAL)
            self.layer_id = layer_idx
            layer_idx += 1
            self.num_experts = config.num_experts
            self.top_k = config.num_experts_per_tok
            self.norm_topk_prob = config.norm_topk_prob

            self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
            if not enable_ek:
                self.experts = nn.ModuleList(
                    [
                        qwen3_moe.Qwen3MoeMLP(
                            config, intermediate_size=config.moe_intermediate_size
                        )
                        for _ in range(self.num_experts)
                    ]
                )

            # self.experts = ([idx for idx in range(self.num_experts)])

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
            return final_hidden_states

        def normal_forward(
            self,
            *,
            hidden_states: torch.Tensor,
            routing_weights: torch.Tensor,
            selected_experts: torch.Tensor,
            expert_mask: torch.Tensor,
            batch_size: int,
            sequence_length: int,
            hidden_dim: int,
        ):
            final_hidden_states = torch.zeros(
                (batch_size * sequence_length, hidden_dim),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            for expert_idx in range(self.num_experts):
                expert_layer = self.experts[expert_idx]
                idx, top_x = torch.where(expert_mask[expert_idx])
                current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
                current_hidden_states = (
                    expert_layer(current_state) * routing_weights[top_x, idx, None]
                )
                final_hidden_states.index_add_(
                    0, top_x, current_hidden_states.to(hidden_states.dtype)
                )
            final_hidden_states = final_hidden_states.reshape(
                batch_size, sequence_length, hidden_dim
            )
            return final_hidden_states

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            """ """
            batch_size, sequence_length, hidden_dim = hidden_states.shape
            hidden_states = hidden_states.view(-1, hidden_dim)
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

            if enable_ek:
                final = self.ek_forward(
                    hidden_states=hidden_states,
                    routing_weights=routing_weights,
                    selected_experts=selected_experts,
                    batch_size=batch_size,
                    sequence_length=sequence_length,
                    hidden_dim=hidden_dim,
                )
            else:
                final = self.normal_forward(
                    hidden_states=hidden_states,
                    routing_weights=routing_weights,
                    selected_experts=selected_experts,
                    expert_mask=expert_mask,
                    batch_size=batch_size,
                    sequence_length=sequence_length,
                    hidden_dim=hidden_dim,
                )

            return final, router_logits

    delattr(qwen3_moe, "Qwen3MoeSparseMoeBlock")
    setattr(qwen3_moe, "Qwen3MoeSparseMoeBlock", Intercepted)


def evaluate(
    *, 
    model_path="./", 
    prompt="What is MoE Model?", 

    enable_ek=True,
    ek_addr="localhost:50051",
    ek_model_name="qwen3"
):
    intercept_moe(
        enable_ek=enable_ek,
        ek_addr=ek_addr,
        ek_model_name=ek_model_name,
    )
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
    now = time.time()
    generated_ids = model.generate(**model_inputs, max_new_tokens=50)
    end = time.time()
    print("elapsed time:", end - now)
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
    parser.add_argument(
        "--enable_ek",
        type=bool,
        action=argparse.BooleanOptionalAction,
        help="Enable ExpertKit.",
    )
    parser.add_argument(
        "--ek_model_name",
        type=str,
        default="qwen3",
        help="The name of the model used in ExpertKit.",
    )
    parser.add_argument(
        "--ek_addr",
        type=str,
        default="localhost:50051",
        help="The address of the ExpertKit server.",
    )
    args = parser.parse_args()
    evaluate(
        model_path=args.model_path, 
        ek_model_name=args.ek_model_name,

        enable_ek=args.enable_ek,
        ek_addr=args.ek_addr,
        ek_model_name=args.ek_model_name,
    )
