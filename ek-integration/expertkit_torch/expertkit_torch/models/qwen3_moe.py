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
import torch
import torch.nn.functional as F

from typing import Optional
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from transformers.utils.logging import set_verbosity_error
from transformers.models.qwen3_moe import modeling_qwen3_moe as qwen3_moe
from torch import nn
from expertkit_torch.grpc_client import ExpertKitClient

set_verbosity_error()

layer_idx = 0


def intercept_moe(with_ek: bool):
    class Intercepted(nn.Module):
        client: ExpertKitClient = None

        def __init__(self, config):
            super().__init__()
            global layer_idx
            if with_ek and Intercepted.client is None:
                Intercepted.client = ExpertKitClient(config.ek_addr, 10)
            self.layer_id = layer_idx
            layer_idx += 1
            layer_idx = layer_idx % config.num_hidden_layers
            self.num_experts = config.num_experts
            self.top_k = config.num_experts_per_tok
            self.norm_topk_prob = config.norm_topk_prob

            self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
            if not with_ek:
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
                    f"qwen3/l{self.layer_id}-e{expert_idx}"
                    for expert_idx in eids
                ]
                expert_ids.append(ids)

            # TODO
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

            if with_ek:
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

tokenizer:Optional[AutoTokenizer] = None
model:Optional[AutoModelForCausalLM] = None

def evaluate_batch(*, model_path="./", prompts=None, enable_ek=True):
    """
    Batch inference version of the evaluate function.
    
    Args:
        model_path: Path to the pretrained model
        prompts: List of prompt strings for batch processing
        enable_ek: Whether to enable expert knowledge
    
    Returns:
        List of dictionaries containing thinking_content and content for each prompt
    """
    if prompts is None:
        prompts = ["What is MoE Model?"]
    
    # Ensure prompts is a list
    if isinstance(prompts, str):
        prompts = [prompts]
    
    intercept_moe(enable_ek)

    # Load the tokenizer and the model only once
    global tokenizer, model
    if tokenizer is None:    
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_path,
        )
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_path,
            torch_dtype="auto",
        )
    
    # Prepare batch messages
    batch_messages = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        batch_messages.append(text)
    
    # Tokenize batch inputs with padding
    model_inputs = tokenizer(
        batch_messages, 
        return_tensors="pt", 
        padding=True, 
        truncation=True
    ).to(model.device)
    
    # Conduct batch text completion
    now = time.time()
    generated_ids = model.generate(
        **model_inputs, 
        max_new_tokens=50,
        pad_token_id=tokenizer.eos_token_id  # Handle padding properly
    )
    end = time.time()
    
    # Calculate TPS metrics
    generation_time = end - now
    total_input_tokens = model_inputs.input_ids.numel()
    total_output_tokens = generated_ids.numel() - total_input_tokens
    
    # Calculate different TPS metrics
    total_tps = (total_input_tokens + total_output_tokens) / generation_time
    output_tps = total_output_tokens / generation_time
    
    print(f"\n--- Test Size {len(prompts)} ---")
    print(f"Batch inference elapsed time: {generation_time:.2f}s for {len(prompts)} prompts")
    print(f"Average time per prompt: {generation_time/len(prompts):.3f}s")
    print(f"Total tokens processed: {total_input_tokens + total_output_tokens} ({total_input_tokens} input + {total_output_tokens} output)")
    print(f"Total TPS (input+output): {total_tps:.2f} tokens/second")
    print(f"Output TPS (generation only): {output_tps:.2f} tokens/second")
    print(f"Average TPS per prompt: {output_tps/len(prompts):.2f} tokens/second")
    print()
    
    # Process each generated sequence in the batch
    results = []
    for i in range(len(prompts)):
        # Extract output tokens for this sequence
        input_length = len(model_inputs.input_ids[i])
        output_ids = generated_ids[i][input_length:].tolist()
        
        # Remove padding tokens
        if tokenizer.pad_token_id is not None:
            output_ids = [token_id for token_id in output_ids if token_id != tokenizer.pad_token_id]
        
        # Parse thinking content
        try:
            # Find the last occurrence of 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            print(f"Value error for prompt {i}")
            index = 0
        
        thinking_content = tokenizer.decode(
            output_ids[:index], skip_special_tokens=True
        ).strip("\n")
        content = tokenizer.decode(
            output_ids[index:], skip_special_tokens=True
        ).strip("\n")
        
        results.append({
            "prompt": prompts[i],
            "thinking_content": thinking_content,
            "content": content,
            "input_tokens": len(model_inputs.input_ids[i]),
            "output_tokens": len(output_ids),
        })
    
    # Print results for debugging
    # if len(results) > 0:
    #     print(f"Output of first result")
    #     print(f"Prompt: {results[0]['prompt']}")
    #     print(f"Input tokens: {results[0]['input_tokens']}, Output tokens: {result['output_tokens']}")
    #     print(f"Thinking content: {results[0]['thinking_content']}")
    #     print(f"Content: {results[0]['content']}")
    #     print()
    
    # Return results with performance metrics
    return {
        "results": results,
        "performance": {
            "total_time": generation_time,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tps": total_tps,
            "output_tps": output_tps,
            "average_time_per_prompt": generation_time / len(prompts),
            "effective_tps_per_prompt": output_tps / len(prompts),
        }
    }


# Example usage:
def main():
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

    test_prompts = [
        "What is MoE Model?",
        "Explain the benefits of mixture of experts.",
        "How does MoE improve model efficiency?",
        "Compare MoE with dense models.",
    ] * 512
    
    test_batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    for batch_size in test_batch_sizes:
        batch_result = evaluate_batch(prompts=test_prompts[:batch_size], model_path=args.model_path, enable_ek=args.enable_ek)

if __name__ == "__main__":
    main()