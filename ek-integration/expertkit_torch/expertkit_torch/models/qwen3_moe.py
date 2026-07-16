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
import json
import os
from typing import Any

import torch
import torch.nn.functional as F
from expertkit_torch.client import RoutedMoEClient
from expertkit_torch.utils.profiler_manager import ProfilerManager
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers.models.qwen3_moe import modeling_qwen3_moe as qwen3_moe
from transformers.utils.logging import set_verbosity_error

set_verbosity_error()

DEFAULT_TIMEOUT_SECONDS = 6
layer_idx = 0

# The default device should be set according to the environment.
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


def intercept_moe(
    enable_ek: bool = True,
    ek_addr: str = "localhost:5002",
    ek_instance_id: int = 1,
):
    class InterceptedMoE(nn.Module):
        client: RoutedMoEClient | None = None

        def __init__(self, config):
            super().__init__()
            global layer_idx
            if enable_ek and InterceptedMoE.client is None:
                InterceptedMoE.client = RoutedMoEClient(
                    ek_addr,
                    instance_id=ek_instance_id,
                    num_layers=config.num_hidden_layers,
                    experts_per_layer=config.num_experts,
                    hidden_dim=config.hidden_size,
                    top_k=config.num_experts_per_tok,
                    timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
                )
            self.layer_id = layer_idx
            layer_idx += 1
            layer_idx = layer_idx % config.num_hidden_layers
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
            assert self.client is not None
            output = self.client.forward_layer(
                layer_id=self.layer_id,
                hidden_states=hidden_states,
                expert_ids=selected_experts,
                routing_weights=routing_weights,
            )
            return output.reshape(batch_size, sequence_length, hidden_dim)

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
            batch_size, sequence_length, hidden_dim = hidden_states.shape
            hidden_states = hidden_states.view(-1, hidden_dim)

            # Process router logits (no need to time separately)
            router_logits = self.gate(hidden_states)
            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
            routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
            if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
                routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
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
                routing_weights = routing_weights.to(hidden_states.dtype)
                expert_mask = torch.nn.functional.one_hot(
                    selected_experts, num_classes=self.num_experts
                ).permute(2, 1, 0)
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

    qwen3_moe.Qwen3MoeSparseMoeBlock = InterceptedMoE


tokenizer: AutoTokenizer | None = None
model: AutoModelForCausalLM | None = None


def evaluate_batch(
    *,
    model_path="./",
    prompts="What is MoE Model?",
    output_max_length=64,
    enable_ek=True,
    ek_addr="localhost:5002",
    ek_instance_id=1,
) -> dict[str, Any]:
    """
    Batch inference with performance profiling.

    Args:
        model_path: Path to the pretrained model
        prompts: List of prompt strings for batch processing
        enable_ek: Whether to enable expert knowledge

    Returns:
        Dictionary containing results and performance metrics
    """
    if prompts is None:
        prompts = ["What is MoE Model?"]

    # Convert str to list
    if isinstance(prompts, str):
        prompts = [prompts]

    # First intercept the MoE module - completely independent of profiling
    intercept_moe(
        enable_ek=enable_ek,
        ek_addr=ek_addr,
        ek_instance_id=ek_instance_id,
    )

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
        ).to(device)

    # Initialize profiler manager with context manager
    with ProfilerManager(batch_size=len(prompts)) as profiler:
        # Wrap model with profiler - completely non-invasive
        profiler.wrap_model(model)

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
            truncation=True,
        ).to(model.device)

        # Generate responses - profiling happens automatically via hooks
        generated_ids = model.generate(
            **model_inputs, max_new_tokens=output_max_length, pad_token_id=tokenizer.eos_token_id
        )

        # Process generated sequences
        results = []
        for i in range(len(prompts)):
            # Extract output tokens
            input_length = len(model_inputs.input_ids[i])
            output_ids = generated_ids[i][input_length:].tolist()

            # Remove padding tokens
            if tokenizer.pad_token_id is not None:
                output_ids = [
                    token_id for token_id in output_ids if token_id != tokenizer.pad_token_id
                ]

            # Extract thinking content
            try:
                # Find </think> token (151668)
                index = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                # Thinking not finished
                index = len(output_ids) - 1

            thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip(
                "\n"
            )

            content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

            results.append(
                {
                    "prompt": prompts[i],
                    "thinking_content": thinking_content,
                    "content": content,
                    "input_tokens": len(model_inputs.input_ids[i]),
                    "output_tokens": len(output_ids),
                }
            )

        # Context manager exit will automatically unwrap the model and print the report
        return {"results": results, "performance": profiler.report()}


def sharegpt(path, max_prompt_len=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File does not exist: {path}")
    if not os.path.isfile(path):
        raise ValueError(f"Path is not a file: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    prompts = []
    for item in data:
        for conversation in item["conversations"]:
            if conversation["from"] == "human":
                if max_prompt_len is not None:
                    prompts.append(conversation["value"][:max_prompt_len])
                else:
                    prompts.append(conversation["value"])
    return prompts


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
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable ExpertKit.",
    )
    parser.add_argument(
        "--ek_instance_id",
        type=int,
        default=1,
        help="The numeric model instance ID registered with the Controller.",
    )
    parser.add_argument(
        "--ek_addr",
        type=str,
        default="localhost:5002",
        help="The address of the ExpertKit controller.",
    )
    parser.add_argument(
        "--detail_profile",
        action="store_true",
        help="Enable detailed profiling of model components (attention vs expert).",
    )
    parser.add_argument(
        "--output_max",
        type=int,
        default=64,
        help="The maximum output length for the model.",
    )
    parser.add_argument(
        "--dataset",
        choices=["none", "sharegpt"],
        default="none",
        help="The dataset to use for evaluation.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to the dataset file.",
    )
    parser.add_argument(
        "--print_response",
        action="store_true",
        help="Print the response content.",
    )
    parser.add_argument(
        "--max_prompt_len",
        type=int,
        default=None,
        help="Maximum length of each prompt (applicable for ShareGPT dataset).",
    )
    parser.add_argument(
        "--prompt_num",
        type=int,
        default=512,
        help="The number of prompts to use for evaluation.",
    )
    args = parser.parse_args()

    if args.dataset == "none":
        # Use default prompts if no dataset is specified
        test_prompts = [
            "What is MoE Model?",
            "Explain the benefits of mixture of experts.",
            "How does MoE improve model efficiency?",
            "Compare MoE with dense models.",
        ] * args.prompt_num
        test_prompts = test_prompts[: args.prompt_num]
    elif args.dataset == "sharegpt":
        # Validate that dataset_path is provided
        if args.dataset_path is None:
            raise ValueError("You must provide --dataset_path when using the 'sharegpt' dataset.")
        # Load prompts from ShareGPT dataset
        test_prompts = sharegpt(args.dataset_path, max_prompt_len=args.max_prompt_len)
        if len(test_prompts) < args.prompt_num:
            test_prompts *= (args.prompt_num // len(test_prompts)) + 1
        test_prompts = test_prompts[: args.prompt_num]
    else:
        raise ValueError("Invalid dataset specified.")

    test_batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    aggregated_results = []
    for batch_size in test_batch_sizes:
        for prompts in range(0, len(test_prompts), batch_size):
            if prompts / batch_size >= 6:
                break
            batch_result = evaluate_batch(
                model_path=args.model_path,
                prompts=test_prompts[prompts : prompts + batch_size],
                enable_ek=args.enable_ek,
                ek_addr=args.ek_addr,
                ek_instance_id=args.ek_instance_id,
                output_max_length=args.output_max,
            )
            aggregated_results.extend(batch_result["results"])

    if args.print_response:
        for result in aggregated_results[:5]:
            print()
            print(f"Prompt: {result['prompt']}")
            print(f"Thinking Content: {result['thinking_content']}")
            print(f"Response: {result['content']}")
            print(
                f"Input Tokens: {result['input_tokens']}, Output Tokens: {result['output_tokens']}"
            )
            print("-" * 40)


if __name__ == "__main__":
    main()
