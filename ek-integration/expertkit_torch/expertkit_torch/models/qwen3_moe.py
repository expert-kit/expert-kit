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

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers.models.qwen3_moe import modeling_qwen3_moe as qwen3_moe
from transformers.utils.logging import set_verbosity_error

from expertkit_torch.client import RoutedMoEClient

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
) -> None:
    """Install the routed Expert Kit MoE block before loading the Qwen model."""

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
    model_path: str = "./",
    prompts: str | list[str] | None = "What is MoE Model?",
    output_max_length: int = 64,
    enable_ek: bool = True,
    ek_addr: str = "localhost:5002",
    ek_instance_id: int = 1,
) -> dict[str, Any]:
    """Generate responses with either routed Expert Kit or local Qwen experts.

    Args:
        model_path: Local path to the pretrained Qwen checkpoint.
        prompts: One prompt or a list of prompts.
        output_max_length: Maximum number of generated tokens per prompt.
        enable_ek: Whether to route expert computation through Expert Kit.
        ek_addr: Controller Topology endpoint in ``host:port`` form.
        ek_instance_id: Numeric model instance registered with the Controller.

    Returns:
        A mapping containing one decoded result per prompt.
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

    model_inputs = tokenizer(
        batch_messages,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=output_max_length,
        pad_token_id=tokenizer.eos_token_id,
    )

    results = []
    for index, prompt in enumerate(prompts):
        input_length = len(model_inputs.input_ids[index])
        output_ids = generated_ids[index][input_length:].tolist()
        if tokenizer.pad_token_id is not None:
            output_ids = [token_id for token_id in output_ids if token_id != tokenizer.pad_token_id]

        try:
            thinking_end = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            thinking_end = max(len(output_ids) - 1, 0)

        thinking_content = tokenizer.decode(
            output_ids[:thinking_end],
            skip_special_tokens=True,
        ).strip("\n")
        content = tokenizer.decode(
            output_ids[thinking_end:],
            skip_special_tokens=True,
        ).strip("\n")
        results.append(
            {
                "prompt": prompt,
                "thinking_content": thinking_content,
                "content": content,
                "input_tokens": len(model_inputs.input_ids[index]),
                "output_tokens": len(output_ids),
            }
        )

    return {"results": results}
