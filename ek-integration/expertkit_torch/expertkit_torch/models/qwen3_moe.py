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

"""Qwen3-MoE routed block for the pinned Transformers implementation."""

from __future__ import annotations

import torch
from torch import nn
from transformers.models.qwen3_moe import modeling_qwen3_moe

from expertkit_torch.client import RoutedMoEClient
from expertkit_torch.models._common import RoutedLayerIds


def create_routed_moe_class(
    client: RoutedMoEClient,
    layer_ids: RoutedLayerIds,
) -> type[nn.Module]:
    """Create a Qwen3-MoE class bound to one model and client."""

    class RoutedQwen3MoeSparseMoeBlock(nn.Module):
        """Preserve Qwen routing and return the remote aggregated result."""

        def __init__(self, config) -> None:
            super().__init__()
            self.layer_id = layer_ids.take()
            self.gate = modeling_qwen3_moe.Qwen3MoeTopKRouter(config)

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            """Route final top-k assignments through Expert Kit."""

            batch_size, sequence_length, hidden_dim = hidden_states.shape
            flattened = hidden_states.reshape(-1, hidden_dim)
            _, routing_weights, expert_ids = self.gate(flattened)
            routed = client.forward_layer(
                layer_id=self.layer_id,
                hidden_states=flattened,
                expert_ids=expert_ids,
                routing_weights=routing_weights,
            )
            return routed.reshape(batch_size, sequence_length, hidden_dim)

    return RoutedQwen3MoeSparseMoeBlock
