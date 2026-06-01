import argparse
import json
import os
import time
import torch
import torch.nn.functional as F

from typing import Optional, Dict, Any, List
from transformers import AutoTokenizer
from transformers.utils.logging import set_verbosity_error
from transformers.models.deepseek_v2 import modeling_deepseek_v2 as ds_v2
from transformers.models.deepseek_v2 import configuration_deepseek_v2 as ds_v2_config
from transformers import modeling_utils as mu
from torch import nn
from expertkit_torch.grpc_client_new import ExpertKitClient

from expertkit_torch.utils.profiler_manager import ProfilerManager
from line_profiler import profile

set_verbosity_error()

# default timeout interval for ek client, in seconds
DEFAULT_TIMEOUT_INTVAL = 100
layer_idx = None

# The default device should be set according to the environment.
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


def intercept_missing():
    """
    Intercept the missing function in the DeepseekV2 model.
    """

    def missing_function(self, *args, **kwargs):
        return ([], [])

    delattr(mu, "_find_mismatched_keys")
    delattr(mu, "_find_missing_and_unexpected_keys")
    setattr(mu, "_find_mismatched_keys", missing_function)
    setattr(mu, "_find_missing_and_unexpected_keys", missing_function)


def intercept_moe(
    enable_ek: bool = True,
    ek_addr: str = "localhost:5002",
    ek_model_name: str = "deepseek_v2",
    enable_direct_path: bool = True,
    first_k_dense_replace: int = None,
):
    global layer_idx
    if first_k_dense_replace is not None:
        layer_idx = first_k_dense_replace

    class InterceptedDeepseekV2MoE(nn.Module):
        """
        A mixed expert module containing shared experts.
        """

        client: ExpertKitClient = None

        def __init__(self, config):
            super().__init__()
            global layer_idx
            # Auto-detect first_k_dense_replace from config on first instantiation
            if layer_idx is None:
                layer_idx = getattr(config, "first_k_dense_replace", 0)
            self.layer_id = layer_idx
            layer_idx += 1
            self.config = config
            self.num_experts = config.n_routed_experts
            self.top_k = config.num_experts_per_tok
            self.norm_topk_prob = config.norm_topk_prob
            self.routed_scaling_factor = getattr(
                config, "routed_scaling_factor", 1.0)
            self.topk_method = getattr(config, "topk_method", "greedy")
            self.n_group = getattr(config, "n_group", 1)
            self.topk_group = getattr(config, "topk_group", 1)

            if enable_ek and InterceptedDeepseekV2MoE.client is None:
                InterceptedDeepseekV2MoE.client = ExpertKitClient(
                    controller_addr=ek_addr,
                    timeout_sec=DEFAULT_TIMEOUT_INTVAL,
                )
                print(
                    f"[ExpertKit] Client initialized: controller={ek_addr}, direct_path={enable_direct_path}")

            # gating
            self.gate = nn.Linear(
                config.hidden_size, self.num_experts, bias=False)

            if not enable_ek:
                self.experts = nn.ModuleList(
                    [
                        ds_v2.DeepseekV2MLP(
                            config, intermediate_size=config.moe_intermediate_size
                        )
                        for _ in range(self.num_experts)
                    ]
                )

            self.shared_experts = ds_v2.DeepseekV2MLP(
                config=config,
                intermediate_size=config.moe_intermediate_size
                * config.n_shared_experts,
            )

        def route_tokens(self, hidden_states: torch.Tensor):
            """
            V2 gating: softmax scoring with greedy or group_limited_greedy routing.
            """
            router_logits = self.gate(
                hidden_states.view(-1, hidden_states.shape[-1]))
            scores = F.softmax(router_logits, dim=1, dtype=torch.float)

            if self.topk_method == "greedy":
                topk_weights, topk_indices = torch.topk(
                    scores, k=self.top_k, dim=-1, sorted=False
                )
            elif self.topk_method == "group_limited_greedy":
                # Group experts and select top groups first
                group_scores = (
                    scores.view(-1, self.n_group,
                                self.num_experts // self.n_group)
                    .max(dim=-1)[0]
                )
                group_idx = torch.topk(
                    group_scores, k=self.topk_group, dim=-1, sorted=False
                )[1]
                group_mask = torch.zeros_like(group_scores)
                group_mask.scatter_(1, group_idx, 1)
                score_mask = (
                    group_mask.unsqueeze(-1)
                    .expand(-1, self.n_group, self.num_experts // self.n_group)
                    .reshape(-1, self.num_experts)
                )
                scores_for_choice = scores.masked_fill(~score_mask.bool(), 0.0)
                topk_weights, topk_indices = torch.topk(
                    scores_for_choice, k=self.top_k, dim=-1, sorted=False
                )
            else:
                raise ValueError(f"Unknown topk_method: {self.topk_method}")

            if self.norm_topk_prob:
                denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
                topk_weights = topk_weights / denominator
            else:
                topk_weights = topk_weights * self.routed_scaling_factor

            topk_weights = topk_weights.to(hidden_states.dtype)
            return topk_indices, topk_weights

        def ek_moe(
            self,
            hidden_states: torch.Tensor,
            topk_indices: torch.Tensor,
            topk_weights: torch.Tensor,
        ):
            start_time = time.time()

            expert_ids = []
            total_seq_len, _ = hidden_states.shape
            for seq_idx in range(total_seq_len):
                eids = topk_indices[seq_idx].tolist()
                ids = [
                    f"{ek_model_name}/l{self.layer_id}-e{expert_idx}"
                    for expert_idx in eids
                ]
                expert_ids.append(ids)

            if self.client is None:
                raise SystemError("client is None, please check the address")

            outputs = self.client.forward_expert(
                expert_ids=expert_ids, hidden_state=hidden_states
            )
            outputs = outputs.to(device=hidden_states.device,
                                 dtype=hidden_states.dtype)
            expanded_weights = topk_weights.unsqueeze(-1)
            output = torch.sum(expanded_weights * outputs, dim=1)

            final_hidden_states = output.type(hidden_states.dtype).reshape(
                hidden_states.shape
            )

            end_time = time.time()

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
                topk_indices, num_classes=self.num_experts
            )
            expert_mask = expert_mask.permute(2, 0, 1)

            for expert_idx in range(self.num_experts):
                expert = self.experts[expert_idx]
                mask = expert_mask[expert_idx]
                token_indices, weight_indices = torch.where(mask)

                if token_indices.numel() > 0:
                    expert_weights = topk_weights[token_indices,
                                                  weight_indices]
                    expert_input = hidden_states[token_indices]
                    expert_output = expert(expert_input)
                    weighted_output = expert_output * \
                        expert_weights.unsqueeze(-1)
                    final_hidden_states.index_add_(
                        0, token_indices, weighted_output)

            return final_hidden_states.type(hidden_states.dtype)

        def forward(self, hidden_states):
            residuals = hidden_states
            orig_shape = hidden_states.shape
            topk_indices, topk_weights = self.route_tokens(hidden_states)
            hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
            if enable_ek:
                hidden_states = self.ek_moe(
                    hidden_states, topk_indices, topk_weights
                ).view(*orig_shape)
            else:
                hidden_states = self.moe(
                    hidden_states, topk_indices, topk_weights
                ).view(*orig_shape)
            hidden_states = hidden_states + self.shared_experts(residuals)
            return hidden_states

    delattr(ds_v2, "DeepseekV2MoE")
    setattr(ds_v2, "DeepseekV2MoE", InterceptedDeepseekV2MoE)


tokenizer: Optional[AutoTokenizer] = None
model = None


def evaluate_batch(
    *,
    model_path="./",
    prompts="What is MoE Model?",
    output_max_length=64,
    enable_ek=True,
    ek_addr="localhost:5002",
    ek_model_name="deepseek_v2",
    enable_direct_path=True,
    first_k_dense_replace=None,
) -> Dict[str, Any]:
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

    intercept_missing()
    intercept_moe(
        enable_ek=enable_ek,
        ek_addr=ek_addr,
        ek_model_name=ek_model_name,
        enable_direct_path=enable_direct_path,
        first_k_dense_replace=first_k_dense_replace,
    )

    # Load the tokenizer and the model only once
    global tokenizer, model
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_path,
        )
    if model is None:
        model_config = ds_v2_config.DeepseekV2Config.from_pretrained(
            model_path)
        model = ds_v2.DeepseekV2ForCausalLM.from_pretrained(
            model_path,
            config=model_config,
            local_files_only=True,
            device_map=device,
            torch_dtype=torch.bfloat16 if device == "cuda" else "auto"
        )

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
            **model_inputs,
            max_new_tokens=output_max_length,
            pad_token_id=tokenizer.eos_token_id
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
                    token_id for token_id in output_ids if token_id != tokenizer.pad_token_id]

            content = tokenizer.decode(
                output_ids,
                skip_special_tokens=True
            ).strip("\n")

            results.append({
                "prompt": prompts[i],
                "content": content,
                "input_tokens": len(model_inputs.input_ids[i]),
                "output_tokens": len(output_ids),
            })

        # Context manager exit will automatically unwrap the model and print the report
        return {
            "results": results,
            "performance": profiler.report()
        }


def sharegpt(path, max_prompt_len=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File does not exist: {path}")
    if not os.path.isfile(path):
        raise ValueError(f"Path is not a file: {path}")
    with open(path, "r", encoding="utf-8") as f:
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
        "--ek_model_name",
        type=str,
        default="deepseek_v2",
        help="The name of the model used in ExpertKit.",
    )
    parser.add_argument(
        "--ek_addr",
        type=str,
        default="localhost:5002",
        help="The address of the ExpertKit controller.",
    )
    parser.add_argument(
        "--ek_direct_path",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable direct worker communication (bypasses controller forwarding). "
             "Implements request decomposition to match worker's expected format.",
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
    parser.add_argument(
        "--first_k_dense_replace",
        type=int,
        default=None,
        help="Number of initial dense layers (MoE starts after this). Defaults to model config value.",
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
        test_prompts = test_prompts[:args.prompt_num]
    elif args.dataset == "sharegpt":
        # Validate that dataset_path is provided
        if args.dataset_path is None:
            raise ValueError(
                "You must provide --dataset_path when using the 'sharegpt' dataset.")
        # Load prompts from ShareGPT dataset
        test_prompts = sharegpt(
            args.dataset_path, max_prompt_len=args.max_prompt_len)
        if len(test_prompts) < args.prompt_num:
            test_prompts *= (args.prompt_num // len(test_prompts)) + 1
        test_prompts = test_prompts[:args.prompt_num]
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
                prompts=test_prompts[prompts:prompts + batch_size],
                enable_ek=args.enable_ek,
                ek_addr=args.ek_addr,
                ek_model_name=args.ek_model_name,
                enable_direct_path=args.ek_direct_path,
                output_max_length=args.output_max,
                first_k_dense_replace=args.first_k_dense_replace,
            )
            aggregated_results.extend(batch_result["results"])

    if args.print_response:
        for result in aggregated_results[:5]:
            print()
            print(f"Prompt: {result['prompt']}")
            print(f"Response: {result['content']}")
            print(
                f"Input Tokens: {result['input_tokens']}, Output Tokens: {result['output_tokens']}")
            print("-" * 40)


if __name__ == "__main__":
    main()
