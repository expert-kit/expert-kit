import os
import json
from argparse import ArgumentParser
import time
from typing import List

import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from safetensors.torch import load_model, save_file

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from expertkit_torch.models.deepseek_v3.model import Transformer, ModelArgs

DEFAULT_DEVICE = torch.device("cuda")


def sample(logits, temperature: float = 1.0):
    """
    Samples a token from the logits using temperature scaling.

    Args:
        logits (torch.Tensor): The logits tensor for token predictions.
        temperature (float, optional): Temperature for scaling logits. Defaults to 1.0.

    Returns:
        torch.Tensor: The sampled token.
    """
    logits = logits / max(temperature, 1e-5)
    probs = torch.softmax(logits, dim=-1)
    return probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)


@torch.inference_mode()
def generate(
    model: Transformer,
    prompt_tokens: List[List[int]],
    max_new_tokens: int,
    eos_id: int,
    temperature: float = 1.0,
    device: str = DEFAULT_DEVICE,
) -> List[List[int]]:
    """
    Generates new tokens based on the given prompt tokens using the specified model.

    Args:
        model (Transformer): The transformer model used for token generation.
        prompt_tokens (List[List[int]]): A list of lists containing the prompt tokens for each sequence.
        max_new_tokens (int): The maximum number of new tokens to generate.
        eos_id (int): The end-of-sequence token ID.
        temperature (float, optional): The temperature value for sampling. Defaults to 1.0.

    Returns:
        List[List[int]]: A list of lists containing the generated tokens for each sequence.
    """
    prompt_lens = [len(t) for t in prompt_tokens]
    assert (
        max(prompt_lens) <= model.max_seq_len
    ), f"Prompt length exceeds model maximum sequence length (max_seq_len={model.max_seq_len})"
    total_len = min(model.max_seq_len, max_new_tokens + max(prompt_lens))
    tokens = torch.full(
        (len(prompt_tokens), total_len), -1, dtype=torch.long, device=device
    )
    for i, t in enumerate(prompt_tokens):
        tokens[i, : len(t)] = torch.tensor(t, dtype=torch.long, device=device)
    prev_pos = 0
    finished = torch.tensor([False] * len(prompt_tokens), device=device)
    prompt_mask = tokens != -1
    for cur_pos in range(min(prompt_lens), total_len):
        logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
        if temperature > 0:
            next_token = sample(logits, temperature)
        else:
            next_token = logits.argmax(dim=-1)
        next_token = torch.where(
            prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token
        )
        tokens[:, cur_pos] = next_token
        finished |= torch.logical_and(~prompt_mask[:, cur_pos], next_token == eos_id)
        prev_pos = cur_pos
        if finished.all():
            break
    completion_tokens = []
    for i, toks in enumerate(tokens.tolist()):
        toks = toks[prompt_lens[i] : prompt_lens[i] + max_new_tokens]
        if eos_id in toks:
            toks = toks[: toks.index(eos_id)]
        completion_tokens.append(toks)
    return completion_tokens


def deepseekv3_test(
    ckpt_path: str,
    config: str,
    input_file: str = "",
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    random_seed: bool = True,
    mode: str = "local",
) -> None:
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if world_size > 1:
        dist.init_process_group("nccl")
    global print
    if rank != 0:
        print = lambda *_, **__: None
    if device == torch.device("cuda"):
        torch.cuda.set_device(local_rank)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    if random_seed:
        seed = int(time.time()) % (2**32)
        torch.manual_seed(seed)
    else:
        torch.manual_seed(965)
    with open(config) as f:
        args = ModelArgs(**json.load(f))
    args.expertkit_mode = mode
    print(args)
    with device:
        model = Transformer(args)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    tokenizer.decode(
        generate(model, [tokenizer.encode("DeepSeek")], 2, -1, 1.0, device=device)[0]
    )

    print("Loading model from", os.path.join(ckpt_path, f"model.safetensors"))
    load_model(model, os.path.join(ckpt_path, f"model.safetensors"))
    print("model loaded")

    with open(input_file) as f:
        prompts = [line.strip() for line in f.readlines()]
    assert (
        len(prompts) <= args.max_batch_size
    ), f"Number of prompts exceeds maximum batch size ({args.max_batch_size})"
    prompt_tokens = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True
        )
        for prompt in prompts
    ]
    completion_tokens = generate(
        model,
        prompt_tokens,
        max_new_tokens,
        tokenizer.eos_token_id,
        temperature,
        device=device,
    )
    completions = tokenizer.batch_decode(completion_tokens, skip_special_tokens=True)
    for prompt, completion in zip(prompts, completions):
        print("Prompt:", prompt)
        print("Completion:", completion)
        print()

    if world_size > 1:
        dist.destroy_process_group()

    return completions


def random_init_model_save(dir: str, config: str, device=DEFAULT_DEVICE) -> None:
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if device == torch.device("cuda"):
        torch.cuda.set_device(local_rank)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    seed = int(time.time()) % (2**32)
    torch.manual_seed(seed)
    with open(config) as f:
        args = ModelArgs(**json.load(f))
    args.save_model = True
    print(args)
    with torch.device(device):
        model = Transformer(args)
    save_file(model.state_dict(), f"{dir}/model.safetensors")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ckpt-path", type=str)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input-file", type=str, default="")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--mode", type=str, default="local")
    parser.add_argument("--save-dir", type=str, default="")
    args = parser.parse_args()
    if args.save_dir != "":
        random_init_model_save(args.save_dir, args.config)
    else:
        assert (
            args.input_file or args.interactive
        ), "Either input-file or interactive mode must be specified"
        deepseekv3_test(
            args.ckpt_path,
            args.config,
            args.input_file,
            args.max_new_tokens,
            args.temperature,
            mode=args.mode,
        )
