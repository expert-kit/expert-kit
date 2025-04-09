import os
import json
from argparse import ArgumentParser
import time
from typing import List

import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from safetensors.torch import load_model

from model import Transformer, ModelArgs

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
    temperature: float = 1.0
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
    assert max(prompt_lens) <= model.max_seq_len, f"Prompt length exceeds model maximum sequence length (max_seq_len={model.max_seq_len})"
    total_len = min(model.max_seq_len, max_new_tokens + max(prompt_lens))
    tokens = torch.full((len(prompt_tokens), total_len), -1, dtype=torch.long, device="cuda")
    for i, t in enumerate(prompt_tokens):
        tokens[i, :len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
    prev_pos = 0
    finished = torch.tensor([False] * len(prompt_tokens), device="cuda")
    prompt_mask = tokens != -1
    for cur_pos in range(min(prompt_lens), total_len):
        logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
        if temperature > 0:
            next_token = sample(logits, temperature)
        else:
            next_token = logits.argmax(dim=-1)
        next_token = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token)
        tokens[:, cur_pos] = next_token
        finished |= torch.logical_and(~prompt_mask[:, cur_pos], next_token == eos_id)
        prev_pos = cur_pos
        if finished.all():
            break
    completion_tokens = []
    for i, toks in enumerate(tokens.tolist()):
        toks = toks[prompt_lens[i]:prompt_lens[i]+max_new_tokens]
        if eos_id in toks:
            toks = toks[:toks.index(eos_id)]
        completion_tokens.append(toks)
    return completion_tokens


def main(
    ckpt_path: str,
    config: str,
    input_file: str = "",
    max_new_tokens: int = 100,
    temperature: float = 1.0,
) -> None:
  world_size = int(os.getenv("WORLD_SIZE", "1"))
  rank = int(os.getenv("RANK", "0"))
  local_rank = int(os.getenv("LOCAL_RANK", "0"))
  if world_size > 1:
      dist.init_process_group("nccl")
  global print
  if rank != 0:
      print = lambda *_, **__: None
  torch.cuda.set_device(local_rank)
  torch.set_default_dtype(torch.bfloat16)
  torch.set_num_threads(8)
  seed = int(time.time()) % (2**32)  # 使用当前时间的秒数作为种子，并限制在 32 位整数范围内
  torch.manual_seed(seed)
#   random.seed(seed)
#   np.random.seed(seed)
#   torch.manual_seed(965)
  with open(config) as f:
      args = ModelArgs(**json.load(f))
  print(args)
  with torch.device("cuda"):
      model = Transformer(args)
  tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
  tokenizer.decode(generate(model, [tokenizer.encode("DeepSeek")], 2, -1, 1.)[0])
  
  # load_model(model, os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors"))

  
  with open(input_file) as f:
      prompts = [line.strip() for line in f.readlines()]
  assert len(prompts) <= args.max_batch_size, f"Number of prompts exceeds maximum batch size ({args.max_batch_size})"
  prompt_tokens = [tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True) for prompt in prompts]
  completion_tokens = generate(model, prompt_tokens, max_new_tokens, tokenizer.eos_token_id, temperature)
  completions = tokenizer.batch_decode(completion_tokens, skip_special_tokens=True)
  for prompt, completion in zip(prompts, completions):
      print("Prompt:", prompt)
      print("Completion:", completion)
      print()

  if world_size > 1:
      dist.destroy_process_group()
  


if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("--ckpt-path", type=str, required=True)
  parser.add_argument("--config", type=str, required=True)
  parser.add_argument("--input-file", type=str, default="")
  parser.add_argument("--interactive", action="store_true")
  parser.add_argument("--max-new-tokens", type=int, default=200)
  parser.add_argument("--temperature", type=float, default=0.2)
  args = parser.parse_args()
  assert args.input_file or args.interactive, "Either input-file or interactive mode must be specified"
  main(args.ckpt_path, args.config, args.input_file, args.max_new_tokens, args.temperature)