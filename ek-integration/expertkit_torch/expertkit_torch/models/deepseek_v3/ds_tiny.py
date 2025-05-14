import os
from transformers import AutoTokenizer
from transformers.models.deepseek_v3 import modeling_deepseek_v3 as ds_v3
from transformers.models.deepseek_v3 import configuration_deepseek_v3 as ds_v3_config
import json
import torch
import os
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    return args


def create_random_model():
    args = get_args()
    f = open(args.config_path, "r")
    config_obj = json.load(f)
    torch.manual_seed(0)
    f.close()

    model_args = ds_v3_config.DeepseekV3Config(**config_obj)
    print(f"config loaded from {args.config_path}")
    model = ds_v3.DeepseekV3ForCausalLM(
        config=model_args,
    )
    os.makedirs(args.output_path, exist_ok=True)
    model.save_pretrained(args.output_path, max_shard_size="20MB")
    print(f"model saved to {args.output_path}")


if __name__ == "__main__":
    create_random_model()
