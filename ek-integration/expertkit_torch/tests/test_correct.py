import os
import json
from argparse import ArgumentParser
import time
from typing import List

import unittest
from unittest.mock import Mock, patch, MagicMock

import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from safetensors.torch import load_model, save_file

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from expertkit_torch.model import Transformer, ModelArgs
from test_torch import *

class TestDeepseekMini(unittest.TestCase):

    def test_correct(self):
        ckpt_path = "../tokenizer/"
        config = "../expertkit_torch/configs/config_test_mini.json"
        input_file = "../examples/local_test_mini.txt"
        max_tokens = 100
        temperature = 0.8
        random_seed = False        

        print("local mode:\n")
        local_results = deepseekv3_test(ckpt_path, config, input_file, max_tokens, temperature, random_seed, mode="local")

        print("grpc mode:\n")

        grpc_results = deepseekv3_test(ckpt_path, config, input_file, max_tokens, temperature, random_seed, mode="grpc")

        self.assertListEqual(local_results, grpc_results)


if __name__ == '__main__':
    unittest.main()