import unittest


import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from test_torch import *


class TestDeepseekMini(unittest.TestCase):

    def test_correct(self):
        ckpt_path = "../tokenizer/"
        config = "../expertkit_torch/configs/config_1.5B.json"
        input_file = "../examples/local_test_mini.txt"
        temperature = 0.8
        random_seed = False

        print("local mode:\n")
        local_results = deepseekv3_test(
            ckpt_path,
            config,
            input_file,
            max_new_tokens=1,
            temperature=temperature,
            random_seed= False,
            mode="local",
        )

        print("grpc mode:\n")

        grpc_results = deepseekv3_test(
            ckpt_path,
            config,
            input_file,
            max_new_tokens=1,
            temperature=temperature,
            random_seed=random_seed,
            mode="grpc",
        )

        self.assertListEqual(local_results, grpc_results)


if __name__ == "__main__":
    unittest.main()
