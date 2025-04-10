#!/bin/bash

# first activate venv (replace with yourself venv)
source ~/.ds3-env/bin/activate

python ../tests/test_torch.py --ckpt-path ../tokenizer/ --config ../expertkit_torch/configs/config_test_mini.json --temperature 0.7 --max-new-tokens 100 --input-file local_test_mini.txt