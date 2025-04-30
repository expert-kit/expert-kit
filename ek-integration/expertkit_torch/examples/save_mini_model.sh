#!/bin/bash

# first activate venv (replace with yourself venv)
source ~/.ds3-env/bin/activate

python ../tests/test_torch.py --save-dir ../tokenizer/ --config ../expertkit_torch/configs/config_test_mini.json