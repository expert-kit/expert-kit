#!/bin/bash

source ~/.ds3-env/bin/activate

export PYTHONPATH="$PYTHONPATH:.."
python ../expertkit_torch/main.py --ckpt-path ../tokenizer/ --config ../expertkit_torch/configs/config_test_mini.json --temperature 0.7 --max-new-tokens 100 --input-file local_test_mini.txt

# python main.py --ckpt-path ../pt/ --config configs/config_3.5B.json --temperature 0.7 --max-new-tokens 200 --input-file test.txt