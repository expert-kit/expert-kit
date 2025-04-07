#!/bin/bash

source ~/.ds3-env/bin/activate

python ../inference/main.py --ckpt-path ../tokenizer/ --config ../inference/configs/config_test_mini.json --temperature 0.7 --max-new-tokens 100 --input-file local_test.txt

# python main.py --ckpt-path ../pt/ --config configs/config_3.5B.json --temperature 0.7 --max-new-tokens 200 --input-file test.txt