#!/usr/bin/sh

# replace the weights_path with your own path
python mock_server.py \
    --port 50051 \
    --mode compute \
    --hidden_dim 32 \
    --expert_dim 16 \
    --weights_path /home/liuyang/expert-kit/ek-integration/expertkit_torch/tokenizer/model.safetensors \
    --latency_ms 0
