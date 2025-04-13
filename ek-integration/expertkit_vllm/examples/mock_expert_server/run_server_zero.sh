#!/usr/bin/sh
python mock_server.py \
    --port 50051 \
    --mode zero \
    --hidden_dim 2048 \
    --expert_dim 7168 \
    --latency_ms 0
