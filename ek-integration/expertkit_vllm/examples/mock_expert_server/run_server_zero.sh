#!/usr/bin/sh
python mock_server.py \
    --port 50051 \
    --mode zeros \
    --hidden_dim 7168 \
    --expert_dim 2048 \
    --latency_ms 0
