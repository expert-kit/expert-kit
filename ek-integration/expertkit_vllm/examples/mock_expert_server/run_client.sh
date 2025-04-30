#!/usr/bin/sh
python test_client.py --server localhost:50051

export VLLM_MLA_DISABLE=1
export EXPERTKIT_ENABLE=1