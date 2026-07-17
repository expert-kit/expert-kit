# Expert Kit Torch integration

This package connects a Hugging Face Qwen MoE model to Expert Kit. The model
keeps attention and routing in its local PyTorch process. Each routed MoE layer
sends the final expert assignments and FP32 routing weights through
`expertkit-transport`, then receives the weighted result aggregated across
Workers.

The current adapter targets the `Qwen3MoeSparseMoeBlock` implementation in
Transformers 4.57.3. Older per-expert clients and the unmigrated DeepSeek and
Mixtral adapters are not included.

## Install

Create the locked environment from this directory:

```bash
uv sync --locked
source .venv/bin/activate
```

The local uv configuration installs `expertkit-transport` from
`../../ek-transport`. A packaged deployment must make the same
`expertkit-transport==0.1.0` release available.

## Required deployment

Start the Controller, Weight Server, and one Python Worker process per compute
device first. The Controller must publish every expert needed by the model as
ready before generation begins. The numeric instance ID passed below must match
the model instance registered by those Workers.

The MVP uses plaintext gRPC and HTTP. Run it only on a trusted, isolated
cluster network with network-level access restrictions. TLS and application
authentication are not implemented.

## Qwen generation check

The migration check always uses one prompt and allows at most 20 new tokens:

```bash
ek-qwen-smoke \
  --model-path /models/Qwen3-30B-A3B \
  --controller-endpoint 10.0.0.10:5002 \
  --instance-id 1
```

A successful run prints `Qwen smoke passed`, the generated-token count, and
non-empty generated text. The command fails when the request does not traverse
Expert Kit successfully, produces no readable text, or violates the fixed
generation limit.

The same check can be run through pytest when a real deployment is available:

```bash
EK_QWEN_MODEL_PATH=/models/Qwen3-30B-A3B \
EK_QWEN_CONTROLLER_ENDPOINT=10.0.0.10:5002 \
EK_QWEN_INSTANCE_ID=1 \
uv run pytest tests/test_qwen_generation.py -m qwen
```

This test requires the actual model weights, compatible GPUs, and a running
Expert Kit deployment. It is skipped when `EK_QWEN_MODEL_PATH` is absent.
