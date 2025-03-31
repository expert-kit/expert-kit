# vLLM ExpertMesh Plugin

ExpertMesh Plugin for vLLM framework.

## Installation

#TODO

## Usage

### 1. Setup ExpertKit Service

First, ensure your ExpertKit service is running and accessible. The service should implement the `ExpertComputation` gRPC interface defined in `expert.proto`.

### 2. Model Configuration

When loading a model with vLLM, add the `expertkit_addr` parameter to your model configuration:

```python
from vllm import LLM

# Configure ExpertKit
model_config = {
    "expertkit_addr": "localhost:50051",  # Address of your ExpertKit service
    "expertkit_timeout_sec": 2.0,         # Optional: gRPC timeout (default: 2.0s)
}

# Create LLM with ExpertKit configuration
llm = LLM(
    model="deepseek-ai/deepseek-v2-base", 
    tensor_parallel_size=1,
    trust_remote_code=True,
    model_config=model_config
)
```

### 3. Enable ExpertKit Plugin

Set the `EXPERTKIT_ENABLE` environment variable to activate the plugin:

```bash
export EXPERTKIT_ENABLE=1
```

### 4. Generate Text

Generate text as you normally would with vLLM:

```python
outputs = llm.generate("Hello, world!", max_tokens=100)
```

## Architecture

This plugin replaces the `DeepseekV2MoE` implementation with `ExpertKitMoE`, which routes expert computation to ExpertKit service.

## Requirements

- vLLM
- PyTorch >= 2.6.0
- grpcio >= 1.71.0
- Protobuf >= 5.29.4

## Deployment Example

To deploy the ExpertKit service on a separate GPU machine:

```bash
# On the ExpertKit server
python expertkit_server.py --port 50051

# On the vLLM server
export EXPERTKIT_ENABLE=1
python -m vllm.entrypoints.openai.api_server \
    --model deepseek-ai/deepseek-v2-base \
    --model-config '{"expertkit_addr": "expert-server:50051"}'
```

## References

- [vLLM Documentation](https://github.com/vllm-project/vllm)
- [gRPC Documentation](https://grpc.io/docs/languages/python/)
- [DeepSeek-R1 Documentation](https://github.com/deepseek-ai/DeepSeek-R1)