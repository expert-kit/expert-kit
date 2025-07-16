# vLLM ExpertMesh Plugin

ExpertMesh Plugin for vLLM framework.

## Installation

Install the plugin in development mode:

```bash
pip install -e .
```

## Usage

### 1. Setup Expert-Kit Service

First, ensure your Expert-Kit service is running and accessible. Refer to [Deploying Qwen3-30B-A3B with Expert-Kit](https://github.com/expert-kit/expert-kit/blob/dev/doc/tutorial/standalone/qwen3-moe-a3b-demo.md) for details.

### 2. Model Configuration

Expert-Kit configuration can be set through model configuration parameters or environment variables. The plugin supports the following configuration options:

#### Configuration Parameters

- `ek_mode`: Operation mode, default: `"expert_mode"`
- `ek_backend_addr`: Address of your Expert-Kit service, default: `"localhost:5002"`
- `ek_debug_mode`: Enable debug mode, default: `False`
- `ek_client_timeout`: gRPC timeout in seconds, default: `2`
- `ek_model_name`: Model name for Expert-Kit service (required)

#### Method 1: Model Configuration

When loading a model with vLLM, add Expert-Kit parameters to your model configuration:

```python
from vllm import LLM

# Configure Expert-Kit through model config
model_config = {
    "ek_mode": "expert_mode",
    "ek_backend_addr": "localhost:5002",
    "ek_debug_mode": False,
    "ek_client_timeout": 2,
    "ek_model_name": "Qwen/Qwen3-MoE-A3B"
}

# Create LLM with Expert-Kit configuration
llm = LLM(
    model="Qwen/Qwen3-MoE-A3B", 
    tensor_parallel_size=1,
    trust_remote_code=True,
    model_config=model_config
)
```

#### Method 2: Environment Variables

Alternatively, configure Expert-Kit using environment variables:

```bash
export EK_ENABLE=1
export EK_MODE="expert_mode"
export EK_ADDR="localhost:5002"
export EK_DEBUG_MODE="0"
export EK_CLIENT_TIMEOUT="2"
export EK_MODEL_NAME="Qwen/Qwen3-MoE-A3B"
```

Note: Environment variables take precedence over model configuration parameters.

### 3. Enable Expert-Kit Plugin

Set the `EK_ENABLE` environment variable to activate the plugin:

```bash
export EK_ENABLE=1
```

### 4. Generate Text

Generate text as you normally would with vLLM:

```python
# Enable ExpertKit
import os
os.environ["EK_ENABLE"] = "1"

from vllm import LLM

# Method 1: Using model config
llm = LLM(
    model="Qwen/Qwen3-MoE-A3B",
    tensor_parallel_size=1,
    trust_remote_code=True,
    model_config={
        "ek_backend_addr": "localhost:5002",
        "ek_model_name": "Qwen/Qwen3-MoE-A3B"
    }
)

# Generate text
outputs = llm.generate("Hello, world!", max_tokens=100)
print(outputs[0].outputs[0].text)
```

## Supported Models

This plugin currently supports:
- **Qwen3-MoE-A3B**: `Qwen/Qwen3-MoE-A3B` (requires vLLM >= 0.8.4)
- **DeepSeek-V2**: `deepseek-ai/deepseek-v2-base`

## Architecture

This plugin replaces the `DeepseekV2MoE` implementation with `ExpertKitMoE`, which routes expert computation to Expert-Kit service.

## Requirements

- vLLM >= 0.8.4 (required for Qwen3-MoE support)
- grpcio >= 1.71.0
- Protobuf >= 5.29.4

## Configuration Priority

Configuration parameters are resolved in the following order (higher priority overrides lower):

1. Environment variables (highest priority)
2. Model configuration parameters
3. Default values (lowest priority)

## Deployment Example


```python
from vllm import LLM
import os

os.environ["VLLM_MLA_DISABLE"] = "1"

os.environ["EK_ENABLE"] = "0"
os.environ["EK_MODEL_NAME"] = "qwen3"
os.environ["EK_MODE"] = "expert_mode"
os.environ["EK_ADDR"] = "localhost:5002"
os.environ["EK_CLIENT_TIMEOUT"] = "2"
os.environ["EK_DEBUG_MODE"] = "0"

prompts = [
    "Hello, my name is",
    "The president of the United",
]

llm = LLM(
        model="Qwen/Qwen3-30B-A3B",
        trust_remote_code=True,

        max_model_len=16,
        enforce_eager=True,
        cpu_offload_gb=64,
        max_num_batched_tokens=1024
    )

outputs = llm.generate(prompts)

```

## Troubleshooting

### Common Issues

1. **Missing EK_MODEL_NAME**: Ensure `ek_model_name` is set in model config or `EK_MODEL_NAME` environment variable is set.

2. **Connection timeout**: Increase `ek_client_timeout` value if your Expert-Kit service is slow to respond.

3. **Debug mode**: Set `ek_debug_mode=True` or `EK_DEBUG_MODE=1` to enable detailed logging.