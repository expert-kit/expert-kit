# Deploying Qwen3-30B-A3B with Expert-Kit

## Overview

`Qwen3-30B-A3B` is a powerful MoE (Mixture of Experts) model with advanced reasoning capabilities. This guide explains how to deploy and run Qwen3-30B-A3B using Expert-Kit.

> **Prerequisites**: Ensure you have sufficient hardware before beginning installation. The model requires less memory than larger models due to its MoE architecture that activates only 3.3B of its 30.5B parameters.

## Table of Contents
- [Hardware Requirements](#hardware-requirements)
- [Weight Requirements](#weight-requirements)
- [Installation](#installation)
- [Deployment](#deployment)
- [Testing Your Deployment](#testing-your-deployment)

## Hardware Requirements

Environments we have already tested:

| Component | Minimum Specifications |
|-----------|----------------------|
| RAM | 32GB+ |
| CPU | Modern multi-core processor |
| Accelerator | High-end consumer GPU (RTX 3090/4090) or dual mid-range GPUs (2x RTX 3060/4060Ti) |
| Storage | 100GB+ SSD |

Note: The Qwen3-30B-A3B model can potentially run on systems with less powerful hardware compared to other models of similar capabilities due to its MoE architecture that activates only 3.3B parameters at inference time.

## Weight Requirements
The following model weight versions are compatible with Expert-Kit:

| Weight Version | Size | Status | Compatibility | Download Link |
|----------------|------|--------|---------------|--------------|
| **Qwen3-30B-A3B** | ~60GB | ✅ Tested | Full compatibility | [Hugging Face](https://huggingface.co/Qwen/Qwen3-30B-A3B) |

## Installation

### 1. Set Up the Development Environment

Source Code and Workspace Preparation

```bash
# Clone the repository
git clone https://github.com/expert-kit/expert-kit.git
cd expert-kit

# Create necessary directories
# Directory for libtorch libraries
mkdir -p vendor
# Directory for checkpoint weights (managed by Expert-Kit)
mkdir -p /tmp/expert-kit/cache
```

Configure Essential Environment Variables

```bash
# Set required environment variables (add to ~/.bashrc or ~/.zshrc for persistence)
## LibTorch configuration
export LIBTORCH=$(realpath ./vendor/libtorch)
export LD_LIBRARY_PATH=$(realpath ./vendor/libtorch/lib)
export DYLD_FALLBACK_LIBRARY_PATH=$(realpath ./vendor/libtorch/lib)
## Model weights location
export QWEN3_30B_A3B_ROOT="$(realpath ./[place_to_store_weight]/qwen3-30b-a3b/)"
## Configuration file path
export EK_CONFIG="$(realpath ./dev/hello-world.config.yaml)"
```

### 2. Install Dependencies

```bash
# Download and install libtorch
# For MacOS (ARM64):
# wget https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.7.0.zip -O /tmp/libtorch.zip
# For Linux:
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.7.0%2Bcpu.zip -O /tmp/libtorch.zip

# Extract libtorch
unzip /tmp/libtorch.zip -d ./vendor/

# Build Expert-Kit
cargo build --release
```

## Deployment

### 1. Prepare model weights

Download the model weights for inference.
```bash
# Download from Hugging Face
huggingface-cli download Qwen/Qwen3-30B-A3B --local-dir ${QWEN3_30B_A3B_ROOT}
```

### 2. Start the Database and Weight Server

```bash
# Terminal 1: Start the metadata database
docker-compose -f dev/meta-db.docker-compose.yaml up -d

# Terminal 1: Run the weight server (keep this terminal open)
cargo run --bin ek-cli weight-server --model "${QWEN3_30B_A3B_ROOT}"
```

### 3. Initialize the Metadata

```bash
# Terminal 2: Prepare the database
cargo run --bin ek-cli db migrate

# Register the model
## ⚠ Warning: Current version, model name parameter must match the last segment of the model path
cargo run --bin ek-cli model upsert --name qwen3-30b-a3b

# Schedule the experts (extract expert info from weight, and assign to worker)
cargo run --bin ek-cli schedule static --inventory ./dev/local.inventory.yaml
```

### 3. Launch the Controller and Worker

```bash
# Terminal 2: Start the controller (keep this terminal open)
cargo run --bin ek-cli controller

# Terminal 3: Start the worker (keep this terminal open)
cargo run --bin ek-cli worker
# Note: After starting the worker, the terminal will display weight loading information
```

## Testing Your Deployment

```bash
# Terminal 4: Run an inference test
# Set up the frontend Python environment
uv sync

# Navigate to the testing directory
cd ek-integration/expertkit_torch/

# Test with a simple script
python3 -m expertkit_torch.models.qwen3_moe \
  --model_path "${QWEN3_30B_A3B_ROOT}"
```

example output:
```
<think>
Okay, the user is asking about what an MoE model is. I need to explain this clearly. Let me start by recalling that MoE stands for Mixture of Experts. I remember it's a type of neural network architecture where different...
```