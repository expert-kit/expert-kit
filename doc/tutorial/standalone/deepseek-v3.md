# Deploying Deepseek-R1 with Expert-Kit

## Overview

`Deepseek-R1` is a powerful MoE(Mixtural of Expert) model with advanced reasoning capabilities. 
This guide explains how to deploy and run Deepseek-R1 using Expert-Kit.

> ⚠️ **Prerequisites**: Ensure you have sufficient hardware (especially sufficient memory capacity to place 671b model weight) before beginning installation.

## Table of Contents
- [Hardware Requirements](#hardware-requirements)
- [Weight Requirements](#weight-requirements)
- [Installation](#installation)
- [Deployment](#deployment)
- [Testing Your Deployment](#testing-your-deployment)

## Hardware Requirements

Environments We have already tested.

| Component | Ascend + Kunpeng |
|-----------|----------------------|
| RAM | 2TB |
| CPU | Kunpeng 920 |
| Accelerator | Ascend 910B3 x1 |
| Storage | 2T SSD |

## Weight Requirements
The following model weight versions have been tested with Expert-Kit:

| Weight Version | Size | Status | Compatibility | Download Link |
|----------------|------|--------|---------------|--------------|
| **DeepSeek-R1-BF16** | 1.3TB+ | ✅ Tested | Full compatibility | [Hugging Face](https://huggingface.co/unsloth/DeepSeek-R1-BF16) |
| **DeepSeek-R1-Block-INT8** | 690GB+ | 🔄 Testing | Requires quantization support (in development) | [Hugging Face](https://huggingface.co/meituan/DeepSeek-R1-Block-INT8) |

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
export DEEPSEEK_R1_ROOT="$(realpath ./[place_to_store_weight]/deepseek-r1/)"
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
# download BF16 version from huggingface for example
huggingface-cli download unsloth/DeepSeek-R1-BF16 --local-dir ${DEEPSEEK_R1_ROOT}
```

### 2. Start the Database and Weight Server

```bash
# Terminal 1: Start the metadata database
docker-compose -f dev/meta-db.docker-compose.yaml up -d

# Terminal 1: Run the weight server (keep this terminal open)
cargo run --bin ek-cli weight-server --model "${DEEPSEEK_R1_ROOT}"
```

### 3. Initialize the Metadata

```bash
# Terminal 2: Prepare the database
cargo run --bin ek-cli db migrate

# Register the model
## ⚠ Warning: Current version, model name parameter must match the last segment of the model path
cargo run --bin ek-cli model upsert --name deepseek-r1

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

# Test with a simple scripts
python3 -m expertkit_torch.models.deepseek_v3.model \
  --model_path "${DEEPSEEK_R1_ROOT}" \
  --ek_addr 127.0.0.1:5002
```
