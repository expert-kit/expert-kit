# expert-kit Test Infrastructure

- This example provides compose files for quickly setting up a local expert-kit testing environment. 
- It contains the minimum required components to get a functional local expert-kit cluster.

## Components

The test infrastructure includes:

- **`ek-postgres`**: Stores metadata
- **`ek-weight-server`**: Manages model weights
- **`ek-controller`**: Coordinates workers and dispatches expert compute requests
- **`ek-worker`**: Executes expert computation tasks

## Configuration Files

1. **`compose.override.yaml`**: Contains user-specific variables (ports, model directories, etc.)
2. **Backbone compose file** (choose one):
   - `compose.build.yaml`: For compiling containers from scratch
   - `compose.yaml`: For using pre-built containers
3. **`config.yaml`**: Main configuration settings
4. **`local.inventory.yaml`**: Local inventory configuration

## Quick Start Guide

### Customizing Your Setup

Modify the `compose.override.yaml` file to adjust user-specific settings such as ports and model directories.

### Option 1: Using Pre-built Containers

For the fastest setup with pre-built containers:

```bash
docker-compose -f compose.yaml -f compose.override.yaml up -d
```

### Option 2: Building Containers from Scratch

If you need to build custom containers:

```bash
docker-compose -f compose.build.yaml -f compose.override.yaml up -d
```

## Quick test

After launching, verify that your test environment is working properly:

```bash
# Check that all services are running
docker-compose ps

cd ../../ek-integration/expertkit_torch/

# Execute Qwen3 test
python3 -m expertkit_torch.models.qwen3_moe --model_path=$MODEL_ROOT --enable_ek
```