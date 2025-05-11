# Run Deepseek-Tiny model in Expert-Kit

`DeepSeek-Tiny(ds-tiny)` is a education-only model used to demonstrate the capabilities of the Expert-Kit. It shares the same architecture as the full Deepseek-v3 model and is tuned to be smaller by reducing the number of parameters. `ds-tiny` only contains 1.5B parameters and can be easily run on a single machine, thus help users to quickly test the Expert-Kit features.

## Step by Step Guide

1. Setup expert-kit development environment and prepare source code

```bash
git clone  https://github.com/expert-kit/expert-kit.git
cd expert-kit


# make sure these directories exist
mkdir vendor
mkdir -p /tmp/expert-kit/cache

# make sure these environment variables are set, especially when you create a new terminal session.
export LIBTORCH=$(realpath ./vendor/libtorch)
export DYLD_FALLBACK_LIBRARY_PATH=$(realpath ./vendor/libtorch/lib)
export LD_LIBRARY_PATH=$(realpath ./vendor/libtorch/lib)
export DS_TINY_ROOT="$(realpath ./ek-db/resources/ds-tiny/)"
export EK_CONFIG="$(realpath ./dev/hello-world.config.yaml)"


# download libtorch from https://pytorch.org/, and place it in the vendor directory of expert-kit
# Mac: https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.7.0.zip
# Linux: https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.7.0%2Bcpu.zip
wget https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.7.0.zip -O /tmp/libtorch.zip
unzip /tmp/libtorch.zip -d ./vendor/


# download the expert-kit source code
# since we are using git lfs, make sure you have git-lfs installed and initialized
git lfs fetch --all  # download the ds-tiny weight

cargo build --release
uv sync
```

2. run weight server and meta db

```bash
# run meta db
docker-compose -f dev/meta-db.docker-compose.yaml up -d

# run weight server
cargo run --bin ek-cli weight-server --model "${DS_TINY_ROOT}"
```

3. prepare the metadata

```bash
cargo run --bin ek-cli db migrate
cargo run --bin ek-cli model upsert --name ds-tiny
cargo run --bin ek-cli schedule  static --inventory ./dev/local.inventory.yaml
```

4. run the frontend controller and backend worker

```bash
# run the frontend controller
cargo run --bin ek-cli controller
# create a new terminal session, then run worker
# pay attention to the required environment variable
cargo run --bin ek-cli worker
```

5. run a simple test

```bash
cd ek-integration/expertkit_torch/
# compare the result between the vanilla pytorch and expert-kit offloaded
python3 -m expertkit_torch.models.deepseek_v3.model --model_path "${DS_TINY_ROOT}" --ek_addr 127.0.0.1:5002

# some random char would be generated
# example: Seeds TESToth inventory inventory inventory inventory inventoryothothothothothothothothothothothothothothothothothothothothothothothoth апреothothoth conson conson conson conson conson地道 conson conson conson conson conson地道 conson conson
```
