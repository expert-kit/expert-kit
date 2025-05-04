# Run Deepseek-Tiny model in Expert-Kit

`DeepSeek-Tiny(ds-tiny)` is a education-only model used to demonstrate the capabilities of the Expert-Kit. It shares the same architecture as the full Deepseek-v3 model and is tuned to be smaller by reducing the number of parameters. `ds-tiny` only contains 1.5B parameters and can be easily run on a single machine, thus help users to quickly test the Expert-Kit features.

## Step by Step Guide

1. Setup expert-kit development environment and prepare source code

```bash
git clone  https://github.com/expert-kit/expert-kit.git
cd expert-kit
cargo build
uv sync
```

2. Download the `ds-tiny` model weights

```bash
# we assume you are in the expert-kit root directory
mkdir -p ./data
wget <!TBD!> -o /tmp/ds-tiny.tar.gz
tar -xvf /tmp/ds-tiny.tar.gz -C ./data/ds-tiny/
```

3. Run the meta server and prepare the local test config

```bash
cd dev
docker-compose up -d

export CONVERTED_ROOT=$(realpath ./data/ek-ds-tiny)
mkdir -p $CONVERTED_ROOT

mkdir -p /etc/expert-kit

cat > /etc/expert-kit/config.yaml <<EOF
storage_provider: fs
storage_fs_path: $CONVERTED_ROOT
db_dsn: postgres://dev:dev@localhost:5432/dev
EOF

```

3. convert the model weights to the format used by expert-kit and prepare the cluster meta.

```bash

# split the model weights in order to make expert-kit load weight in expert granularity.
# This will cost a while, please be patient. 
# If the process is interrupted, you can resume it by running the same command again.
python py/expert_split.py \
    --model_dir ./data/ds-tiny \
    --fs_path "$CONVERTED_ROOT" \
    --model_idx_file model.safetensors.index.json \
    --check_remote  \
    --upload_missing

# register the model weights to the meta server
python py/create_model.py \
    --model_name ds_tiny \
    --fs_path "$CONVERTED_ROOT"

# create the expert distribution config based on the static configs.
python py/schedule_static.py \
  --fs_path "$CONVERTED_ROOT"  \
  --instance_name local_test \
  --inventory ./dev/local.inventory.yaml \
  --model_name ds_tiny
```

4. run the frontend controller and backend worker

```bash
# run the frontend controller
cargo run  --release --bin controller

# create a new terminal session, then run worker
EK_HOSTNAME=local-dev cargo run  --release --bin worker
```

5. run a simple test


```bash

# lib torch will be dynamically loaded by the expert-kit, so you need to set the environment variable to point to the libtorch.so
# for mac 
export DYLD_FALLBACK_LIBRARY_PATH=<LIB_TORCH_PATH>
# for linux: TODO

cd ek-integration/expertkit_torch/tests
# compare the result between the vanilla pytorch and expert-kit offloaded
python3 test_correct.py
```
