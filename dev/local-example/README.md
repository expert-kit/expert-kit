# Local CUDA deployment

This example runs the default Torch Worker on one NVIDIA GPU. The Host must
have the NVIDIA driver, Docker Engine, Docker Compose v2, and NVIDIA Container
Toolkit installed. Verify container GPU access before starting Expert Kit:

```bash
docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu24.04 nvidia-smi
```

The Worker image installs the locked CUDA-enabled PyTorch wheel. The Host
driver is provided to the container by NVIDIA Container Toolkit; it is not
copied into the image.

By default Compose exposes Host GPU 0 to the Worker. Select another Host GPU
with `EK_WORKER_GPU`:

```bash
EK_WORKER_GPU=1 docker compose -f dev/local-example/compose.build.yaml up --build
```

Only the selected Host GPU is exposed. It is numbered `0` inside the container,
so [`worker.yaml`](./worker.yaml) keeps `worker.device: cuda:0` and
`CUDA_VISIBLE_DEVICES=0`. One Compose Worker service runs one Worker process
for that device. Add a separately named service, Worker ID, endpoint, and GPU
reservation for each additional device.

Validate either Compose file without starting services:

```bash
docker compose -f dev/local-example/compose.yaml config
docker compose -f dev/local-example/compose.build.yaml config
```
