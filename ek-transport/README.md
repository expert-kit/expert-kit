# Expert Kit Transport

`expertkit-transport` is the shared Python communication package between
Frontend integrations and Workers. It accepts a complete routed-MoE layer,
groups and splits it by Worker, dispatches bounded asynchronous calls, retries
eligible failures within the original deadline, and aggregates weighted
partial outputs in FP32 before returning the model activation dtype.

The MVP contains one adapter using `grpc.aio`. Protobuf encoding, message-size
limits, request matching, waiting capacity, and Host staging stay inside that
adapter. Future communication mechanisms can implement the same package
contracts without changing Compute backends.

## Install and test

```bash
uv sync --project ek-transport --locked
uv run --project ek-transport ruff check ek-transport/src ek-transport/tests
uv run --project ek-transport ruff format --check ek-transport/src ek-transport/tests
uv run --project ek-transport pytest ek-transport/tests
```

Most users install this package through `expertkit-worker`,
`expertkit-torch`, or `expertkit-vllm` instead of calling it directly.

## Current limits

- Input and output values use `torch.Tensor`.
- Computation uses unary plaintext gRPC and copies Tensor bytes through Host
  memory.
- SHM, RDMA, NCCL, NVSHMEM, Arrow Flight, and Mooncake adapters are not present.
- TLS, authentication, and authorization are not implemented. Endpoints must
  remain inside a trusted isolated network.
