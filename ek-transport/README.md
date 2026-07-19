# Expert Kit Transport

`expertkit-transport` is the shared Python communication package between
Frontend integrations and Workers. It accepts a complete routed-MoE layer,
groups and splits it by Worker, dispatches bounded asynchronous calls, retries
eligible failures within the original deadline, and aggregates weighted
partial outputs in FP32 before returning the model activation dtype.

The package contains a portable `grpc.aio` data path and an experimental
same-host shared-memory data path. Shared memory keeps gRPC for small session,
notification, completion, and error messages, but places Tensor bytes in fixed
reusable slots. Both paths use the same orchestration and Worker admission.
Future communication mechanisms can implement the same package contracts
without changing Compute backends.

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
- Cross-host computation uses unary plaintext gRPC and copies Tensor bytes
  through Host memory.
- Shared memory requires both processes to see the same `/dev/shm` namespace.
  The current private-file permissions also require the same Unix user. CUDA
  mappings are registered as pinned memory, but transfers between the Frontend
  GPU and Worker GPU still pass through Host memory.
- A Frontend crash cannot send normal session cleanup. The Worker keeps that
  session mapped until it exits; the 64-session bound prevents unbounded
  registration.
- RDMA, NCCL, NVSHMEM, Arrow Flight, and Mooncake adapters are not present.
- TLS, authentication, and authorization are not implemented. Endpoints must
  remain inside a trusted isolated network.
