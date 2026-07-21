# Expert Kit Transport

`expertkit-transport` is the shared Python communication package between
Frontend integrations and Workers. It accepts a complete routed-MoE layer,
groups and splits it by Worker, dispatches bounded asynchronous calls, retries
eligible failures within the original deadline, and aggregates weighted
partial outputs in FP32 before returning the model activation dtype.

The package contains a portable `grpc.aio` data path and an experimental
same-host shared-memory data path. Shared memory keeps gRPC for small session,
notification, completion, and error messages, but places Tensor bytes in fixed
reusable slots. Both paths use the same routing behavior and bounded Worker
admission. A Worker process enables exactly one data Transport.

Generic layer routing is under `expertkit_transport.routing`. Controller
topology watching is under `expertkit_transport.controller`. The concrete
Frontend sender and Worker receiver for each data path live together under
`expertkit_transport.transports.grpc` or `expertkit_transport.transports.shm`.
Compute backends do not import these concrete implementations.

## Install and test

```bash
uv sync --project ek-transport --locked
uv run --project ek-transport ruff check ek-transport/src ek-transport/tests
uv run --project ek-transport ruff format --check ek-transport/src ek-transport/tests
uv run --project ek-transport pytest ek-transport/tests
```

Most users install this package through `expertkit-worker`,
`expertkit-torch`, or `expertkit-vllm` instead of calling it directly.

## Adding another Transport

A Transport must implement both sides of one data path:

1. Add a directory under `expertkit_transport/transports/<name>` containing a
   Frontend `WorkerTransport` and a Worker `WorkerBatchReceiver`. The receiver
   uses `ReceiverQueue` for bounded waiting, drain, and idle tracking instead
   of creating another Worker queue.
2. Keep payload encoding, transfer buffers, completion signaling, and protocol
   errors inside that directory. Routing receives and returns ordinary
   `torch.Tensor` values and must not call protocol-specific buffer hooks.
3. Add one explicit Frontend branch to
   `expertkit_transport/transports/factory.py` and one explicit Worker branch to
   `expertkit_worker/factory.py`.
4. Add a `WorkerTransportType` value to the shared protocol, preserve it through
   Controller topology, and add one strict Worker configuration variant.
5. Run the common behavior tests with:

   ```bash
   uv run --project ek-transport pytest ek-transport/tests/conformance
   ```

Do not edit the internals of the existing gRPC or SHM implementations to add a
new Transport. SHM's small gRPC service is private notification machinery; it
does not enable the gRPC Tensor payload path in an SHM Worker.

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
- RDMA, NCCL, NVSHMEM, Arrow Flight, and Mooncake Transports are not present.
- TLS, authentication, and authorization are not implemented. Endpoints must
  remain inside a trusted isolated network.
