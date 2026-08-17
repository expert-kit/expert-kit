# Expert Kit Transport

`expertkit-transport` is the shared Python communication package between
Frontend integrations and Workers. It accepts a complete routed-MoE layer,
groups and splits it by Worker, dispatches bounded asynchronous calls, retries
eligible failures within the original deadline, and aggregates weighted
partial outputs in FP32 before returning the model activation dtype.

The package contains a portable `grpc.aio` data path, an experimental same-host
shared-memory data path, a CUDA-only NCCL data path, and a Mooncake Transfer
Engine data path. Shared memory keeps gRPC for small session, notification,
completion, and error messages, but places Tensor bytes in fixed reusable
slots. NCCL and Transfer Engine also retain a private gRPC control plane.
NCCL moves payloads through an ordered static process group; Transfer Engine
uses registered fixed arenas so the Worker can READ inputs and WRITE outputs
with one-sided transfers. All paths use the same routing behavior and bounded
Worker admission. A Worker process enables exactly one data Transport.

Generic layer routing is under `expertkit_transport.routing`. Controller
topology watching is under `expertkit_transport.controller`. The concrete
Frontend sender and Worker receiver for each data path live together under
`expertkit_transport.transports.grpc`, `expertkit_transport.transports.shm`,
`expertkit_transport.transports.nccl`, or
`expertkit_transport.transports.transfer_engine`. Compute backends do not
import these concrete implementations.

## Install and test

```bash
uv sync --locked
uv run --package expertkit-transport ruff check ek-transport/src ek-transport/tests
uv run --package expertkit-transport ruff format --check ek-transport/src ek-transport/tests
uv run --package expertkit-transport pytest ek-transport/tests
```

The Transfer Engine extra intentionally does not install a public Mooncake
wheel. Run `uv sync --locked --extra transfer-engine` first, then install the EK
safety-patched Linux wheel into that environment and invoke `.venv/bin/python`
or `uv run --no-sync`. The runtime refuses to register memory unless the binding
advertises terminal batch semantics and, for CUDA, a GPUDirect acquire fence.
The forced `nvlink_intra` profile additionally requires registration reference
counts and drained IPC-cache invalidation. Experimental RDMA additionally
requires `EK_FORCE_CONFIGURED_RDMA_TRANSPORT`,
`EK_DRAINED_RDMA_REMOTE_DESCRIPTOR_INVALIDATION`, and
`invalidate_drained_rdma_segment(target_session)`; both forced profiles must
report the exact installed backend through `get_configured_backend()`. Importing
a stock wheel is not enough. The reproducible source patches, validated wheel
versions and hashes, build flags, and native API are recorded in
[`third_party/mooncake/README.md`](../third_party/mooncake/README.md).
The current dual-backend artifact is linked against CUDA 12.8; follow that
document's runtime-matching guidance when PyTorch bundles an older CUDA runtime.

The real NCCL multi-process tests require at least two visible CUDA devices and
a PyTorch build with NCCL support. They are marked `cuda` and skip cleanly when
those prerequisites are absent.

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
   uv run --package expertkit-transport pytest ek-transport/tests/conformance
   ```

Do not edit the internals of the existing gRPC or SHM implementations to add a
new Transport. SHM's small gRPC service is private notification machinery; it
does not enable the gRPC Tensor payload path in an SHM Worker.

## Current limits

- Input and output values use `torch.Tensor`.
- The gRPC Tensor data path uses unary plaintext RPCs and copies Tensor bytes
  through Host memory.
- Shared memory requires both processes to see the same `/dev/shm` namespace.
  The current private-file permissions also require the same Unix user. CUDA
  mappings are registered as pinned memory, but transfers between the Frontend
  GPU and Worker GPU still pass through Host memory.
- Transfer Engine retains at most 4096 process-lifetime Frontend epoch
  tombstones. A closed epoch cannot be reopened. If a Frontend crashes before
  `CloseSession`, the Worker rejects a different runtime generation at the same
  endpoint; restart that Worker before reusing the endpoint.
- NCCL requires CUDA on every participant and a static, uniquely ranked process
  group. Membership changes and communicator failures currently require the
  group to be recreated; elastic rank replacement is not implemented.
- If `torch.distributed` is already initialized, the MVP reuses its global
  `WORLD` group and requires that group to contain exactly the configured EK
  ranks. It cannot yet create an independent cross-job group alongside an
  unrelated vLLM or torchrun world.
- NCCL point-to-point operations do not provide request tags. Calls to one peer
  are submitted in protocol order, and cancellation or a deadline cannot abort
  an operation after its matching NCCL work has been submitted. Cleanup waits
  for that work before the communication slot is reused.
- Transfer Engine keeps one process-wide Mooncake engine and one registered
  arena per Frontend connection or Worker receiver. The first implementation
  copies once between that arena and the caller/execution slot on each side;
  it is not an execution-slot zero-copy path yet.
- Mooncake's synchronous Python calls run in a bounded executor. Because the
  native API cannot safely cancel an in-flight DMA, cancellation and deadline
  handling retain the slot until the native operation is terminal.
- P2P handshake mode requires every process to advertise a distinct reachable
  `host:port`. One process runtime still selects exactly one backend; a single
  runtime cannot mix intra-node NVLink and cross-host RDMA peers. OpenSession
  rejects peers whose selected backend differs.
- Graceful `nvlink_intra` and capability-gated RDMA retirement use the same
  two-phase close: prepare gates every sibling session for a remote target and
  drains DMA, then the owner synchronously deregisters its arena before commit.
  Commit invalidates the Worker's backend-specific remote-target cache. RDMA
  evicts only the stale address/rkey descriptor; it deliberately keeps shared
  QPs installed. Any ambiguous step quarantines the arena and requires process
  restart.
- RDMA is disabled unless both sides explicitly set
  `enable_experimental_rdma=True`, use indexed CUDA devices and explicit RDMA
  device names, use `P2PHANDSHAKE`, and install a native binding that can prove
  the exact installed transport set is `{rdma}`, report `rdma` as the actual
  backend, and evict a drained remote descriptor. A wheel missing any one of
  those contracts fails before memory registration.
- RDMA route removal and re-add within the same live remote runtime generation
  reuse the still-valid QP after refreshing the descriptor. Because graceful
  close does not destroy a shared QP, the Worker retains an endpoint-to-runtime
  generation tombstone. A new process generation must advertise a new endpoint
  or restart the Worker before reusing the old endpoint. Network partitions,
  deregistration failures, and crash windows remain fault-injection gates before
  this opt-in profile can be promoted to production.
- TCP and per-peer mixed backend selection remain lower-level experiments.
  TENT selection is rejected even when its environment variables are set to
  `0`; run with `MC_USE_TENT` and `MC_USE_TEV1` unset.
- NVSHMEM and Arrow Flight Transports are not present.
- TLS, authentication, and authorization are not implemented. Endpoints must
  remain inside a trusted isolated network.
