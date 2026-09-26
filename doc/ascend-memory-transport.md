# Ascend memory transport

## Implementation

The model-facing APIs are unchanged: `BlockingRoutedMoEClient.execute()` and
`WorkerTransport.execute()` still accept the same tensors. The TE runtime is
now a Tensor facade over `MemoryRegion`, `MemorySlice`, and `MemoryTransport`.
The memory contract has no expert IDs, layer IDs, Controller, or model types.

The first NPU driver uses Mooncake Ascend Direct with synchronous ADXL
transfers. E READs the A input arena and WRITEs the result into A's arena.
gRPC carries session setup, request metadata, and completion/close messages.
Activation payloads do not travel through gRPC in this profile.

Each process binds one indexed NPU. Registered arenas retain their original
allocation, align the registration address and length to 2 MiB, and remain
registered across requests. Optional `max_registered_bytes` limits the total
registered memory. Worker admission also accounts for alignment overhead.

Before a native call, executor threads select the correct NPU. Local staging
events order producer copies before READ/WRITE. After known successful remote
completion, an ACL device synchronization orders local consumers. This first
receive barrier synchronizes the whole device; it needs performance profiling
before replacement with a narrower fence.

Cancellation and deadlines wait for the submitted native call to return.
Native transfer failures are treated as an unknown DMA state: the process
retains registrations and owners, refuses further requests, and requires
restart. There is no automatic failed-transfer retry or payload fallback to
TCP. Closed Ascend endpoints retain a runtime-generation tombstone; live peer
restart at the same endpoint is unsupported.

## Build

Use Linux ARM64, CPython 3.12, and CANN 9.0.1. CUDA is not required. The
separate `third_party/mooncake/build-manifest-ascend.toml` records upstream,
submodule archive checksums, and the ordered patch series. Existing CUDA
build inputs remain separate.

Builder packages include CMake, GCC/G++, Git, glog/gflags, jsoncpp, ibverbs,
unwind, YAML, CURL, SSL, and NUMA development headers. Python needs setuptools
and wheel. Supply the three verified archives named in the manifest:

```bash
python third_party/mooncake/build-ascend.py \
  --input-dir /path/to/verified-archives \
  --work-dir /path/outside/repository/native-build \
  --artifact-dir /path/outside/repository/artifacts --jobs 8
```

The wheel has a `linux_aarch64` platform tag and depends on the runtime's CANN
and system libraries, including `libglog.so.0`, gflags, jsoncpp, yaml-cpp,
ibverbs, and NUMA. It is not a self-contained manylinux wheel. Install it in
both A and E environments. The accompanying JSON receipt records hashes and
required native capabilities.

The native patches force the installed transport set to `{ascend}`, report
`ascend_direct`, fix the upstream async environment parsing, expose the
receive barrier, and prevent freeing/retrying an ambiguous failed batch.

## Deployment

Add this to the Ascend cluster YAML:

```yaml
transport:
  type: transfer_engine
  protocol: ascend_direct
  max_workers: 2
  client_max_in_flight: 1
```

The generator creates an A config file and sets `EK_TRANSPORT_CONFIG`.
Runtime device selection comes from the existing client, so different DP
processes can share the file without all selecting NPU 0. Mooncake binds and
publishes a distinct actual P2P port per process.

A and E use host networking with peer-reachable host addresses and distinct
control/weight-peer ports. Mount Ascend drivers and `/etc/hccn.conf` as needed
by the environment. The initial profile requires:

```text
ASCEND_USE_ASYNC_TRANSFER=0
ASCEND_BUFFER_POOL=0:0
ASCEND_USE_SHORT_CONNECTION=0
```

Keep V4 ModelSlim-specific settings when generating V4 deployment files:
`quantization: ascend`, eager mode, batch limits, and required weight metadata.

## Verification

`scripts/smoke-ascend-memory.py` runs real NPU READ and WRITE between two
processes, validates every byte, and uses non-default producer streams.
Its TCP channel exchanges descriptors and acknowledgements only. Choose
different NPUs and the same run ID, iteration count, and control endpoint.

```bash
python scripts/smoke-ascend-memory.py --role listener --device npu:0 \
  --segment-host HOST_A --control-host HOST_A --control-port 47180 --run-id test-1
python scripts/smoke-ascend-memory.py --role initiator --device npu:8 \
  --segment-host HOST_B --control-host HOST_A --control-port 47180 --run-id test-1
```

For model validation, start the existing V4-Flash 8A8E deployment using the new
profile, wait for all experts and frontend readiness, and submit real HTTP
generation requests. Inspect both A and E logs for native failures, stalled
transfers, unhealthy runtimes, or output errors. Compare identical prompts,
seeds, and concurrency with the previous gRPC run.

Forced Ascend selection does not establish which physical route ADXL used.
Same-host HCCS, cross-host RDMA, and one runtime serving both kinds of peers
must each be verified on the hardware. No model speedup or hot-restart support
is claimed before those experiments succeed.
