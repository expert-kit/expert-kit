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

For source-mounted validation, preserve the image's existing CANN Python paths
when adding source directories to `PYTHONPATH`. vLLM Ascend imports the CANN
`acl` Python module during worker initialization. Test that import and NPU
enumeration in the same container configuration before loading the model.

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

## Initial real-model results (2026-09-26)

Code `55185b8` and the CANN 9.0.1 Ascend wheel ran
DeepSeek-V4-Flash-0731-w8a8 on one 16-device Ascend host with 8A8E.
All 11,008 experts loaded. Eight real chat requests passed content checks,
including repeated arithmetic, Chinese output, and four concurrent requests.
The running A/E logs contained no ERROR, traceback, or quarantine messages.
A initialized eight Ascend Direct engines; each E initialized one.

The online benchmark used random input 128 / output 16 tokens, seed 0,
temperature 0, ignored EOS, and one warmup. Both profiles used eager execution
and the same model, topology, and batch settings. The gRPC baseline was
measured on the preceding day.

| Concurrency | Successful / failed | Ascend output tok/s | gRPC output tok/s | Ascend mean TTFT / TPOT |
| --- | --- | --- | --- | --- |
| 1 | 8 / 0 | 0.814 | 0.891 | 3.002 s / 1.111 s |
| 8 | 16 / 0 | 4.963 | 5.261 | 4.811 s / 1.395 s |

Real-model execution works with the new backend. This initial configuration
showed approximately 8.7% / 5.7% lower output throughput than the previous
gRPC runs. These small runs do not isolate transport time. Profile native
transfer, device synchronization, per-layer control, and expert computation
before selecting the next performance change. Physical HCCS routing,
cross-host RDMA, and numerical equivalence across all layers remain unverified.

During cleanup, Attention exited with code 0. All E containers exceeded the
30-second Docker stop grace and exited with code 137; `OOMKilled` was false.
The host subsequently showed no NPU processes. Graceful E shutdown remains
unverified: inspect frontend CloseSession delivery, the receiver's 30-second
session barrier, and native teardown before changing retirement semantics.

## Matched critical-path profiling (2026-09-26)

Both transports were rerun on the same host, source, images, model, topology,
and batch settings. The profile used four formal requests, one warmup, random
128-token input / 16-token output, and concurrency one. Independent Controller
databases avoided stale placement-generation state during the switch.
Counters were enabled only after model readiness. All formal requests passed.

| Single-token phase, mean wall time | gRPC | Ascend Direct |
| --- | --- | --- |
| A complete blocking MoE layer, including dummy calls | 18.301 ms | 22.722 ms |
| A input preparation, including executor wait | 3.015 ms | 1.462 ms |
| A output retrieval, including executor wait | 0.762 ms | 0.513 ms |
| A complete request to one E | 9.897 ms | 11.806 ms |
| E input preparation | 0.315 ms | 0.766 ms |
| E Torch expert submission | 2.044 ms | 2.483 ms |

Profiled mean TPOT was 984.02 ms for gRPC and 1154.52 ms for Ascend Direct.
These small runs include instrumentation overhead and are separate from the
unprofiled throughput results above. Phase samples overlap, span different
workers, and include dummy/warmup work; their means cannot be summed into an
exact useful-request latency decomposition. The Torch submission timer also
includes Python routing/indexing and asynchronous operator submission, rather
than only completed NPU kernels.

The new path improves A input/output preparation while adding explicit
native READ/WRITE completion, receive barriers, routing D2H validation, and
control-stream handling. Ascend native READ/WRITE bodies averaged 0.325 /
0.270 ms; their awaited interfaces averaged 0.756 / 0.577 ms. The difference
includes executor queueing, device-context setup, dispatch, and event-loop
wakeup. It does not isolate queueing alone. A uses one shared two-thread
native executor, whereas gRPC owns an executor per E connection; a separate
thread-count experiment is needed to measure this change's causal effect.
Receive-fence native bodies averaged 0.049 ms on A / 0.085 ms on E, so these
measurements do not support attributing the entire regression to device
synchronization alone.

Call-stack tags in the gRPC run identified 26,143 dummy calls among 29,368
single-token remote MoE layer calls (89.02%). The remaining 3,225 real decode
calls and 215 prefill calls match five requests, 15 decode steps, and 43 layers.
The common EK integration currently forwards vLLM DP dummy passes to remote
experts. The DP engine runs a dummy pass when its rank executes no ready
request while local or global work remains; it pauses when all ranks are idle.
An 18-second idle observation after the benchmark recorded zero remote calls.
This is a call-count ratio, not a measured wasted-compute percentage. The
Ascend run did not have stack tags; its similar counts and common integration
show the same risk without establishing an identical exact ratio.

The next changes should explicitly mark DP placeholder execution in the EK
integration and skip remote expert work after verifying collective requirements;
send small routing metadata with control messages to avoid D2H revalidation;
and reduce executor handoffs and staging copies with verified completion
contracts. Preserve required DP synchronization and memory-lifetime barriers.
All changes still need real-model correctness and performance checks.

Raw counters, benchmark logs, the opt-in profiler, and the actual vLLM/Ascend
source snapshots are retained on `.85` under
`/home/xxl/ek-npu-transport-build/model-results/phase-profile/`. The dedicated
profile model is stopped and the original Ascend Direct generated configs
are restored after the experiment. Physical HCCS routing and cross-host RDMA
remain unverified.
