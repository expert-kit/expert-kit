# Reproducing the Expert Kit Mooncake binding

Expert Kit keeps the source of truth for its transport backends in Git. A
fresh checkout contains the EK Python implementation, the pinned Mooncake
source manifest, the ordered patch series, and the scripts needed to build and
verify the native binding. It deliberately does **not** contain a prebuilt
Mooncake wheel.

## Which backends need Mooncake?

| EK backend | Mooncake build required? | Additional runtime requirement |
| --- | --- | --- |
| gRPC | No | The normal locked Python environment |
| SHM | No | Same-host shared memory; CUDA registration when GPU buffers are used |
| NCCL | No | A compatible PyTorch/CUDA/NCCL runtime |
| Transfer Engine `nvlink_intra` | Yes | The EK-patched Mooncake binding |
| Transfer Engine `rdma` | Yes | The EK-patched binding plus a compatible RDMA/CUDA environment |

The patches are reviewable source changes, not executable packages. Python
ultimately imports a compiled native extension, so every Transfer Engine
deployment must either build that extension locally or install an artifact
built from the same manifest inside the trusted environment.

## Start from a fresh checkout

Clone the transport branch explicitly; the repository default branch may not
contain these backends:

```bash
git clone --branch ek-transport-bakend --single-branch \
  <EK_REPOSITORY_URL> expert-kit
cd expert-kit
git status --short --branch
uv sync --locked
```

At this point gRPC, SHM, and NCCL can be configured without installing
Mooncake. The workspace lock currently selects Torch 2.11/CUDA 13 for those
backends; it is **not** a valid target environment for the CUDA 12.8 Transfer
Engine artifact. Continue below only when using a Transfer Engine backend.

The build is pinned by these version-controlled files:

```text
third_party/mooncake/build-manifest.toml
third_party/mooncake/build-constraints.txt
third_party/mooncake/build-requirements.lock
third_party/mooncake/patches/ek-731c4521.patch
third_party/mooncake/patches/ek-rdma-731c4521.patch
```

`build-manifest.toml` is authoritative for the upstream revision, ordered
patch hashes, submodule revisions, build flags, Python/platform constraints,
and required EK capabilities. `build-requirements.lock` pins the builder's
Python wheels by SHA256; `build-constraints.txt` applies the same versions to
upstream PEP 517 subprocesses. Do not update only one of these inputs.

## Prepare the pinned builder

The audited builder is intentionally narrow: Ubuntu 22.04 on x86-64 with
glibc 2.35, GCC/G++ 11.4.0, CMake 3.22.1, GNU Make 4.3, and CPython 3.12.13.
The following Ubuntu development packages must already be provisioned by the
system administrator:

```text
build-essential cmake git
libibverbs-dev libgoogle-glog-dev libgtest-dev libjsoncpp-dev
libunwind-dev libnuma-dev libpython3-dev libssl-dev libyaml-cpp-dev
libcurl4-openssl-dev pkg-config patchelf libc6-dev libc-bin
```

This is the dependency set for the CMake profile in the manifest, not
Mooncake's larger all-feature package list. The script checks the package
database, the selected CPython `Python.h`, and the required command versions
before creating the build workspace. It never installs system packages or
invokes a privilege-elevation tool.

CUDA must be the complete 12.8.61 toolkit. `/usr/local/cuda` must resolve to
that toolkit, the `nvcc` selected from `PATH` (and `CUDA_HOME`, when already
set) must resolve to the same toolkit, and its CUDA headers and `libcudart`
development libraries must be present. A different `nvcc` earlier in `PATH`
is rejected instead of being silently mixed into the build.

## Build locally from the pinned upstream source

Choose directories that are local to, or on trusted storage inside, the build
environment. Both directories must be outside the Expert Kit Git worktree and
must not contain one another. The example uses placeholders intentionally:

```bash
EK_BUILD_PYTHON=/usr/local/bin/python3.12
MOONCAKE_WORK_DIR=/path/to/local/mooncake-work
MOONCAKE_ARTIFACT_DIR=/path/to/local/mooncake-artifacts

./third_party/mooncake/build.sh \
  --source-url https://github.com/kvcache-ai/Mooncake.git \
  --work-dir "$MOONCAKE_WORK_DIR" \
  --artifact-dir "$MOONCAKE_ARTIFACT_DIR" \
  --python "$EK_BUILD_PYTHON" \
  --jobs 16
```

Without `--wheelhouse`, the script downloads only the wheel files accepted by
`build-requirements.lock`, checks their SHA256 values, and then takes all build
subprocesses offline. An air-gapped or centrally managed builder can instead
pass `--wheelhouse /path/to/trusted/python-wheelhouse`; every required wheel
must already be present and match the lock.

`--install` is optional and accepted only when `--python` belongs to a virtual
environment; the script will not mutate a system Python. The target virtual
environment must already contain the exact runtime recorded by the manifest:
CPython 3.12.13, `torch==2.10.0+cu128`, and `torch.version.cuda == "12.8"`,
plus the normal EK and Mooncake Python dependencies. These are the versions
recorded by the successful DeepSeek A/B experiment. The verifier rejects the
workspace's CUDA 13 Torch build instead of relying on loader order or a global
`LD_PRELOAD` workaround. `--source-url` accepts only a
credential-free `https://` or local `file://` URL. Git and recursive submodule
fetches are restricted to those same two protocols.

For an existing trusted Mooncake mirror or checkout, replace `--source-url`
with the mutually exclusive source-directory form:

```bash
MOONCAKE_SOURCE_DIR=/path/to/mooncake

./third_party/mooncake/build.sh \
  --source-dir "$MOONCAKE_SOURCE_DIR" \
  --work-dir "$MOONCAKE_WORK_DIR" \
  --artifact-dir "$MOONCAKE_ARTIFACT_DIR" \
  --python "$EK_BUILD_PYTHON" \
  --jobs 16
```

The script first verifies that its manifest, constraints, scripts, and patches
are tracked and byte-for-byte clean at the current Expert Kit commit. It then
checks out the manifest revision, verifies and applies the ordered patches,
and builds the pinned native package. When explicitly requested, `--install`
installs that exact output with `--no-deps` into the already prepared virtual
environment selected by `--python`, runs `pip check`, and performs a
Torch-first ABI verification.

Publication is a single versioned bundle under `--artifact-dir`:

```text
<artifact-dir>/
  mooncake_transfer_engine-<manifest-version>-<platform>.bundle/
    mooncake_transfer_engine-<manifest-version>-<platform>.whl
    mooncake_transfer_engine-<manifest-version>-<platform>.whl.receipt.json
    BUNDLE-COMPLETE.sha256
```

The bundle stays hidden in a same-filesystem staging directory until the wheel
and receipt have been verified, flushed, and covered by the completeness
marker. One atomic directory rename then makes the complete bundle visible.
An atomic directory lock serializes publishers of the same version across
hosts sharing the artifact store. If a terminated builder leaves a lock,
inspect the retained build workspace and hidden staging directory before an
administrator removes the stale lock.

The receipt records the Expert Kit commit, the clean-input result, and SHA256
hashes for `build.sh`, `verify.py`, the constraints, every patch, the manifest,
the combined patched diff, and the wheel. It contains no proxy configuration
or local filesystem paths.

Verify the installed binding before starting EK. The verifier checks the
manifest version, both EK capability sets, native dependencies, and the target
platform without constructing a Transfer Engine:

```bash
TE_PYTHON=/path/to/validated-te-venv/bin/python

"$TE_PYTHON" third_party/mooncake/verify.py \
  --manifest third_party/mooncake/build-manifest.toml \
  --require-torch \
  --json -
"$TE_PYTHON" scripts/verify-transport-backends.py \
  --backend te-rdma \
  --pretty
```

Use `--backend te-nvlink` in the static backend preflight for the
`nvlink_intra` profile. Verification is fail-closed: a stock public Mooncake
wheel does not provide the EK safety and backend-selection capabilities and
must not be substituted. The static checks do not initialize CUDA, NCCL,
network state, or a Mooncake Engine; the deployment smoke test must still
confirm that the configured runtime reports exactly the selected backend.

Run EK with `.venv/bin/python` or `uv run --no-sync`. A later unconstrained
environment sync cannot recreate a locally built native wheel by itself.

## Build once and share only inside the enterprise environment

When multiple hosts share a trusted artifact store, one controlled builder can
produce the native artifact once:

```bash
EK_BUILD_PYTHON=/usr/local/bin/python3.12
MOONCAKE_SOURCE_DIR=/path/to/trusted/mooncake-source
MOONCAKE_WORK_DIR=/path/to/trusted/build-work
MOONCAKE_ARTIFACT_DIR=/path/to/trusted/shared-artifacts

./third_party/mooncake/build.sh \
  --source-dir "$MOONCAKE_SOURCE_DIR" \
  --work-dir "$MOONCAKE_WORK_DIR" \
  --artifact-dir "$MOONCAKE_ARTIFACT_DIR" \
  --python "$EK_BUILD_PYTHON" \
  --jobs 16
```

Keep the entire atomically published bundle in that trusted store:

```text
mooncake_transfer_engine-<manifest-version>-<platform>.bundle/
```

On each consumer host, point `MOONCAKE_BUNDLE` at the exact bundle and
`MOONCAKE_WHEEL` at the wheel inside it. Check the completeness marker before
installing into the locked EK environment, then verify that installed package
and the wheel together. Keep the receipt in the same bundle for provenance
review:

```bash
TE_PYTHON=/path/to/validated-te-venv/bin/python
MOONCAKE_BUNDLE=/path/to/trusted/shared-artifacts/mooncake_transfer_engine-<manifest-version>-<platform>.bundle
MOONCAKE_WHEEL="$MOONCAKE_BUNDLE/mooncake_transfer_engine-<manifest-version>-<platform>.whl"

(cd "$MOONCAKE_BUNDLE" && sha256sum --check BUNDLE-COMPLETE.sha256)

uv pip install --python "$TE_PYTHON" --no-deps "$MOONCAKE_WHEEL"
"$TE_PYTHON" -m pip check
"$TE_PYTHON" third_party/mooncake/verify.py \
  --manifest third_party/mooncake/build-manifest.toml \
  --wheel "$MOONCAKE_WHEEL" \
  --require-torch \
  --json -
"$TE_PYTHON" scripts/verify-transport-backends.py \
  --backend te-rdma \
  --pretty
```

The artifact is intentionally not uploaded to a public source repository.
Wheel files, local artifact directories, and build receipts are ignored by
the repository. Never place proxy credentials, access tokens, internal hosts,
or environment-specific absolute paths in the manifest, scripts, receipt, or
Git history.

## Run a real data-path smoke

Static verification proves provenance and ABI prerequisites but deliberately
does not initialize a GPU or network. Before serving a model, run the
version-controlled two-process smoke and require element-for-element equality.
For cross-host RDMA, start the listener first:

```bash
"$TE_PYTHON" scripts/smoke-transfer-engine.py \
  --role listener \
  --backend rdma \
  --control-endpoint <listener-bind-host>:55000 \
  --segment-name <listener-rdma-host>:55001 \
  --device cuda:0 \
  --rdma-device <listener-rdma-device> \
  --run-id <shared-random-run-id>
```

Then start the initiator on the peer:

```bash
"$TE_PYTHON" scripts/smoke-transfer-engine.py \
  --role initiator \
  --backend rdma \
  --control-endpoint <listener-connect-host>:55000 \
  --segment-name <initiator-rdma-host>:55001 \
  --device cuda:0 \
  --rdma-device <initiator-rdma-device> \
  --run-id <shared-random-run-id>
```

The script initializes the real engine, requires the native selected backend
to be exactly `rdma`, transfers a CUDA tensor, performs the GPUDirect acquire,
and compares every element before the drain/invalidation/unregister handshake.
For same-host NVLink, select `nvlink_intra`, omit `--rdma-device`, use distinct
GPUs, and give both process segments unique ports. Always unset
`MC_USE_TENT`, `MC_USE_TEV1`, and `MC_FORCE_TCP`.

## Runtime safety boundaries

Always leave `MC_USE_TENT` and `MC_USE_TEV1` unset. EK supports one explicitly
forced Transfer Engine profile per runtime and rejects silent fallback.

Every patched binding must expose these common Boolean capabilities:

```text
EK_SAFE_TERMINAL_BATCH_SYNC
EK_HAS_GPUDIRECT_ACQUIRE
```

The forced `nvlink_intra` profile additionally requires:

```text
EK_INTRA_NVLINK_REGISTRATION_REFCOUNT
EK_FORCE_CONFIGURED_TRANSPORT
EK_DRAINED_NVLINK_INTRA_LOCAL_INVALIDATION
```

It must report `nvlink_intra` from `get_configured_backend()`, expose
`flush_gpudirect_writes()`, and provide
`invalidate_drained_nvlink_intra_segment(target_session)`.

The experimental forced `rdma` profile supports only `P2PHANDSHAKE` metadata
and additionally requires:

```text
EK_FORCE_CONFIGURED_RDMA_TRANSPORT
EK_DRAINED_RDMA_REMOTE_DESCRIPTOR_INVALIDATION
```

It must report exactly `rdma` and expose
`invalidate_drained_rdma_segment(target_session)`. Reusing an RDMA endpoint for
a new process generation requires restarting the Worker. Ambiguous transfer,
deregistration, or close outcomes remain fail-stop: quarantine the affected
arena and restart the affected EK processes rather than reusing an address or
rkey.

The native artifact is platform-specific. Match the Python ABI, architecture,
CUDA runtime, compiler ABI, and RDMA libraries recorded by the build manifest
and receipt. In particular, a dual RDMA/NVLink artifact can retain CUDA symbols
needed by the NVLink objects even while `rdma` is selected; the verifier must
pass in the same process environment used to launch EK.
