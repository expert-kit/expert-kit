#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
MANIFEST="${SCRIPT_DIR}/build-manifest.toml"
CONSTRAINTS="${SCRIPT_DIR}/build-constraints.txt"
BUILD_LOCK="${SCRIPT_DIR}/build-requirements.lock"
VERIFY_SCRIPT="${SCRIPT_DIR}/verify.py"

SOURCE_DIR_ARG=""
SOURCE_URL=""
WORK_DIR_ARG=""
ARTIFACT_DIR_ARG=""
PYTHON_ARG=""
WHEELHOUSE_ARG=""
JOBS=""
INSTALL=0

usage() {
  cat <<'EOF'
Build the EK-patched Mooncake Transfer Engine from audited source.

Usage:
  build.sh (--source-dir PATH | --source-url URL) \
    --work-dir PATH --artifact-dir PATH --python PYTHON \
    [--wheelhouse PATH] [--jobs N] [--install]

Required:
  --source-dir PATH    Existing Mooncake checkout with both submodules present.
  --source-url URL     Git URL for Mooncake; used instead of --source-dir.
  --work-dir PATH      New, non-existing directory for this build.
  --artifact-dir PATH  Existing or new private directory for verified artifacts.
  --python PYTHON      CPython executable recorded by build-manifest.toml.

Optional:
  --wheelhouse PATH    Existing directory containing every wheel in the hashed
                       build lock. Without it, the script downloads those exact
                       hashed wheels into --work-dir before going offline.
  --jobs N             Parallel build jobs (default: detected CPU count).
  --install            Install the verified wheel without resolving dependencies
                       into --python, which must already be a complete virtual
                       environment, then verify with Torch loaded first.
  -h, --help           Show this help.

The script accepts only credential-free https:// or file:// source URLs. It
never installs OS packages, invokes sudo, uploads artifacts, or deletes
--work-dir. Proxy and package-index settings are inherited from the calling
environment and are not written to the receipt.
EOF
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

need_value() {
  local option="$1"
  local value="${2-}"
  [[ -n "${value}" ]] || die "${option} requires a value"
  [[ "${value}" != --* ]] || die "${option} requires a value"
}

while (($#)); do
  case "$1" in
    --source-dir)
      need_value "$1" "${2-}"
      SOURCE_DIR_ARG="$2"
      shift 2
      ;;
    --source-url)
      need_value "$1" "${2-}"
      SOURCE_URL="$2"
      shift 2
      ;;
    --work-dir)
      need_value "$1" "${2-}"
      WORK_DIR_ARG="$2"
      shift 2
      ;;
    --artifact-dir)
      need_value "$1" "${2-}"
      ARTIFACT_DIR_ARG="$2"
      shift 2
      ;;
    --python)
      need_value "$1" "${2-}"
      PYTHON_ARG="$2"
      shift 2
      ;;
    --wheelhouse)
      need_value "$1" "${2-}"
      WHEELHOUSE_ARG="$2"
      shift 2
      ;;
    --jobs)
      need_value "$1" "${2-}"
      JOBS="$2"
      shift 2
      ;;
    --install)
      INSTALL=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

[[ -f "${MANIFEST}" ]] || die "missing manifest: ${MANIFEST}"
[[ -f "${CONSTRAINTS}" ]] || die "missing constraints: ${CONSTRAINTS}"
[[ -f "${BUILD_LOCK}" ]] || die "missing hashed build lock: ${BUILD_LOCK}"
[[ -f "${VERIFY_SCRIPT}" ]] || die "missing verifier: ${VERIFY_SCRIPT}"
[[ -n "${WORK_DIR_ARG}" ]] || die "--work-dir is required"
[[ -n "${ARTIFACT_DIR_ARG}" ]] || die "--artifact-dir is required"
[[ -n "${PYTHON_ARG}" ]] || die "--python is required"

if [[ -n "${SOURCE_DIR_ARG}" && -n "${SOURCE_URL}" ]]; then
  die "--source-dir and --source-url are mutually exclusive"
fi
if [[ -z "${SOURCE_DIR_ARG}" && -z "${SOURCE_URL}" ]]; then
  die "one of --source-dir or --source-url is required"
fi

if [[ "${PYTHON_ARG}" == */* ]]; then
  [[ -x "${PYTHON_ARG}" ]] || die "Python is not executable: ${PYTHON_ARG}"
  PYTHON="${PYTHON_ARG}"
else
  PYTHON="$(command -v -- "${PYTHON_ARG}" || true)"
  [[ -n "${PYTHON}" ]] || die "Python executable not found: ${PYTHON_ARG}"
fi
PYTHON="$(cd -- "$(dirname -- "${PYTHON}")" && pwd -P)/$(basename -- "${PYTHON}")"
"${PYTHON}" -c 'import tomllib' >/dev/null 2>&1 ||
  die "--python must provide the standard-library tomllib module"

abspath() {
  "${PYTHON}" - "$1" <<'PY'
import pathlib
import sys
print(pathlib.Path(sys.argv[1]).expanduser().resolve(strict=False))
PY
}

path_is_within() {
  "${PYTHON}" - "$1" "$2" <<'PY'
import pathlib
import sys

candidate = pathlib.Path(sys.argv[1]).resolve(strict=False)
parent = pathlib.Path(sys.argv[2]).resolve(strict=False)
try:
    candidate.relative_to(parent)
except ValueError:
    raise SystemExit(1)
PY
}

if ((INSTALL)); then
  IS_VIRTUAL_ENV="$(${PYTHON} - <<'PY'
import pathlib
import sys

prefix = pathlib.Path(sys.prefix).resolve()
base_prefix = pathlib.Path(getattr(sys, "base_prefix", sys.prefix)).resolve()
print("1" if prefix != base_prefix and (prefix / "pyvenv.cfg").is_file() else "0")
PY
  )"
  [[ "${IS_VIRTUAL_ENV}" == "1" ]] ||
    die "--install requires --python to belong to a virtual environment"
fi

if [[ -n "${SOURCE_URL}" ]]; then
  if ! "${PYTHON}" - "${SOURCE_URL}" <<'PY'
import sys
import urllib.parse

url = sys.argv[1]
if any(ord(character) < 32 or ord(character) == 127 for character in url):
    raise SystemExit("--source-url must not contain control characters")
parsed = urllib.parse.urlsplit(url)
if parsed.scheme not in {"https", "file"}:
    raise SystemExit("--source-url must use credential-free https:// or file://")
if parsed.username is not None or parsed.password is not None:
    raise SystemExit("--source-url must not contain credentials")
if parsed.query or parsed.fragment:
    raise SystemExit("--source-url must not contain a query or fragment")
if parsed.scheme == "https" and not parsed.hostname:
    raise SystemExit("an https:// source URL must contain a host")
if parsed.scheme == "file":
    if parsed.netloc not in {"", "localhost"}:
        raise SystemExit("a file:// source URL must be local")
    if not parsed.path.startswith("/"):
        raise SystemExit("a file:// source URL must contain an absolute path")
PY
  then
    die "invalid --source-url"
  fi
fi

# Limit Git and every recursively initialized submodule to the two audited
# source transports. In particular this blocks ssh, git, and ext helpers even
# when the caller has permissive global Git configuration.
export GIT_ALLOW_PROTOCOL="https:file"

manifest_scalar() {
  "${PYTHON}" - "${MANIFEST}" "$1" <<'PY'
import sys
import tomllib

with open(sys.argv[1], "rb") as stream:
    value = tomllib.load(stream)
for component in sys.argv[2].split("."):
    value = value[component]
print(value)
PY
}

manifest_array() {
  "${PYTHON}" - "${MANIFEST}" "$1" <<'PY'
import sys
import tomllib

with open(sys.argv[1], "rb") as stream:
    value = tomllib.load(stream)
for component in sys.argv[2].split("."):
    value = value[component]
for item in value:
    if "\n" in item:
        raise SystemExit("manifest array entries must not contain newlines")
    print(item)
PY
}

EXPECTED_PYTHON="$(manifest_scalar toolchain.python)"
ACTUAL_PYTHON="$(${PYTHON} -c 'import platform; print(platform.python_version())')"
[[ "${ACTUAL_PYTHON}" == "${EXPECTED_PYTHON}" ]] ||
  die "Python ${EXPECTED_PYTHON} is required; found ${ACTUAL_PYTHON} at ${PYTHON}"
check_python_implementation="$(${PYTHON} -c 'import platform; print(platform.python_implementation())')"
[[ "${check_python_implementation}" == "$(manifest_scalar toolchain.python_implementation)" ]] ||
  die "--python must be CPython"
export SOURCE_DATE_EPOCH="$(manifest_scalar upstream.source_date_epoch)"
export PYTHONHASHSEED=0
export TZ=UTC

if [[ -z "${JOBS}" ]]; then
  JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || true)"
  [[ "${JOBS}" =~ ^[1-9][0-9]*$ ]] || JOBS=1
fi
[[ "${JOBS}" =~ ^[1-9][0-9]*$ ]] || die "--jobs must be a positive integer"

for forbidden in $(manifest_array verification.forbidden_environment); do
  if [[ -n "${!forbidden+x}" ]]; then
    die "unsupported environment variable must be unset: ${forbidden}"
  fi
done

for command_name in \
  git cmake ctest make gcc g++ ldd getconf sha256sum awk sed grep find \
  patchelf pkg-config dpkg-query readelf basename dirname mkdir mktemp cp mv \
  rmdir rm ls head uname tail; do
  command -v -- "${command_name}" >/dev/null ||
    die "required command not found: ${command_name}"
done

check_exact() {
  local label="$1"
  local actual="$2"
  local expected="$3"
  [[ "${actual}" == "${expected}" ]] ||
    die "${label} must be ${expected}; found ${actual:-<empty>}"
}

readarray -t OS_RELEASE < <("${PYTHON}" - <<'PY'
values = {}
with open("/etc/os-release", encoding="utf-8") as stream:
    for line in stream:
        key, separator, value = line.rstrip().partition("=")
        if separator:
            values[key] = value.strip('"')
print(values.get("ID", ""))
print(values.get("VERSION_ID", ""))
PY
)
check_exact "OS" "${OS_RELEASE[0]-}" "$(manifest_scalar platform.os_id)"
check_exact "OS version" "${OS_RELEASE[1]-}" "$(manifest_scalar platform.os_version_id)"

# These are the Ubuntu development packages used by the pinned Transfer Engine
# profile. The script verifies them but deliberately never invokes apt or sudo.
REQUIRED_UBUNTU_PACKAGES=(
  build-essential
  cmake
  git
  libibverbs-dev
  libgoogle-glog-dev
  libgtest-dev
  libjsoncpp-dev
  libunwind-dev
  libnuma-dev
  libpython3-dev
  libssl-dev
  libyaml-cpp-dev
  libcurl4-openssl-dev
  pkg-config
  patchelf
  libc6-dev
  libc-bin
)
MISSING_UBUNTU_PACKAGES=()
for package_name in "${REQUIRED_UBUNTU_PACKAGES[@]}"; do
  package_status="$(
    dpkg-query --show --showformat='${Status}' "${package_name}" 2>/dev/null || true
  )"
  [[ "${package_status}" == "install ok installed" ]] ||
    MISSING_UBUNTU_PACKAGES+=("${package_name}")
done
if ((${#MISSING_UBUNTU_PACKAGES[@]})); then
  printf 'error: missing required Ubuntu development packages:\n' >&2
  printf '  %s\n' "${MISSING_UBUNTU_PACKAGES[@]}" >&2
  die "ask the system administrator to provision the packages listed above"
fi

check_exact "architecture" "$(uname -m)" "$(manifest_scalar platform.architecture)"
check_exact "glibc" "$(getconf GNU_LIBC_VERSION | awk '{print $2}')" \
  "$(manifest_scalar platform.glibc)"
check_exact "GCC" "$(gcc -dumpfullversion -dumpversion)" \
  "$(manifest_scalar toolchain.gcc)"
check_exact "G++" "$(g++ -dumpfullversion -dumpversion)" \
  "$(manifest_scalar toolchain.gcc)"
check_exact "CMake" "$(cmake --version | sed -n '1s/^cmake version //p')" \
  "$(manifest_scalar toolchain.cmake)"
check_exact "GNU Make" "$(make --version | sed -n '1s/^GNU Make //p')" \
  "$(manifest_scalar toolchain.make)"

PYTHON_INCLUDE_DIR="$(${PYTHON} - <<'PY'
import sysconfig
print(sysconfig.get_path("include"))
PY
)"
[[ -f "${PYTHON_INCLUDE_DIR}/Python.h" ]] ||
  die "CPython development header not found: ${PYTHON_INCLUDE_DIR}/Python.h"

CUDA_LINK="/usr/local/cuda"
[[ -d "${CUDA_LINK}" ]] ||
  die "the pinned CUDA toolkit link is missing: ${CUDA_LINK}"
CUDA_TOOLKIT_ROOT="$(abspath "${CUDA_LINK}")"
CUDA_LINK_NVCC="${CUDA_LINK}/bin/nvcc"
[[ -x "${CUDA_LINK_NVCC}" ]] ||
  die "the pinned CUDA toolkit has no executable nvcc: ${CUDA_LINK_NVCC}"
NVCC="$(command -v nvcc || true)"
[[ -n "${NVCC}" ]] || NVCC="${CUDA_LINK_NVCC}"
NVCC="$(abspath "${NVCC}")"
check_exact "nvcc toolkit" "${NVCC}" "$(abspath "${CUDA_LINK_NVCC}")"
if [[ -n "${CUDA_HOME-}" ]]; then
  check_exact "CUDA_HOME toolkit" "$(abspath "${CUDA_HOME}")" \
    "${CUDA_TOOLKIT_ROOT}"
fi
export CUDA_HOME="${CUDA_LINK}"
export CUDACXX="${NVCC}"

for cuda_input in \
  include/cuda.h \
  include/cuda_runtime.h \
  include/cuda_runtime_api.h \
  lib64/libcudart.so \
  lib64/libcudart.so.12 \
  lib64/libcudart_static.a; do
  [[ -e "${CUDA_TOOLKIT_ROOT}/${cuda_input}" ]] ||
    die "the pinned CUDA toolkit is incomplete: ${CUDA_TOOLKIT_ROOT}/${cuda_input}"
done
CUDA_VERSION="$(${NVCC} --version | sed -n 's/.*V\([0-9][0-9.]*\).*/\1/p' | tail -n 1)"
check_exact "CUDA compiler" "${CUDA_VERSION}" "$(manifest_scalar toolchain.cuda)"

EK_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null || true)"
[[ -n "${EK_ROOT}" ]] ||
  die "build assets must be run from a committed Expert Kit Git worktree"
EK_ROOT="$(abspath "${EK_ROOT}")"
path_is_within "${SCRIPT_DIR}" "${EK_ROOT}" ||
  die "build asset directory is outside the Expert Kit Git worktree"
EK_COMMIT="$(git -C "${EK_ROOT}" rev-parse --verify HEAD)"
[[ "${EK_COMMIT}" =~ ^[0-9a-f]{40}$ ]] ||
  die "cannot resolve the Expert Kit commit"

readarray -t PATCH_RELATIVE_PATHS < <("${PYTHON}" - "${MANIFEST}" <<'PY'
import sys
import tomllib

with open(sys.argv[1], "rb") as stream:
    manifest = tomllib.load(stream)
for patch in manifest["patches"]["files"]:
    print(patch["path"])
PY
)
INPUT_ASSET_PATHS=(
  "${MANIFEST}"
  "${CONSTRAINTS}"
  "${BUILD_LOCK}"
  "${SCRIPT_DIR}/build.sh"
  "${VERIFY_SCRIPT}"
)
for patch_relative in "${PATCH_RELATIVE_PATHS[@]}"; do
  INPUT_ASSET_PATHS+=("${SCRIPT_DIR}/${patch_relative}")
done

verify_input_assets_clean() {
  local input_asset input_asset_relative committed_blob working_blob
  check_exact "Expert Kit commit" \
    "$(git -C "${EK_ROOT}" rev-parse --verify HEAD)" "${EK_COMMIT}"
  for input_asset in "${INPUT_ASSET_PATHS[@]}"; do
    input_asset="$(abspath "${input_asset}")"
    [[ -f "${input_asset}" && ! -L "${input_asset}" ]] ||
      die "build input asset must be a regular non-symlink file: ${input_asset}"
    path_is_within "${input_asset}" "${EK_ROOT}" ||
      die "build input asset is outside the Expert Kit worktree: ${input_asset}"
    input_asset_relative="$(${PYTHON} - "${input_asset}" "${EK_ROOT}" <<'PY'
import pathlib
import sys
print(pathlib.Path(sys.argv[1]).relative_to(pathlib.Path(sys.argv[2])).as_posix())
PY
    )"
    git -C "${EK_ROOT}" ls-files --error-unmatch -- "${input_asset_relative}" \
      >/dev/null 2>&1 ||
      die "build input asset is not tracked by Git: ${input_asset_relative}"
    committed_blob="$(git -C "${EK_ROOT}" rev-parse "HEAD:${input_asset_relative}")"
    working_blob="$(git hash-object --no-filters -- "${input_asset}")"
    check_exact "committed build input ${input_asset_relative}" \
      "${working_blob}" "${committed_blob}"
  done
}
verify_input_assets_clean
INPUT_ASSETS_CLEAN=1

WORK_DIR="$(abspath "${WORK_DIR_ARG}")"
ARTIFACT_DIR="$(abspath "${ARTIFACT_DIR_ARG}")"
[[ "${WORK_DIR}" != "/" ]] || die "--work-dir cannot be /"
[[ "${ARTIFACT_DIR}" != "/" ]] || die "--artifact-dir cannot be /"
[[ "${WORK_DIR}" != "${ARTIFACT_DIR}" ]] ||
  die "--work-dir and --artifact-dir must be different"
if path_is_within "${WORK_DIR}" "${EK_ROOT}"; then
  die "--work-dir must be outside the Expert Kit Git worktree"
fi
if path_is_within "${ARTIFACT_DIR}" "${EK_ROOT}"; then
  die "--artifact-dir must be outside the Expert Kit Git worktree"
fi
if path_is_within "${WORK_DIR}" "${ARTIFACT_DIR}" ||
  path_is_within "${ARTIFACT_DIR}" "${WORK_DIR}"; then
  die "--work-dir and --artifact-dir must not contain one another"
fi
[[ ! -e "${WORK_DIR}" ]] ||
  die "--work-dir must not already exist: ${WORK_DIR}"
mkdir -p -- "$(dirname -- "${WORK_DIR}")" "${ARTIFACT_DIR}"
mkdir -- "${WORK_DIR}"

SOURCE_DIR=""
if [[ -n "${SOURCE_DIR_ARG}" ]]; then
  SOURCE_DIR="$(abspath "${SOURCE_DIR_ARG}")"
  [[ -d "${SOURCE_DIR}" ]] || die "--source-dir is not a directory: ${SOURCE_DIR}"
  git -C "${SOURCE_DIR}" rev-parse --is-inside-work-tree >/dev/null 2>&1 ||
    die "--source-dir is not a Git worktree: ${SOURCE_DIR}"
else
  : # URL syntax and protocol were validated before any Git operation.
fi

SOURCE_TREE="${WORK_DIR}/source"
YLT_BUILD_DIR="${WORK_DIR}/yalantinglibs-build"
PREFIX_DIR="${WORK_DIR}/prefix"
CMAKE_BUILD_DIR="${WORK_DIR}/cmake-build"
BUILD_VENV="${WORK_DIR}/build-venv"
if [[ -n "${WHEELHOUSE_ARG}" ]]; then
  PYTHON_WHEELHOUSE="$(abspath "${WHEELHOUSE_ARG}")"
  [[ -d "${PYTHON_WHEELHOUSE}" ]] ||
    die "--wheelhouse is not a directory: ${PYTHON_WHEELHOUSE}"
else
  PYTHON_WHEELHOUSE="${WORK_DIR}/python-wheelhouse"
fi
VERIFICATION_JSON="${WORK_DIR}/wheel-verification.json"
TARGET_VERIFICATION_JSON="${WORK_DIR}/target-verification.json"

BASE_COMMIT="$(manifest_scalar upstream.commit)"
if [[ -n "${SOURCE_DIR}" ]]; then
  git -C "${SOURCE_DIR}" cat-file -e "${BASE_COMMIT}^{commit}" 2>/dev/null ||
    die "--source-dir does not contain audited commit ${BASE_COMMIT}"
  git clone --no-hardlinks --no-checkout -- "${SOURCE_DIR}" "${SOURCE_TREE}"
else
  # Fetch only the audited commit instead of transferring the complete
  # upstream history. The explicit object ID remains the trust anchor, and the
  # checkout below still verifies it before any patch is applied.
  git init --quiet "${SOURCE_TREE}"
  git -C "${SOURCE_TREE}" remote add origin "${SOURCE_URL}"
  git -C "${SOURCE_TREE}" fetch --depth 1 --no-tags origin "${BASE_COMMIT}"
fi
git -C "${SOURCE_TREE}" checkout --detach "${BASE_COMMIT}"
check_exact "Mooncake commit" \
  "$(git -C "${SOURCE_TREE}" rev-parse HEAD)" "${BASE_COMMIT}"

git -C "${SOURCE_TREE}" submodule sync --recursive
if [[ -n "${SOURCE_DIR}" ]]; then
  readarray -t SUBMODULE_ROWS < <("${PYTHON}" - "${MANIFEST}" <<'PY'
import sys
import tomllib

with open(sys.argv[1], "rb") as stream:
    manifest = tomllib.load(stream)
for name, item in manifest["submodules"].items():
    print(f"{name}\t{item['path']}\t{item['commit']}")
PY
  )
  for row in "${SUBMODULE_ROWS[@]}"; do
    IFS=$'\t' read -r submodule_name submodule_path submodule_commit <<<"${row}"
    local_submodule="${SOURCE_DIR}/${submodule_path}"
    git -C "${local_submodule}" cat-file -e "${submodule_commit}^{commit}" 2>/dev/null ||
      die "source submodule is missing ${submodule_commit}: ${local_submodule}"
    # Persist the override only in the isolated clone. A command-scoped `-c`
    # URL is enough to copy objects but leaves the submodule marked inactive,
    # which would make the status check below fail despite a correct checkout.
    git -C "${SOURCE_TREE}" config \
      "submodule.${submodule_path}.url" "${local_submodule}"
    git -C "${SOURCE_TREE}" config \
      "submodule.${submodule_path}.active" true
  done
  git -C "${SOURCE_TREE}" -c protocol.file.allow=always \
    submodule update --init --recursive
else
  git -C "${SOURCE_TREE}" submodule update --init --recursive
fi

SUBMODULE_STATUS="$(git -C "${SOURCE_TREE}" submodule status --recursive)"
if grep -Eq '^[+-U]' <<<"${SUBMODULE_STATUS}"; then
  die "one or more Mooncake submodules are missing or at the wrong commit"
fi

while IFS=$'\t' read -r submodule_name submodule_path submodule_commit; do
  actual_submodule_commit="$(git -C "${SOURCE_TREE}/${submodule_path}" rev-parse HEAD)"
  check_exact "submodule ${submodule_name}" \
    "${actual_submodule_commit}" "${submodule_commit}"
done < <("${PYTHON}" - "${MANIFEST}" <<'PY'
import sys
import tomllib

with open(sys.argv[1], "rb") as stream:
    manifest = tomllib.load(stream)
for name, item in manifest["submodules"].items():
    print(f"{name}\t{item['path']}\t{item['commit']}")
PY
)

if [[ -n "$(git -C "${SOURCE_TREE}" status --porcelain=v1 --untracked-files=all)" ]]; then
  die "audited source checkout is not clean before patching"
fi

while IFS=$'\t' read -r patch_relative patch_sha256; do
  patch_path="${SCRIPT_DIR}/${patch_relative}"
  [[ -f "${patch_path}" ]] || die "missing patch: ${patch_path}"
  check_exact "patch SHA256 (${patch_relative})" \
    "$(sha256sum "${patch_path}" | awk '{print $1}')" "${patch_sha256}"
  git -C "${SOURCE_TREE}" apply --check -- "${patch_path}"
  git -C "${SOURCE_TREE}" apply -- "${patch_path}"
done < <("${PYTHON}" - "${MANIFEST}" <<'PY'
import sys
import tomllib

with open(sys.argv[1], "rb") as stream:
    manifest = tomllib.load(stream)
for patch in manifest["patches"]["files"]:
    print(f"{patch['path']}\t{patch['sha256']}")
PY
)

git -C "${SOURCE_TREE}" diff --check
[[ -z "$(git -C "${SOURCE_TREE}" ls-files --others --exclude-standard)" ]] ||
  die "patch application unexpectedly created untracked files"
PATCHED_DIFF_SHA256="$({
  git -C "${SOURCE_TREE}" -c core.abbrev=40 \
    diff --binary --full-index "${BASE_COMMIT}" -- .
} | sha256sum | awk '{print $1}')"
check_exact "combined patched diff SHA256" "${PATCHED_DIFF_SHA256}" \
  "$(manifest_scalar patches.combined_diff_sha256)"

PATCHED_VERSION="$(${PYTHON} - "${SOURCE_TREE}/mooncake-wheel/pyproject.toml" <<'PY'
import sys
import tomllib
with open(sys.argv[1], "rb") as stream:
    print(tomllib.load(stream)["project"]["version"])
PY
)"
check_exact "patched wheel version" "${PATCHED_VERSION}" \
  "$(manifest_scalar artifact.version)"

"${PYTHON}" -m venv "${BUILD_VENV}"
BUILD_PYTHON="${BUILD_VENV}/bin/python"
if [[ -z "${WHEELHOUSE_ARG}" ]]; then
  mkdir -- "${PYTHON_WHEELHOUSE}"
  "${BUILD_PYTHON}" -m pip download \
    --require-hashes \
    --only-binary=:all: \
    --requirement "${BUILD_LOCK}" \
    --dest "${PYTHON_WHEELHOUSE}"
fi
export PIP_CONSTRAINT="${CONSTRAINTS}"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_INPUT=1
export PIP_NO_INDEX=1
export PIP_FIND_LINKS="${PYTHON_WHEELHOUSE}"
"${BUILD_PYTHON}" -m pip install \
  --no-index \
  --find-links "${PYTHON_WHEELHOUSE}" \
  --require-hashes \
  --requirement "${BUILD_LOCK}"
"${BUILD_PYTHON}" -m pip check

YLT_GENERATOR="$(manifest_scalar yalantinglibs.generator)"
YLT_BUILD_TYPE="$(manifest_scalar yalantinglibs.build_type)"
readarray -t YLT_DEFINITIONS < <(manifest_array yalantinglibs.cmake_definitions)
YLT_CMAKE_ARGS=(
  -S "${SOURCE_TREE}/extern/yalantinglibs"
  -B "${YLT_BUILD_DIR}"
  -G "${YLT_GENERATOR}"
  "-DCMAKE_BUILD_TYPE=${YLT_BUILD_TYPE}"
  "-DCMAKE_INSTALL_PREFIX=${PREFIX_DIR}"
  "-DCMAKE_CXX_COMPILER=$(command -v g++)"
)
for definition in "${YLT_DEFINITIONS[@]}"; do
  YLT_CMAKE_ARGS+=("-D${definition}")
done
cmake "${YLT_CMAKE_ARGS[@]}"
cmake --build "${YLT_BUILD_DIR}" --parallel "${JOBS}"
cmake --install "${YLT_BUILD_DIR}"

GENERATOR="$(manifest_scalar build.generator)"
BUILD_TYPE="$(manifest_scalar build.build_type)"
readarray -t CMAKE_DEFINITIONS < <(manifest_array build.cmake_definitions)
CMAKE_ARGS=(
  -S "${SOURCE_TREE}"
  -B "${CMAKE_BUILD_DIR}"
  -G "${GENERATOR}"
  "-DCMAKE_BUILD_TYPE=${BUILD_TYPE}"
  "-DCMAKE_PREFIX_PATH=${PREFIX_DIR}"
  "-DCMAKE_CXX_FLAGS=-I${PREFIX_DIR}/include"
  "-DCMAKE_C_COMPILER=$(command -v gcc)"
  "-DCMAKE_CXX_COMPILER=$(command -v g++)"
  "-DCMAKE_CUDA_COMPILER=${NVCC}"
  "-DCUDAToolkit_ROOT=${CUDA_LINK}"
  "-DPYTHON_EXECUTABLE=${BUILD_PYTHON}"
  "-DPython3_EXECUTABLE=${BUILD_PYTHON}"
)
for definition in "${CMAKE_DEFINITIONS[@]}"; do
  CMAKE_ARGS+=("-D${definition}")
done
PATH="${BUILD_VENV}/bin:${PATH}" cmake "${CMAKE_ARGS[@]}"

CMAKE_CACHE="${CMAKE_BUILD_DIR}/CMakeCache.txt"
[[ -f "${CMAKE_CACHE}" ]] || die "CMake did not produce ${CMAKE_CACHE}"
for definition in "CMAKE_BUILD_TYPE=${BUILD_TYPE}" "${CMAKE_DEFINITIONS[@]}"; do
  definition_name="${definition%%=*}"
  definition_value="${definition#*=}"
  grep -Eq "^${definition_name}(:[^=]*)?=${definition_value}$" "${CMAKE_CACHE}" ||
    die "CMake cache does not contain audited setting ${definition}"
done

CONFIGURED_NVCC="$(sed -n 's/^CMAKE_CUDA_COMPILER:FILEPATH=//p' "${CMAKE_CACHE}")"
[[ -x "${CONFIGURED_NVCC}" ]] || die "CMake did not configure an executable CUDA compiler"
check_exact "configured nvcc toolkit" "$(abspath "${CONFIGURED_NVCC}")" \
  "$(abspath "${CUDA_LINK_NVCC}")"
CONFIGURED_CUDA_VERSION="$(${CONFIGURED_NVCC} --version | sed -n 's/.*V\([0-9][0-9.]*\).*/\1/p' | tail -n 1)"
check_exact "configured CUDA compiler" "${CONFIGURED_CUDA_VERSION}" \
  "$(manifest_scalar toolchain.cuda)"

readarray -t IBVERBS_CACHE_LINES < <(
  grep -E '^IBVERBS_[A-Z0-9_]*(:[^=]*)?=' "${CMAKE_CACHE}" || true
)
((${#IBVERBS_CACHE_LINES[@]} > 0)) ||
  die "CMake did not record any IBVERBS discovery result"
if printf '%s\n' "${IBVERBS_CACHE_LINES[@]}" | grep -q 'NOTFOUND'; then
  die "CMake found an unusable IBVERBS path"
fi

readarray -t ENGINE_FLAG_FILES < <(
  find "${CMAKE_BUILD_DIR}/mooncake-integration" \
    -type f -path '*/engine.dir/flags.make' -print
)
((${#ENGINE_FLAG_FILES[@]} == 1)) ||
  die "expected exactly one generated engine flags.make; found ${#ENGINE_FLAG_FILES[@]}"
grep -Eq '(^|[^A-Z0-9_])USE_RDMA([^A-Z0-9_]|$)' "${ENGINE_FLAG_FILES[0]}" ||
  die "engine compile flags do not contain USE_RDMA"

cmake --build "${CMAKE_BUILD_DIR}" --parallel "${JOBS}"
CTEST_LIST="$(ctest --test-dir "${CMAKE_BUILD_DIR}" -N -R '^transport_uint_test$')"
grep -Eq 'Test[[:space:]]+#[0-9]+:[[:space:]]+transport_uint_test' <<<"${CTEST_LIST}" ||
  die "CMake build does not register transport_uint_test"
ctest --test-dir "${CMAKE_BUILD_DIR}" \
  --output-on-failure \
  -R '^transport_uint_test$'

WHEEL_SCRIPT_RELATIVE="$(manifest_scalar build.upstream_wheel_script)"
WHEEL_SCRIPT="${SOURCE_TREE}/${WHEEL_SCRIPT_RELATIVE}"
[[ -f "${WHEEL_SCRIPT}" ]] || die "missing upstream wheel script: ${WHEEL_SCRIPT}"
WHEEL_PYTHON_VERSION="$(manifest_scalar build.upstream_wheel_python)"
WHEEL_STAGING_NAME="$(manifest_scalar build.wheel_staging_name)"
[[ "${WHEEL_STAGING_NAME}" =~ ^\.[A-Za-z0-9][A-Za-z0-9._-]*$ ]] ||
  die "manifest wheel_staging_name must be a controlled relative basename"
RELATIVE_BUILD_DIR="$(${PYTHON} - "${SOURCE_TREE}" "${CMAKE_BUILD_DIR}" <<'PY'
import os
import sys
relative = os.path.relpath(sys.argv[2], sys.argv[1])
if os.path.isabs(relative):
    raise SystemExit("build directory must be relative to source")
print(relative)
PY
)"

(
  cd -- "${SOURCE_TREE}"
  unset CI FREE_BUILD_DIR
  unset NON_CUDA_BUILD CU13_BUILD NPU_BUILD EFA_BUILD EFA_NON_CUDA_BUILD MUSA_BUILD
  export PATH="${BUILD_VENV}/bin:${PATH}"
  export BUILD_DIR="${RELATIVE_BUILD_DIR}"
  export PLATFORM_TAG="$(manifest_scalar artifact.platform_tag)"
  export PYTHON_VERSION="${WHEEL_PYTHON_VERSION}"
  export OUTPUT_DIR="${WHEEL_STAGING_NAME}"
  bash "${WHEEL_SCRIPT_RELATIVE}" "${WHEEL_PYTHON_VERSION}" "${WHEEL_STAGING_NAME}"
)

UPSTREAM_WHEEL_DIR="${SOURCE_TREE}/mooncake-wheel/${WHEEL_STAGING_NAME}"
readarray -t BUILT_WHEELS < <(find "${UPSTREAM_WHEEL_DIR}" -maxdepth 1 -type f -name '*.whl' -print)
((${#BUILT_WHEELS[@]} == 1)) ||
  die "expected exactly one upstream wheel; found ${#BUILT_WHEELS[@]}"
EXPECTED_WHEEL_FILENAME="$(manifest_scalar artifact.wheel_filename)"
check_exact "wheel filename" "$(basename -- "${BUILT_WHEELS[0]}")" \
  "${EXPECTED_WHEEL_FILENAME}"

# Recheck the live verifier and provenance inputs before they are used to
# approve and publish the native artifact.
verify_input_assets_clean
BUNDLE_NAME="${EXPECTED_WHEEL_FILENAME%.whl}.bundle"
FINAL_BUNDLE="${ARTIFACT_DIR}/${BUNDLE_NAME}"
PUBLISH_LOCK="${ARTIFACT_DIR}/.${BUNDLE_NAME}.publish.lock"
[[ ! -e "${FINAL_BUNDLE}" ]] ||
  die "refusing to overwrite existing verified bundle: ${FINAL_BUNDLE}"
if ! mkdir -- "${PUBLISH_LOCK}"; then
  die "another host is publishing this version, or a stale lock needs review: ${PUBLISH_LOCK}"
fi
PUBLISH_LOCK_HELD=1
release_publish_lock() {
  if [[ "${PUBLISH_LOCK_HELD:-0}" == "1" ]]; then
    rmdir -- "${PUBLISH_LOCK}" 2>/dev/null || true
    PUBLISH_LOCK_HELD=0
  fi
}
trap release_publish_lock EXIT

# mktemp creates the staging directory under the final parent, guaranteeing
# that the final directory rename stays on one filesystem.
PUBLISH_STAGE="$(mktemp -d "${ARTIFACT_DIR}/.${BUNDLE_NAME}.stage.XXXXXX")"
STAGED_WHEEL="${PUBLISH_STAGE}/${EXPECTED_WHEEL_FILENAME}"
STAGED_RECEIPT="${PUBLISH_STAGE}/${EXPECTED_WHEEL_FILENAME}.receipt.json"
STAGED_COMPLETE="${PUBLISH_STAGE}/BUNDLE-COMPLETE.sha256"
cp -- "${BUILT_WHEELS[0]}" "${STAGED_WHEEL}"

"${BUILD_PYTHON}" -m pip install --no-deps --force-reinstall "${STAGED_WHEEL}"
"${BUILD_PYTHON}" "${VERIFY_SCRIPT}" \
  --manifest "${MANIFEST}" \
  --wheel "${STAGED_WHEEL}" \
  --json "${VERIFICATION_JSON}"

if ((INSTALL)); then
  "${PYTHON}" -m pip install \
    --no-deps \
    --force-reinstall \
    "${STAGED_WHEEL}"
  "${PYTHON}" -m pip check
  "${PYTHON}" "${VERIFY_SCRIPT}" \
    --manifest "${MANIFEST}" \
    --require-torch \
    --json "${TARGET_VERIFICATION_JSON}"
fi

MANIFEST_SHA256="$(sha256sum "${MANIFEST}" | awk '{print $1}')"
CONSTRAINTS_SHA256="$(sha256sum "${CONSTRAINTS}" | awk '{print $1}')"
BUILD_LOCK_SHA256="$(sha256sum "${BUILD_LOCK}" | awk '{print $1}')"
BUILD_SCRIPT_SHA256="$(sha256sum "${SCRIPT_DIR}/build.sh" | awk '{print $1}')"
VERIFY_SCRIPT_SHA256="$(sha256sum "${VERIFY_SCRIPT}" | awk '{print $1}')"
WHEEL_SHA256="$(sha256sum "${STAGED_WHEEL}" | awk '{print $1}')"
"${BUILD_PYTHON}" - \
  "${MANIFEST}" "${VERIFICATION_JSON}" "${TARGET_VERIFICATION_JSON}" \
  "${STAGED_WHEEL}" "${STAGED_RECEIPT}" "${MANIFEST_SHA256}" \
  "${CONSTRAINTS_SHA256}" "${BUILD_LOCK_SHA256}" "${BUILD_SCRIPT_SHA256}" \
  "${VERIFY_SCRIPT_SHA256}" "${WHEEL_SHA256}" "${PATCHED_DIFF_SHA256}" \
  "${INSTALL}" "${EK_ROOT}" "${EK_COMMIT}" "${INPUT_ASSETS_CLEAN}" <<'PY'
import datetime
import hashlib
import json
import pathlib
import sys
import tomllib

(
    manifest_path,
    verification_path,
    target_verification_path,
    wheel_path,
    receipt_path,
    manifest_sha256,
    constraints_sha256,
    build_lock_sha256,
    build_script_sha256,
    verify_script_sha256,
    wheel_sha256,
    patched_diff_sha256,
    installed,
    ek_root_path,
    ek_commit,
    input_assets_clean,
) = sys.argv[1:]

manifest_path = pathlib.Path(manifest_path).resolve()
script_dir = manifest_path.parent
ek_root = pathlib.Path(ek_root_path).resolve()

with open(manifest_path, "rb") as stream:
    manifest = tomllib.load(stream)
with open(verification_path, encoding="utf-8") as stream:
    verification = json.load(stream)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


asset_paths = [
    manifest_path,
    script_dir / "build-constraints.txt",
    script_dir / "build-requirements.lock",
    script_dir / "build.sh",
    script_dir / "verify.py",
]
asset_paths.extend(script_dir / item["path"] for item in manifest["patches"]["files"])
input_assets = [
    {
        "path": path.resolve().relative_to(ek_root).as_posix(),
        "sha256": sha256(path),
    }
    for path in asset_paths
]

actual_patches = [
    {
        "path": item["path"],
        "sha256": sha256(script_dir / item["path"]),
    }
    for item in manifest["patches"]["files"]
]

def compact(result):
    return {
        "ok": result["ok"],
        "distribution_version": result.get("distribution", {}).get("version"),
        "capabilities": result.get("capabilities", {}),
        "engine_methods": result.get("engine_methods", {}),
        "ldd_missing": result.get("ldd", {}).get("missing", []),
        "auditwheel_ok": result.get("wheel", {}).get("auditwheel", {}).get("ok"),
    }

receipt = {
    "schema_version": 2,
    "status": "verified",
    "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "expert_kit": {
        "commit": ek_commit,
        "input_assets_clean_against_commit": input_assets_clean == "1",
    },
    "input_assets": input_assets,
    "source": {
        "repository": manifest["upstream"]["repository"],
        "commit": manifest["upstream"]["commit"],
        "source_date_epoch": manifest["upstream"]["source_date_epoch"],
    },
    "submodules": manifest["submodules"],
    "patches": {
        "files": actual_patches,
        "combined_diff_sha256": patched_diff_sha256,
    },
    "build": {
        "manifest_sha256": manifest_sha256,
        "constraints_sha256": constraints_sha256,
        "build_lock_sha256": build_lock_sha256,
        "build_script_sha256": build_script_sha256,
        "verify_script_sha256": verify_script_sha256,
        "platform": manifest["platform"],
        "toolchain": manifest["toolchain"],
        "generator": manifest["build"]["generator"],
        "build_type": manifest["build"]["build_type"],
        "cmake_definitions": manifest["build"]["cmake_definitions"],
    },
    "artifact": {
        "distribution": manifest["artifact"]["distribution"],
        "version": manifest["artifact"]["version"],
        "filename": pathlib.Path(wheel_path).name,
        "size": pathlib.Path(wheel_path).stat().st_size,
        "sha256": wheel_sha256,
    },
    "verification": compact(verification),
    "installed_into_requested_python": installed == "1",
}

if installed == "1":
    with open(target_verification_path, encoding="utf-8") as stream:
        receipt["target_verification"] = compact(json.load(stream))

with open(receipt_path, "w", encoding="utf-8") as stream:
    json.dump(receipt, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY

RECEIPT_SHA256="$(sha256sum "${STAGED_RECEIPT}" | awk '{print $1}')"
"${BUILD_PYTHON}" - \
  "${STAGED_COMPLETE}" "${EXPECTED_WHEEL_FILENAME}" "${WHEEL_SHA256}" \
  "$(basename -- "${STAGED_RECEIPT}")" "${RECEIPT_SHA256}" <<'PY'
import pathlib
import sys

marker_path, wheel_name, wheel_sha256, receipt_name, receipt_sha256 = sys.argv[1:]
pathlib.Path(marker_path).write_text(
    f"{wheel_sha256}  {wheel_name}\n{receipt_sha256}  {receipt_name}\n",
    encoding="utf-8",
)
PY

# Flush both payload files and the completeness marker before the atomic
# directory rename exposes the bundle to consumers on shared storage.
"${BUILD_PYTHON}" - "${PUBLISH_STAGE}" <<'PY'
import os
import errno
import pathlib
import sys

stage = pathlib.Path(sys.argv[1])
for path in stage.iterdir():
    if path.is_file():
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
directory_fd = os.open(stage, os.O_RDONLY | os.O_DIRECTORY)
try:
    try:
        os.fsync(directory_fd)
    except OSError as error:
        if error.errno not in {errno.EINVAL, errno.ENOTSUP}:
            raise
finally:
    os.close(directory_fd)
PY

verify_input_assets_clean
[[ ! -e "${FINAL_BUNDLE}" ]] ||
  die "verified bundle appeared while the publish lock was held: ${FINAL_BUNDLE}"
mv -T -- "${PUBLISH_STAGE}" "${FINAL_BUNDLE}"
"${BUILD_PYTHON}" - "${ARTIFACT_DIR}" <<'PY'
import errno
import os
import sys

directory_fd = os.open(sys.argv[1], os.O_RDONLY | os.O_DIRECTORY)
try:
    try:
        os.fsync(directory_fd)
    except OSError as error:
        if error.errno not in {errno.EINVAL, errno.ENOTSUP}:
            raise
finally:
    os.close(directory_fd)
PY
if ! rmdir -- "${PUBLISH_LOCK}"; then
  die "bundle was published but its cross-host publish lock could not be removed: ${PUBLISH_LOCK}"
fi
PUBLISH_LOCK_HELD=0
trap - EXIT

FINAL_WHEEL="${FINAL_BUNDLE}/${EXPECTED_WHEEL_FILENAME}"
FINAL_RECEIPT="${FINAL_BUNDLE}/${EXPECTED_WHEEL_FILENAME}.receipt.json"
FINAL_COMPLETE="${FINAL_BUNDLE}/BUNDLE-COMPLETE.sha256"

printf 'Verified Mooncake bundle:\n  %s\n' "${FINAL_BUNDLE}"
printf 'Artifact:\n  %s\n' "${FINAL_WHEEL}"
printf 'Receipt:\n  %s\n' "${FINAL_RECEIPT}"
printf 'Completeness marker:\n  %s\n' "${FINAL_COMPLETE}"
printf 'SHA256:\n  %s\n' "${WHEEL_SHA256}"
printf 'Build workspace retained at:\n  %s\n' "${WORK_DIR}"
