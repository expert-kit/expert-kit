#!/usr/bin/env python3
"""Build the pinned ARM64 Ascend binding from verified offline source archives.

The wheel intentionally uses linux_aarch64, with CANN supplied by the runtime
image. It makes no manylinux portability claim and installs no host packages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CAPABILITIES = (
    "EK_ASCEND_SYNC_SUCCESS_COMPLETION",
    "EK_HAS_ASCEND_ACQUIRE",
    "EK_FORCE_CONFIGURED_ASCEND_DIRECT_TRANSPORT",
    "EK_DRAINED_ASCEND_REMOTE_DESCRIPTOR_INVALIDATION",
)


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def run(*args: str, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    subprocess.run(args, cwd=cwd, env=env, check=True)


def unpack(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise ValueError(f"source extraction destination is not empty: {destination}")
    run("tar", "-xzf", str(archive), "--strip-components=1", "-C", str(destination))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.machine() not in {"aarch64", "arm64"}:
        parser.error("this profile requires Linux ARM64")
    if sys.version_info[:2] != (3, 12) or args.jobs <= 0:
        parser.error("CPython 3.12 and a positive jobs value are required")
    cann = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/cann-9.0.1"))
    cann_include = cann / "aarch64-linux/include"
    if not (cann_include / "adxl/adxl_engine.h").is_file():
        parser.error("ASCEND_HOME_PATH must provide the CANN ADXL headers")
    manifest_path = ROOT / "build-manifest-ascend.toml"
    manifest = tomllib.loads(manifest_path.read_text())
    version_path = cann / "aarch64-linux/ascend_toolkit_install.info"
    if not version_path.is_file():
        parser.error("CANN toolkit version information is missing")
    cann_info = dict(
        line.split("=", 1)
        for line in version_path.read_text().splitlines()
        if "=" in line
    )
    if cann_info.get("version") != manifest["artifact"]["cann"]:
        parser.error("CANN version does not match the Ascend manifest")
    inputs = args.input_dir.resolve()
    work = args.work_dir.resolve()
    artifacts = args.artifact_dir.resolve()
    repository = ROOT.parent.parent
    for target in (work, artifacts):
        if target == repository or repository in target.parents:
            parser.error(
                "work and artifact directories must be outside the EK repository"
            )
    specifications = [manifest["upstream"], *manifest["submodules"]]
    for spec in specifications:
        archive = inputs / spec["archive"]
        if digest(archive) != spec["sha256"]:
            parser.error(f"source archive checksum mismatch: {archive.name}")
    for patch in manifest["patches"]:
        if digest(ROOT / patch["path"]) != patch["sha256"]:
            parser.error(f"patch checksum mismatch: {patch['path']}")
    fingerprint = hashlib.sha256(manifest_path.read_bytes()).hexdigest()[:16]
    build_root = work / fingerprint
    source = build_root / "source"
    marker = build_root / "SOURCE-READY"
    build_root.mkdir(parents=True, exist_ok=True)
    if not marker.exists():
        if source.exists():
            parser.error(
                f"incomplete source workspace exists: {source}; use a new work directory"
            )
        unpack(inputs / manifest["upstream"]["archive"], source)
        for spec in manifest["submodules"]:
            unpack(inputs / spec["archive"], source / "extern" / spec["name"])
        for patch in manifest["patches"]:
            run("git", "apply", "--check", str(ROOT / patch["path"]), cwd=source)
            run("git", "apply", str(ROOT / patch["path"]), cwd=source)
        marker.write_text(fingerprint + "\n")
    ylt_prefix = build_root / "ylt-install"
    ylt_build = build_root / "ylt-build"
    run(
        "cmake",
        "-S",
        str(source / "extern/yalantinglibs"),
        "-B",
        str(ylt_build),
        f"-DCMAKE_INSTALL_PREFIX={ylt_prefix}",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_EXAMPLES=OFF",
        "-DBUILD_BENCHMARK=OFF",
        "-DBUILD_UNIT_TESTS=OFF",
        "-DINSTALL_THIRDPARTY=ON",
        "-DINSTALL_STANDALONE=ON",
        "-DINSTALL_INDEPENDENT_THIRDPARTY=ON",
        "-DINSTALL_INDEPENDENT_STANDALONE=ON",
    )
    run("cmake", "--build", str(ylt_build), "--target", "install", "-j", str(args.jobs))
    definitions = [
        "WITH_TE=ON",
        "WITH_STORE=OFF",
        "WITH_STORE_RUST=OFF",
        "WITH_STORE_GO=OFF",
        "WITH_P2P_STORE=OFF",
        "WITH_EP=OFF",
        "WITH_RUST_EXAMPLE=OFF",
        "BUILD_EXAMPLES=OFF",
        "BUILD_BENCHMARK=OFF",
        "BUILD_UNIT_TESTS=OFF",
        "USE_CUDA=OFF",
        "USE_INTRA_NVLINK=OFF",
        "USE_ASCEND=OFF",
        "USE_ASCEND_DIRECT=ON",
        "USE_TCP=OFF",
        "USE_HTTP=OFF",
        "USE_TENT=OFF",
        "WITH_METRICS=OFF",
        "USE_ETCD=OFF",
        "USE_REDIS=OFF",
        "ENABLE_DEBUG_SYMBOLS=OFF",
    ]
    native_build = build_root / "build"
    run(
        "cmake",
        "-S",
        str(source),
        "-B",
        str(native_build),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON",
        "-DCMAKE_INSTALL_RPATH=$ORIGIN",
        f"-DCMAKE_PREFIX_PATH={ylt_prefix}",
        f"-DPython3_EXECUTABLE={sys.executable}",
        f"-DPYTHON_EXECUTABLE={sys.executable}",
        *(f"-D{definition}" for definition in definitions),
    )
    run(
        "cmake",
        "--build",
        str(native_build),
        "--target",
        "engine",
        "-j",
        str(args.jobs),
    )
    stage = build_root / "wheel-stage"
    package = stage / "mooncake"
    package.mkdir(parents=True, exist_ok=True)
    version = manifest["artifact"]["version"]
    (package / "__init__.py").write_text(f'__version__ = "{version}"\n')
    seen_libraries: dict[str, str] = {}
    for library in native_build.rglob("*.so"):
        checksum = digest(library)
        if library.name in seen_libraries and seen_libraries[library.name] != checksum:
            parser.error(f"duplicate shared-library filename: {library.name}")
        seen_libraries[library.name] = checksum
        shutil.copy2(library, package / library.name)
    if not list(package.glob("engine*.so")):
        parser.error("native engine extension is missing")
    check_env = dict(os.environ, PYTHONPATH=str(stage))
    run(
        sys.executable,
        "-c",
        "from mooncake import engine; "
        f"assert all(getattr(engine, name, False) is True for name in {CAPABILITIES!r})",
        env=check_env,
    )
    setup = stage / "setup.py"
    setup.write_text(
        "from setuptools import setup, Distribution\n"
        "class BinaryDistribution(Distribution):\n"
        "    def has_ext_modules(self): return True\n"
        f"setup(name='mooncake-transfer-engine', version={version!r}, "
        "packages=['mooncake'], package_data={'mooncake':['*.so']}, "
        "distclass=BinaryDistribution)\n"
    )
    artifacts.mkdir(parents=True, exist_ok=True)
    run(
        sys.executable,
        str(setup),
        "bdist_wheel",
        "--dist-dir",
        str(artifacts),
        cwd=stage,
    )
    wheels = list(artifacts.glob(f"mooncake_transfer_engine-{version}-*.whl"))
    if len(wheels) != 1:
        parser.error("build did not produce exactly one Ascend wheel")
    receipt = {
        "upstream": manifest["upstream"],
        "submodules": manifest["submodules"],
        "patches": manifest["patches"],
        "definitions": definitions,
        "python": sys.version,
        "architecture": platform.machine(),
        "cann_home": str(cann),
        "cann_version": cann_info["version"],
        "cann_info_sha256": digest(version_path),
        "builder_sha256": digest(Path(__file__)),
        "capabilities": CAPABILITIES,
        "wheel": wheels[0].name,
        "sha256": digest(wheels[0]),
        "manifest_sha256": digest(manifest_path),
    }
    wheels[0].with_suffix(".whl.receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(
        json.dumps({"wheel": str(wheels[0]), "sha256": receipt["sha256"]}), flush=True
    )


if __name__ == "__main__":
    main()
