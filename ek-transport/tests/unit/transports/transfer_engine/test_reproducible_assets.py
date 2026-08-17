"""Static checks for the source-reproducible Mooncake build assets."""

from __future__ import annotations

import hashlib
import importlib.util
import re
import subprocess
import tomllib
import zipfile
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
_MOONCAKE_ROOT = _REPOSITORY_ROOT / "third_party" / "mooncake"
_MANIFEST_PATH = _MOONCAKE_ROOT / "build-manifest.toml"
_VERIFY_PATH = _MOONCAKE_ROOT / "verify.py"
_VERIFY_SPEC = importlib.util.spec_from_file_location("verify_mooncake", _VERIFY_PATH)
assert _VERIFY_SPEC is not None and _VERIFY_SPEC.loader is not None
verify_mooncake = importlib.util.module_from_spec(_VERIFY_SPEC)
_VERIFY_SPEC.loader.exec_module(verify_mooncake)

_EXPECTED_PATCHES = (
    (
        "patches/ek-731c4521.patch",
        "c6168451f17443bc00e01455a55eff411a2787d8c88139aed43adf9048d78e26",
    ),
    (
        "patches/ek-rdma-731c4521.patch",
        "67cfe82e5191c8be2582f989f346a8d21527b87a2601314b36269cec0370b3f2",
    ),
)
_REQUIRED_CAPABILITIES = {
    "EK_SAFE_TERMINAL_BATCH_SYNC",
    "EK_HAS_GPUDIRECT_ACQUIRE",
    "EK_INTRA_NVLINK_REGISTRATION_REFCOUNT",
    "EK_FORCE_CONFIGURED_TRANSPORT",
    "EK_DRAINED_NVLINK_INTRA_LOCAL_INVALIDATION",
    "EK_FORCE_CONFIGURED_RDMA_TRANSPORT",
    "EK_DRAINED_RDMA_REMOTE_DESCRIPTOR_INVALIDATION",
}
_REQUIRED_METHODS = {
    "get_configured_backend",
    "flush_gpudirect_writes",
    "invalidate_drained_nvlink_intra_segment",
    "invalidate_drained_rdma_segment",
}
_RFC1918_ADDRESS = re.compile(
    r"(?<![0-9.])(?:10(?:\.\d{1,3}){3}|172\.(?:1[6-9]|2\d|3[01])"
    r"(?:\.\d{1,3}){2}|192\.168(?:\.\d{1,3}){2})(?![0-9.])"
)
_CREDENTIAL_ASSIGNMENT = re.compile(
    r"(?im)^\s*(?:password|passwd|secret|token|api[_-]?key|access[_-]?key)\s*[:=]"
    r"\s*(?![\"']?(?:\$|<))[^\s#]+"
)


def _manifest() -> dict[str, object]:
    return tomllib.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _DistributionFile:
    def __init__(self, member: str, installed_path: Path) -> None:
        self._member = member
        self._installed_path = installed_path

    def __str__(self) -> str:
        return self._member

    def locate(self) -> Path:
        return self._installed_path


class _Distribution:
    def __init__(
        self,
        *,
        files: list[_DistributionFile] | None,
        record: str | None,
    ) -> None:
        self.files = files
        self._record = record

    def read_text(self, filename: str) -> str | None:
        assert filename == "RECORD"
        return self._record


def test_manifest_pins_source_artifact_and_build_inputs() -> None:
    manifest = _manifest()

    assert manifest["schema_version"] == 1
    assert manifest["upstream"] == {
        "repository": "https://github.com/kvcache-ai/Mooncake.git",
        "commit": "731c4521dae71a78bc6fe4e218ad52cfc6ed3554",
        "source_date_epoch": 1784774179,
    }

    artifact = manifest["artifact"]
    assert artifact["distribution"] == "mooncake-transfer-engine"
    assert artifact["version"] == "0.3.12.dev20260817+ek.731c4521.rdma2"
    assert artifact["python_tag"] == artifact["abi_tag"] == "cp312"
    assert artifact["platform_tag"] == "manylinux_2_35_x86_64"
    assert artifact["wheel_filename"].endswith(
        f"-{artifact['python_tag']}-{artifact['abi_tag']}-{artifact['platform_tag']}.whl"
    )

    for submodule in manifest["submodules"].values():
        assert re.fullmatch(r"[0-9a-f]{40}", submodule["commit"])

    build_definitions = set(manifest["build"]["cmake_definitions"])
    assert {"WITH_TE=ON", "USE_CUDA=ON", "USE_INTRA_NVLINK=ON", "USE_TCP=ON"} <= (build_definitions)

    required_libraries = set(manifest["verification"]["required_shared_libraries"])
    assert {
        "libcudart.so.12",
        "libcuda.so.1",
        "libibverbs.so.1",
        "libmlx5.so.1",
        "libnuma.so.1",
    } <= required_libraries

    constraints = (_MOONCAKE_ROOT / "build-constraints.txt").read_text(encoding="utf-8")
    pins = [line for line in constraints.splitlines() if line and not line.startswith("#")]
    assert pins
    assert all(re.fullmatch(r"[A-Za-z0-9_.-]+==[^\s]+", pin) for pin in pins)

    lock = (_MOONCAKE_ROOT / "build-requirements.lock").read_text(encoding="utf-8")
    locked_requirements = re.findall(
        r"(?m)^([A-Za-z0-9_.-]+==[^\s]+) \\\n"
        r"\s+--hash=sha256:([0-9a-f]{64})$",
        lock,
    )
    assert {requirement for requirement, _hash in locked_requirements} == set(pins)
    assert len({digest for _requirement, digest in locked_requirements}) == len(pins)

    build_script = (_MOONCAKE_ROOT / "build.sh").read_text(encoding="utf-8")
    assert "transport_uint_test" in build_script
    assert "--require-torch" in build_script
    assert "--require-hashes" in build_script
    assert "PIP_NO_INDEX=1" in build_script


def test_ordered_patch_hashes_match_the_manifest() -> None:
    manifest = _manifest()
    entries = manifest["patches"]["files"]
    observed = tuple((entry["path"], entry["sha256"]) for entry in entries)

    assert observed == _EXPECTED_PATCHES
    for relative_path, expected_hash in observed:
        patch_path = (_MOONCAKE_ROOT / relative_path).resolve()
        assert patch_path.is_relative_to(_MOONCAKE_ROOT.resolve())
        assert patch_path.is_file()
        assert _sha256(patch_path) == expected_hash

    assert re.fullmatch(r"[0-9a-f]{64}", manifest["patches"]["combined_diff_sha256"])


def test_manifest_requires_both_patched_transfer_engine_profiles() -> None:
    verification = _manifest()["verification"]

    assert verification["expected_torch_cuda"] == "12.8"
    assert verification["expected_torch_version"] == "2.10.0+cu128"
    assert set(verification["required_module_capabilities"]) >= _REQUIRED_CAPABILITIES
    assert set(verification["required_engine_methods"]) >= _REQUIRED_METHODS
    assert {"MC_USE_TENT", "MC_USE_TEV1", "MC_FORCE_TCP"} <= set(
        verification["forbidden_environment"]
    )

    verifier = (_MOONCAKE_ROOT / "verify.py").read_text(encoding="utf-8")
    assert "expected_torch_cuda" in verifier
    assert "expected_torch_version" in verifier
    assert "distribution.files" in verifier
    assert 'distribution.read_text("RECORD")' in verifier
    assert "installed_engine_sha256" in verifier
    assert "wheel_engine_sha256" in verifier


def test_verifier_matches_engine_to_distribution_record_and_wheel(
    tmp_path: Path,
) -> None:
    engine = tmp_path / "mooncake" / "engine.cpython-312-x86_64-linux-gnu.so"
    engine.parent.mkdir()
    engine.write_bytes(b"audited-engine-binary")
    member = f"mooncake/{engine.name}"
    distribution = _Distribution(
        files=[_DistributionFile(member, engine)],
        record=f"{member},sha256=placeholder,{engine.stat().st_size}\n",
    )

    matched, details, errors = verify_mooncake.distribution_member_for_path(
        distribution, engine.resolve()
    )

    assert matched == member
    assert details["record_contains_engine"] is True
    assert errors == []

    wheel = tmp_path / "artifact.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(member, engine.read_bytes())
    wheel_hash, wheel_size = verify_mooncake.sha256_zip_member(wheel, member)
    assert wheel_hash == _sha256(engine)
    assert wheel_size == engine.stat().st_size


def test_verifier_rejects_unrecorded_or_ambiguous_engine_path(tmp_path: Path) -> None:
    engine = tmp_path / "engine.so"
    engine.write_bytes(b"engine")
    member = "mooncake/engine.so"
    distribution = _Distribution(
        files=[
            _DistributionFile(member, engine),
            _DistributionFile("shadow/engine.so", engine),
        ],
        record=f"{member},,\n",
    )

    matched, details, errors = verify_mooncake.distribution_member_for_path(
        distribution, engine.resolve()
    )

    assert matched is None
    assert len(details["distribution_file_matches"]) == 2
    assert "unambiguous installed-distribution file" in " ".join(errors)


def test_reproduction_assets_contain_no_environment_specific_secrets() -> None:
    asset_paths = (
        _MOONCAKE_ROOT / "README.md",
        _MOONCAKE_ROOT / "build-manifest.toml",
        _MOONCAKE_ROOT / "build-constraints.txt",
        _MOONCAKE_ROOT / "build-requirements.lock",
        _MOONCAKE_ROOT / "build.sh",
        _MOONCAKE_ROOT / "verify.py",
        _REPOSITORY_ROOT / "scripts" / "verify-transport-backends.py",
        _REPOSITORY_ROOT / "scripts" / "smoke-transfer-engine.py",
        *(_MOONCAKE_ROOT / path for path, _hash in _EXPECTED_PATCHES),
    )

    for path in asset_paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        assert not _RFC1918_ADDRESS.search(text), f"RFC1918 address found in {path.name}"
        assert "/ufs_" not in text, f"private filesystem path found in {path.name}"
        assert not re.search(r"/(?:home|Users)/(?!user(?:/|$)|<)[^/\s]+/", text), (
            f"user-specific home path found in {path.name}"
        )
        assert "-----BEGIN PRIVATE KEY-----" not in text
        assert not re.search(r"https?://[^/@\s:]+:[^/@\s]+@", text)
        assert not _CREDENTIAL_ASSIGNMENT.search(text), (
            f"credential-like assignment found in {path.name}"
        )


def test_native_artifacts_are_not_tracked_and_are_ignored() -> None:
    tracked = subprocess.run(
        ("git", "ls-files", "-z"),
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
    ).stdout.split(b"\0")
    tracked_paths = [Path(value.decode()) for value in tracked if value]

    assert not [path for path in tracked_paths if path.suffix == ".whl"]
    assert not [path for path in tracked_paths if path.name.endswith(".receipt.json")]

    ignored_examples = (
        "third_party/mooncake/dist/example.whl",
        "third_party/mooncake/dist/example.whl.receipt.json",
        "third_party/mooncake/work/native-object.o",
        "artifacts/mooncake/example.whl",
    )
    for candidate in ignored_examples:
        result = subprocess.run(
            ("git", "check-ignore", "--quiet", "--no-index", candidate),
            cwd=_REPOSITORY_ROOT,
            check=False,
        )
        assert result.returncode == 0, f"generated artifact is not ignored: {candidate}"


def test_build_script_is_fail_closed_and_publishes_an_atomic_bundle() -> None:
    build_script = (_MOONCAKE_ROOT / "build.sh").read_text(encoding="utf-8")
    readme = (_MOONCAKE_ROOT / "README.md").read_text(encoding="utf-8")

    assert "umask 077" in build_script
    assert 'GIT_ALLOW_PROTOCOL="https:file"' in build_script
    assert '{"https", "file"}' in build_script
    assert "--install requires --python to belong to a virtual environment" in build_script
    assert 'CUDA_LINK="/usr/local/cuda"' in build_script
    assert "--work-dir must be outside the Expert Kit Git worktree" in build_script
    assert "input_assets_clean_against_commit" in build_script
    assert "build_script_sha256" in build_script
    assert "verify_script_sha256" in build_script
    assert "BUNDLE-COMPLETE.sha256" in build_script
    assert ".publish.lock" in build_script
    assert 'mktemp -d "${ARTIFACT_DIR}/.${BUNDLE_NAME}.stage.' in build_script
    assert 'mv -T -- "${PUBLISH_STAGE}" "${FINAL_BUNDLE}"' in build_script
    assert not re.search(r"(?m)^\s*(?:sudo|apt(?:-get)?)\b", build_script)

    for package_name in (
        "libibverbs-dev",
        "libnuma-dev",
        "libcurl4-openssl-dev",
        "libssl-dev",
        "patchelf",
    ):
        assert package_name in readme
    assert "BUNDLE-COMPLETE.sha256" in readme
