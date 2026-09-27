#!/usr/bin/env python3
"""Verify an installed EK Mooncake binding and, optionally, its wheel."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import importlib
import importlib.metadata
import io
import json
import os
import pathlib
import platform
import re
import subprocess
import sys
import tomllib
import zipfile
from email.parser import BytesParser
from email.policy import default as email_policy
from typing import Any

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_MANIFEST = SCRIPT_DIR / "build-manifest.toml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the installed EK-patched Mooncake extension, its native "
            "dependencies, and optional wheel metadata."
        )
    )
    parser.add_argument(
        "--manifest",
        type=pathlib.Path,
        default=DEFAULT_MANIFEST,
        help="build manifest (default: adjacent build-manifest.toml)",
    )
    parser.add_argument(
        "--wheel",
        type=pathlib.Path,
        help="also validate this wheel and run `python -m auditwheel show`",
    )
    parser.add_argument(
        "--require-torch",
        action="store_true",
        help=(
            "require torch and import it before mooncake.engine, matching the EK "
            "runtime load order so CUDA ABI conflicts fail during verification"
        ),
    )
    parser.add_argument(
        "--json",
        metavar="PATH|-",
        help="write the complete result as JSON; '-' writes it to stdout",
    )
    parser.add_argument(
        "--include-diagnostics",
        action="store_true",
        help=(
            "include absolute paths and full ldd/auditwheel output in JSON; "
            "default JSON is safe to retain as build provenance"
        ),
    )
    return parser.parse_args()


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_zip_member(wheel: pathlib.Path, member: str) -> tuple[str, int]:
    """Return the digest and uncompressed size of one unambiguous wheel member."""

    with zipfile.ZipFile(wheel) as archive:
        matches = [info for info in archive.infolist() if info.filename == member]
        if len(matches) != 1:
            raise ValueError(
                f"wheel must contain exactly one engine member {member!r}; "
                f"found {len(matches)}"
            )
        digest = hashlib.sha256()
        with archive.open(matches[0]) as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest(), matches[0].file_size


def distribution_member_for_path(
    distribution: importlib.metadata.Distribution,
    installed_path: pathlib.Path,
) -> tuple[str | None, dict[str, Any], list[str]]:
    """Match an installed file to this distribution's files and RECORD entries."""

    ownership_errors: list[str] = []
    files = distribution.files
    matched_members: list[str] = []
    if files is None:
        ownership_errors.append("installed distribution does not expose files/RECORD")
    else:
        for package_path in files:
            try:
                located_path = pathlib.Path(package_path.locate()).resolve()
            except (OSError, RuntimeError, ValueError):
                continue
            if located_path == installed_path:
                matched_members.append(str(package_path).replace("\\", "/"))

    try:
        record_text = distribution.read_text("RECORD")
    except (OSError, UnicodeError) as exc:
        record_text = None
        ownership_errors.append(f"cannot read installed distribution RECORD: {exc}")
    record_members = (
        {row[0] for row in csv.reader(io.StringIO(record_text)) if row and row[0]}
        if record_text is not None
        else set()
    )
    if record_text is None and not ownership_errors:
        ownership_errors.append("installed distribution RECORD is missing")

    if len(matched_members) != 1:
        ownership_errors.append(
            "mooncake.engine path is not an unambiguous installed-distribution file"
        )
        member = None
    else:
        candidate = matched_members[0]
        candidate_path = pathlib.PurePosixPath(candidate)
        if candidate_path.is_absolute() or ".." in candidate_path.parts:
            ownership_errors.append(
                "mooncake.engine distribution member has an unsafe path"
            )
            member = None
        elif candidate not in record_members:
            ownership_errors.append("mooncake.engine file is absent from RECORD")
            member = None
        else:
            member = candidate

    details: dict[str, Any] = {
        "distribution_file_matches": matched_members,
        "record_available": record_text is not None,
        "record_contains_engine": member is not None,
    }
    return member, details, ownership_errors


def read_os_release() -> dict[str, str]:
    values: dict[str, str] = {}
    with open("/etc/os-release", encoding="utf-8") as stream:
        for line in stream:
            key, separator, value = line.rstrip().partition("=")
            if separator:
                values[key] = value.strip('"')
    return values


def glibc_version() -> str:
    try:
        value = os.confstr("CS_GNU_LIBC_VERSION")
    except (OSError, ValueError):
        return platform.libc_ver()[1]
    if not value:
        return ""
    _, _, version = value.partition(" ")
    return version


def parse_wheel_metadata(wheel: pathlib.Path) -> dict[str, Any]:
    with zipfile.ZipFile(wheel) as archive:
        members = archive.namelist()
        unsafe = [
            name
            for name in members
            if pathlib.PurePosixPath(name).is_absolute()
            or ".." in pathlib.PurePosixPath(name).parts
        ]
        metadata_members = [
            name for name in members if name.endswith(".dist-info/METADATA")
        ]
        wheel_members = [name for name in members if name.endswith(".dist-info/WHEEL")]
        if len(metadata_members) != 1 or len(wheel_members) != 1:
            raise ValueError(
                "wheel must contain exactly one .dist-info/METADATA and WHEEL"
            )
        metadata = BytesParser(policy=email_policy).parsebytes(
            archive.read(metadata_members[0])
        )
        wheel_text = archive.read(wheel_members[0]).decode("utf-8")
    tags = [
        line.removeprefix("Tag: ")
        for line in wheel_text.splitlines()
        if line.startswith("Tag: ")
    ]
    return {
        "name": metadata.get("Name"),
        "version": metadata.get("Version"),
        "tags": tags,
        "unsafe_members": unsafe,
        "member_count": len(members),
    }


def main() -> int:
    args = parse_args()
    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    try:
        with args.manifest.open("rb") as stream:
            manifest = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        print(f"error: cannot read manifest {args.manifest}: {exc}", file=sys.stderr)
        return 2

    result: dict[str, Any] = {
        "schema_version": 1,
        "checked_at": dt.datetime.now(dt.UTC).isoformat(),
        "manifest": {
            "schema_version": manifest.get("schema_version"),
            "sha256": sha256(args.manifest),
        },
        "python": {
            "executable": (
                sys.executable
                if args.include_diagnostics
                else pathlib.Path(sys.executable).name
            ),
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
    }

    require(manifest.get("schema_version") == 1, "unsupported manifest schema")
    require(
        platform.python_version() == manifest["toolchain"]["python"],
        "Python version does not match the audited build",
    )
    require(
        platform.python_implementation()
        == manifest["toolchain"]["python_implementation"],
        "Python implementation does not match the audited build",
    )

    try:
        os_release = read_os_release()
    except OSError as exc:
        os_release = {}
        errors.append(f"cannot read /etc/os-release: {exc}")
    result["platform"] = {
        "os_id": os_release.get("ID", ""),
        "os_version_id": os_release.get("VERSION_ID", ""),
        "architecture": platform.machine(),
        "glibc": glibc_version(),
    }
    for key in ("os_id", "os_version_id", "architecture", "glibc"):
        require(
            result["platform"][key] == manifest["platform"][key],
            f"platform {key} does not match the audited build",
        )

    forbidden_environment = {
        name: {"present": name in os.environ}
        for name in manifest["verification"]["forbidden_environment"]
    }
    result["forbidden_environment"] = forbidden_environment
    for name, state in forbidden_environment.items():
        require(
            not state["present"], f"unsupported environment variable is set: {name}"
        )

    distribution_name = manifest["artifact"]["distribution"]
    installed_distribution: importlib.metadata.Distribution | None = None
    try:
        installed_distribution = importlib.metadata.distribution(distribution_name)
        installed_version = installed_distribution.version
    except importlib.metadata.PackageNotFoundError:
        installed_version = None
        errors.append(f"distribution is not installed: {distribution_name}")
    result["distribution"] = {
        "name": distribution_name,
        "version": installed_version,
    }
    require(
        installed_version == manifest["artifact"]["version"],
        "installed distribution version does not match the manifest",
    )

    expected_torch_cuda = manifest["verification"].get("expected_torch_cuda")
    expected_torch_version = manifest["verification"].get("expected_torch_version")
    require(
        isinstance(expected_torch_cuda, str) and bool(expected_torch_cuda),
        "manifest expected_torch_cuda is missing or invalid",
    )
    require(
        isinstance(expected_torch_version, str) and bool(expected_torch_version),
        "manifest expected_torch_version is missing or invalid",
    )
    torch_module: Any | None = None
    try:
        torch_module = importlib.import_module("torch")
    except Exception as exc:  # noqa: BLE001 - CUDA loader failures must be reported
        result["torch"] = {
            "available": False,
            "expected_cuda": expected_torch_cuda,
            "expected_version": expected_torch_version,
            "error": f"{type(exc).__name__}: {exc}",
        }
        if args.require_torch:
            errors.append(
                f"cannot import torch before Mooncake: {type(exc).__name__}: {exc}"
            )
    else:
        torch_version = getattr(torch_module, "__version__", None)
        torch_build = getattr(torch_module, "version", None)
        torch_cuda = getattr(torch_build, "cuda", None)
        result["torch"] = {
            "available": True,
            "version": str(torch_version) if torch_version is not None else None,
            "cuda": torch_cuda,
            "expected_cuda": expected_torch_cuda,
            "expected_version": expected_torch_version,
        }
        if args.require_torch:
            require(
                str(torch_version) == expected_torch_version,
                "torch.__version__ does not match the audited EK runtime",
            )
            require(
                torch_cuda == expected_torch_cuda,
                "torch.version.cuda does not match the audited Mooncake CUDA ABI",
            )

    engine: Any | None = None
    try:
        engine = importlib.import_module("mooncake.engine")
    except Exception as exc:  # noqa: BLE001 - native loader failures must be reported
        errors.append(f"cannot import mooncake.engine: {type(exc).__name__}: {exc}")

    capabilities: dict[str, Any] = {}
    engine_methods: dict[str, bool] = {}
    engine_path: pathlib.Path | None = None
    engine_distribution_member: str | None = None
    engine_ownership: dict[str, Any] = {
        "distribution_file_matches": [],
        "record_available": False,
        "record_contains_engine": False,
    }
    if engine is not None:
        for capability in manifest["verification"]["required_module_capabilities"]:
            value = getattr(engine, capability, None)
            capabilities[capability] = value
            require(value is True, f"required capability is not true: {capability}")

        engine_class = getattr(engine, "TransferEngine", None)
        require(engine_class is not None, "mooncake.engine.TransferEngine is missing")
        for method in manifest["verification"]["required_engine_methods"]:
            present = engine_class is not None and callable(
                getattr(engine_class, method, None)
            )
            engine_methods[method] = present
            require(present, f"required TransferEngine method is missing: {method}")

        module_file = getattr(engine, "__file__", None)
        if module_file:
            engine_path = pathlib.Path(module_file).resolve()
        require(
            engine_path is not None and engine_path.is_file(),
            "engine shared object is missing",
        )
        require(
            engine_path is not None and engine_path.suffix == ".so",
            "mooncake.engine is not a Linux shared object",
        )

        if engine_path is not None and engine_path.is_file():
            if installed_distribution is None:
                errors.append(
                    "cannot prove mooncake.engine ownership without its distribution"
                )
            else:
                (
                    engine_distribution_member,
                    engine_ownership,
                    ownership_errors,
                ) = distribution_member_for_path(installed_distribution, engine_path)
                errors.extend(ownership_errors)

    result["capabilities"] = capabilities
    result["engine_methods"] = engine_methods
    result["engine"] = {
        "path": (
            str(engine_path)
            if engine_path is not None and args.include_diagnostics
            else engine_path.name
            if engine_path is not None
            else None
        ),
        "distribution_member": engine_distribution_member,
        "ownership": engine_ownership,
    }

    ldd_result: dict[str, Any] = {
        "returncode": None,
        "missing": [],
        "output": "",
    }
    if engine_path is not None and engine_path.is_file():
        try:
            completed = subprocess.run(
                ["ldd", str(engine_path)],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except OSError as exc:
            errors.append(f"cannot run ldd: {exc}")
        else:
            missing = sorted(
                {
                    match.group(1)
                    for line in completed.stdout.splitlines()
                    if (match := re.match(r"^\s*(\S+)\s+=>\s+not found\s*$", line))
                }
            )
            required_libraries = manifest["verification"].get(
                "required_shared_libraries", []
            )
            required_presence = {
                library: library in completed.stdout for library in required_libraries
            }
            ldd_result = {
                "returncode": completed.returncode,
                "missing": missing,
                "required": required_presence,
                "output": completed.stdout if args.include_diagnostics else None,
            }
            require(completed.returncode == 0, "ldd failed for mooncake.engine")
            require(not missing, f"ldd reports missing libraries: {', '.join(missing)}")
            for library, present in required_presence.items():
                require(present, f"ldd does not report required library: {library}")
    result["ldd"] = ldd_result

    if args.wheel is not None:
        wheel = args.wheel.expanduser().resolve()
        wheel_result: dict[str, Any] = {
            "path": str(wheel) if args.include_diagnostics else wheel.name,
            "filename": wheel.name,
            "exists": wheel.is_file(),
        }
        require(wheel.is_file(), f"wheel does not exist: {wheel}")
        if wheel.is_file():
            wheel_result["size"] = wheel.stat().st_size
            wheel_result["sha256"] = sha256(wheel)
            require(
                wheel.name == manifest["artifact"]["wheel_filename"],
                "wheel filename does not match the manifest",
            )
            try:
                metadata = parse_wheel_metadata(wheel)
            except (OSError, ValueError, zipfile.BadZipFile) as exc:
                metadata = {"error": str(exc)}
                errors.append(f"cannot validate wheel metadata: {exc}")
            wheel_result["metadata"] = metadata
            if "error" not in metadata:
                require(
                    metadata["name"] == manifest["artifact"]["distribution"],
                    "wheel distribution name does not match the manifest",
                )
                require(
                    metadata["version"] == manifest["artifact"]["version"],
                    "wheel version does not match the manifest",
                )
                expected_tag = "-".join(
                    (
                        manifest["artifact"]["python_tag"],
                        manifest["artifact"]["abi_tag"],
                        manifest["artifact"]["platform_tag"],
                    )
                )
                require(
                    expected_tag in metadata["tags"],
                    "wheel metadata tag is not audited tag",
                )
                require(
                    not metadata["unsafe_members"],
                    "wheel contains an unsafe member path",
                )

            if engine_path is None or not engine_path.is_file():
                errors.append(
                    "cannot compare wheel engine binary because installed engine is missing"
                )
            elif engine_distribution_member is None:
                errors.append(
                    "cannot compare wheel engine binary without a verified RECORD member"
                )
            else:
                installed_engine_sha256 = sha256(engine_path)
                try:
                    wheel_engine_sha256, wheel_engine_size = sha256_zip_member(
                        wheel, engine_distribution_member
                    )
                except (OSError, ValueError, zipfile.BadZipFile) as exc:
                    wheel_result["engine_binary"] = {
                        "member": engine_distribution_member,
                        "error": str(exc),
                    }
                    errors.append(f"cannot hash wheel engine binary: {exc}")
                else:
                    wheel_result["engine_binary"] = {
                        "member": engine_distribution_member,
                        "installed_engine_sha256": installed_engine_sha256,
                        "wheel_engine_sha256": wheel_engine_sha256,
                        "wheel_uncompressed_size": wheel_engine_size,
                        "matches_installed": (
                            installed_engine_sha256 == wheel_engine_sha256
                        ),
                    }
                    require(
                        installed_engine_sha256 == wheel_engine_sha256,
                        "installed mooncake.engine hash does not match the wheel",
                    )

            completed = subprocess.run(
                [sys.executable, "-m", "auditwheel", "show", str(wheel)],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            auditwheel_ok = completed.returncode == 0
            wheel_result["auditwheel"] = {
                "ok": auditwheel_ok,
                "returncode": completed.returncode,
                "output": completed.stdout if args.include_diagnostics else None,
            }
            require(auditwheel_ok, "auditwheel show failed")
            require(
                manifest["artifact"]["platform_tag"] in completed.stdout,
                "auditwheel output does not confirm the audited platform tag",
            )
        result["wheel"] = wheel_result

    result["errors"] = errors
    result["ok"] = not errors

    if args.json == "-":
        json.dump(result, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    elif args.json:
        output_path = pathlib.Path(args.json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, sort_keys=True)
            stream.write("\n")

    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1

    summary_stream = sys.stderr if args.json == "-" else sys.stdout
    print(
        "Mooncake verification passed: "
        f"{distribution_name}=={installed_version}; "
        f"{len(capabilities)} capabilities; ldd clean",
        file=summary_stream,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
