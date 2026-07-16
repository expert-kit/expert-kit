"""Generate the single Python v2 protobuf binding distributed by this package."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

TRANSPORT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = TRANSPORT_ROOT.parent
PROTO_ROOT = REPOSITORY_ROOT / "ek-proto"
OUTPUT_ROOT = TRANSPORT_ROOT / "src" / "expertkit_transport" / "_proto"
PROTO_FILES = (
    PROTO_ROOT / "ek" / "worker" / "v2" / "common.proto",
    PROTO_ROOT / "ek" / "worker" / "v2" / "computation.proto",
    PROTO_ROOT / "ek" / "control" / "v2" / "lifecycle.proto",
    PROTO_ROOT / "ek" / "control" / "v2" / "weight_control.proto",
)
GENERATED_IMPORT_PREFIX = "expertkit_transport._proto."


def _rewrite_generated_imports(path: Path) -> None:
    contents = path.read_text(encoding="utf-8")
    contents = contents.replace("from ek.", f"from {GENERATED_IMPORT_PREFIX}ek.")
    contents = contents.replace("import ek.", f"import {GENERATED_IMPORT_PREFIX}ek.")
    path.write_text(contents, encoding="utf-8")


def main() -> None:
    """Regenerate all v2 modules and deterministic package markers."""

    shutil.rmtree(OUTPUT_ROOT, ignore_errors=True)
    OUTPUT_ROOT.mkdir(parents=True)
    command = [
        sys.executable,
        "-m",
        "grpc_tools.protoc",
        f"-I{PROTO_ROOT}",
        f"--python_out={OUTPUT_ROOT}",
        f"--pyi_out={OUTPUT_ROOT}",
        f"--grpc_python_out={OUTPUT_ROOT}",
        *(str(path) for path in PROTO_FILES),
    ]
    subprocess.run(command, check=True)

    for path in OUTPUT_ROOT.rglob("*_pb2*.py"):
        _rewrite_generated_imports(path)
    for path in OUTPUT_ROOT.rglob("*_pb2*.pyi"):
        _rewrite_generated_imports(path)

    package_directories = {OUTPUT_ROOT}
    for path in OUTPUT_ROOT.rglob("*.py"):
        directory = path.parent
        while directory != OUTPUT_ROOT:
            package_directories.add(directory)
            directory = directory.parent
    for directory in package_directories:
        (directory / "__init__.py").write_text(
            '"""Generated protobuf package; do not add handwritten domain logic."""\n',
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
