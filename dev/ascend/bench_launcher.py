import subprocess
import yaml
from typing import Any
from pathlib import Path


def to_cmd(file: Path) -> list[str]:
    data: dict[str, Any] = yaml.safe_load(file.read_text())
    normalized_data = {to_snake_case(name): value for name, value in data.items()}
    cmd: list[str] = []

    for name, value in normalized_data.items():
        name = to_kebab_case(name)
        match value:
            case bool():
                if not value:
                    continue
                cmd.append(f"--{name}")
            case _:
                cmd += [
                    f"--{name}",
                    str(value),
                ]

    return cmd


def to_kebab_case(var_name: str) -> str:
    return var_name.replace("_", "-")


def to_snake_case(var_name: str) -> str:
    return var_name.replace("-", "_")


def main(config_path: Path) -> None:
    cmd = [
        "vllm",
        "bench",
        "serve",
    ]
    cmd += to_cmd(config_path)

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Config path of vllm-bench.yaml",
    )
    args = parser.parse_args()
    config_path = Path(args.config)

    main(config_path)
