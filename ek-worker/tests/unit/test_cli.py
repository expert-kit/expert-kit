"""Tests for the Python Worker command-line entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from expertkit_worker import cli
from expertkit_worker.config.models import LoggingConfig


class FakeApplication:
    def __init__(self, *, error: BaseException | None = None) -> None:
        self.error = error
        self.ran = False
        self.removed = False

    def install_signal_handlers(self) -> Any:
        def remove() -> None:
            self.removed = True

        return remove

    async def run(self) -> None:
        self.ran = True
        if self.error is not None:
            raise self.error


class FakeConfig:
    logging = LoggingConfig()


def test_main_loads_config_runs_application_and_removes_handlers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application = FakeApplication()
    configured: list[LoggingConfig] = []
    monkeypatch.setattr(cli, "load_config", lambda _path: FakeConfig())
    monkeypatch.setattr(cli, "configure_logging", configured.append)

    async def build(_config: object) -> FakeApplication:
        return application

    monkeypatch.setattr(cli, "build_worker_application", build)

    assert cli.main(["--config", "/tmp/worker.yaml"]) == 0
    assert application.ran is True
    assert application.removed is True
    assert configured == [FakeConfig.logging]


def test_main_uses_ek_config_when_argument_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application = FakeApplication()
    loaded_paths: list[str] = []
    monkeypatch.setenv("EK_CONFIG", "/tmp/from-environment.yaml")
    monkeypatch.setattr(
        cli,
        "load_config",
        lambda path: loaded_paths.append(path) or FakeConfig(),
    )
    monkeypatch.setattr(cli, "configure_logging", lambda _config: None)

    async def build(_config: object) -> FakeApplication:
        return application

    monkeypatch.setattr(cli, "build_worker_application", build)

    assert cli.main([]) == 0
    assert loaded_paths == ["/tmp/from-environment.yaml"]


def test_explicit_config_overrides_ek_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application = FakeApplication()
    loaded_paths: list[str] = []
    monkeypatch.setenv("EK_CONFIG", "/tmp/from-environment.yaml")
    monkeypatch.setattr(
        cli,
        "load_config",
        lambda path: loaded_paths.append(path) or FakeConfig(),
    )
    monkeypatch.setattr(cli, "configure_logging", lambda _config: None)

    async def build(_config: object) -> FakeApplication:
        return application

    monkeypatch.setattr(cli, "build_worker_application", build)

    assert cli.main(["--config", "/tmp/from-argument.yaml"]) == 0
    assert loaded_paths == ["/tmp/from-argument.yaml"]


def test_main_returns_failure_after_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    application = FakeApplication(error=RuntimeError("fatal device failure"))
    monkeypatch.setattr(cli, "load_config", lambda _path: FakeConfig())
    monkeypatch.setattr(cli, "configure_logging", lambda _config: None)

    async def build(_config: object) -> FakeApplication:
        return application

    monkeypatch.setattr(cli, "build_worker_application", build)

    assert cli.main(["--config", "/tmp/worker.yaml"]) == 1
    assert application.removed is True


def test_main_returns_configuration_error_for_missing_file(tmp_path: Path) -> None:
    assert cli.main(["--config", str(tmp_path / "missing.yaml")]) == 2


def test_main_requires_config_argument_or_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EK_CONFIG", raising=False)
    with pytest.raises(SystemExit) as caught:
        cli.main([])

    assert caught.value.code == 2
