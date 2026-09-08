"""Tests for rank partitioning and global report aggregation."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from queue import Empty
from typing import ClassVar

import pytest

import expertkit_torch.benchmark.launcher as launcher
from expertkit_torch.benchmark.config import BenchmarkConfig
from expertkit_torch.benchmark.launcher import (
    GlobalBenchmarkReport,
    RankAssignment,
    RankBenchmarkReport,
    SpawnLauncher,
    make_assignments,
    select_launcher,
)
from expertkit_torch.benchmark.runner import BatchBenchmark, BenchmarkReport, RunMetrics


def _config(**overrides: object) -> BenchmarkConfig:
    values: dict[str, object] = {
        "mode": "local",
        "device_platform": "cpu",
        "device_ids": (0, 1, 2),
        "model_path": Path("/model"),
        "dataset_path": Path("/dataset.json"),
        "num_prompts": 8,
        "max_concurrency": 6,
        "output_length": 2,
        "warmup_runs": 0,
        **overrides,
    }
    return BenchmarkConfig.model_validate(values)


def test_assignments_are_ordered_and_cover_all_prompts() -> None:
    assignments = make_assignments(_config())

    assert assignments == (
        RankAssignment(0, 0, 0, 3),
        RankAssignment(1, 1, 3, 3),
        RankAssignment(2, 2, 6, 2),
    )


def test_launcher_auto_selects_inline_or_spawn() -> None:
    assert type(select_launcher(_config(device_ids=(0,), max_concurrency=2))).__name__ == (
        "InlineLauncher"
    )
    assert type(select_launcher(_config())).__name__ == "SpawnLauncher"


def _fake_rank_executor(
    config: BenchmarkConfig,
    assignment: RankAssignment,
    *,
    barrier: object = None,
    on_progress: Callable[[int], None] | None = None,
) -> RankBenchmarkReport:
    del config
    if barrier is not None:
        barrier.wait()
    if on_progress is not None:
        on_progress(assignment.num_prompts)
    benchmark = BenchmarkReport(
        model_type="fake",
        mode="local",
        num_prompts=assignment.num_prompts,
        output_length=2,
        warmup_runs=0,
        batches=(
            BatchBenchmark(
                2,
                (
                    RunMetrics(
                        assignment.num_prompts,
                        assignment.num_prompts * 3,
                        3,
                        2,
                        1.0,
                        1.0,
                    ),
                ),
            ),
        ),
    )
    return RankBenchmarkReport(assignment, 1.0, 3.0, benchmark)


def _failing_rank_executor(
    config: BenchmarkConfig,
    assignment: RankAssignment,
    *,
    barrier: object = None,
    on_progress: Callable[[int], None] | None = None,
) -> RankBenchmarkReport:
    del barrier
    if assignment.rank == 1:
        raise RuntimeError("synthetic rank failure")
    return _fake_rank_executor(config, assignment, on_progress=on_progress)


class _FakeBarrier:
    def wait(self) -> None:
        return None


class _FakeQueue:
    def __init__(self) -> None:
        self.messages: list[object] = []

    def put(self, message: object) -> None:
        self.messages.append(message)

    def get(self, timeout: float) -> object:
        del timeout
        if not self.messages:
            raise Empty
        return self.messages.pop(0)


class _FakeProcess:
    def __init__(self, target, args, started: list[_FakeProcess]) -> None:
        self.target = target
        self.args = args
        self.started = started
        self.exitcode: int | None = None
        started.append(self)

    def start(self) -> None:
        self.target(*self.args)
        self.exitcode = 0

    def is_alive(self) -> bool:
        return self.exitcode is None

    def terminate(self) -> None:
        self.exitcode = -15

    def join(self) -> None:
        return None


class _FakeContext:
    def __init__(self) -> None:
        self.started: list[_FakeProcess] = []
        self.queue = _FakeQueue()

    def Barrier(self, count: int) -> _FakeBarrier:
        del count
        return _FakeBarrier()

    def Queue(self) -> _FakeQueue:
        return self.queue

    def Process(self, *, target, args) -> _FakeProcess:
        return _FakeProcess(target, args, self.started)


class _FakeProgress:
    instances: ClassVar[list[_FakeProgress]] = []

    def __init__(self, *, total: int, desc: str, unit: str) -> None:
        self.total = total
        self.desc = desc
        self.unit = unit
        self.updates: list[int] = []
        self.instances.append(self)

    def __enter__(self) -> _FakeProgress:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def update(self, count: int) -> None:
        self.updates.append(count)


def test_spawn_launcher_runs_one_process_per_device(monkeypatch) -> None:
    _FakeProgress.instances.clear()
    monkeypatch.setattr(launcher, "tqdm", _FakeProgress)
    context = _FakeContext()
    report = SpawnLauncher(
        rank_executor=_fake_rank_executor,
        context_factory=lambda _: context,
    ).launch(_config())

    assert len(context.started) == 3
    assert [rank.assignment.device_id for rank in report.rank_reports] == [0, 1, 2]
    assert report.num_prompts == 8
    assert _FakeProgress.instances[0].total == 8
    assert _FakeProgress.instances[0].updates == [3, 3, 2]


def test_spawn_launcher_propagates_child_failure(monkeypatch) -> None:
    monkeypatch.setattr(launcher, "tqdm", _FakeProgress)
    context = _FakeContext()
    with pytest.raises(RuntimeError, match=r"rank 1 failed.*synthetic rank failure"):
        SpawnLauncher(
            rank_executor=_failing_rank_executor,
            context_factory=lambda _: context,
        ).launch(_config())


def test_global_report_aggregates_work_over_shared_wall_time() -> None:
    def rank_report(rank: int, start: float, end: float) -> RankBenchmarkReport:
        return RankBenchmarkReport(
            assignment=RankAssignment(rank, rank, rank * 2, 2),
            started_at=start,
            ended_at=end,
            benchmark=BenchmarkReport(
                model_type="fake",
                mode="local",
                num_prompts=2,
                output_length=2,
                warmup_runs=0,
                batches=(
                    BatchBenchmark(
                        1,
                        (RunMetrics(1, 3, 3, 2, 1.0, 1.0),),
                    ),
                ),
            ),
        )

    report = GlobalBenchmarkReport(
        "fake",
        "local",
        (rank_report(0, 10.0, 13.0), rank_report(1, 10.5, 14.0)),
    )

    assert report.duration_seconds == 4.0
    assert report.num_prompts == 4
    assert report.input_tokens == 6
    assert report.output_tokens == 4
    assert report.output_throughput == 1.0
    assert report.total_throughput == 2.5


@pytest.mark.parametrize(
    "launcher, device_ids",
    [("inline", (0, 1)), ("spawn", (0,))],
)
def test_explicit_launcher_shape_is_validated(
    launcher: str,
    device_ids: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError, match="launcher"):
        _config(launcher=launcher, device_ids=device_ids, max_concurrency=len(device_ids))
