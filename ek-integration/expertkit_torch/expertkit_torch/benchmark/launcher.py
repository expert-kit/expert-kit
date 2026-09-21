"""Single- and multi-process benchmark launchers."""

from __future__ import annotations

import multiprocessing as mp
import time
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass
from queue import Empty
from typing import Any, Protocol

from tqdm import tqdm

from expertkit_torch.benchmark.config import BenchmarkConfig, LauncherKind
from expertkit_torch.benchmark.datasets import ShareGPTDataset
from expertkit_torch.benchmark.runner import BenchmarkReport, run_benchmark
from expertkit_torch.models import load_model


@dataclass(frozen=True, slots=True)
class RankAssignment:
    """The device and deterministic prompt range assigned to one rank."""

    rank: int
    device_id: int
    prompt_offset: int
    num_prompts: int


@dataclass(frozen=True, slots=True)
class RankBenchmarkReport:
    """One rank's measured benchmark and shared timing interval."""

    assignment: RankAssignment
    started_at: float
    ended_at: float
    benchmark: BenchmarkReport

    @property
    def duration(self) -> float:
        return self.ended_at - self.started_at

    def as_dict(self) -> dict[str, object]:
        return {
            "rank": asdict(self.assignment),
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_seconds": self.duration,
            "benchmark": self.benchmark.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class GlobalBenchmarkReport:
    """Retain rank reports and aggregate work over their shared wall time."""

    model_type: str
    mode: str
    rank_reports: tuple[RankBenchmarkReport, ...]

    @property
    def started_at(self) -> float:
        return min(report.started_at for report in self.rank_reports)

    @property
    def ended_at(self) -> float:
        return max(report.ended_at for report in self.rank_reports)

    @property
    def duration_seconds(self) -> float:
        return self.ended_at - self.started_at

    @property
    def num_prompts(self) -> int:
        return sum(report.benchmark.num_prompts for report in self.rank_reports)

    @property
    def output_length(self) -> int:
        return self.rank_reports[0].benchmark.output_length

    @property
    def input_tokens(self) -> int:
        return sum(
            measurement.input_tokens
            for rank in self.rank_reports
            for batch in rank.benchmark.batches
            for measurement in batch.measurements
        )

    @property
    def output_tokens(self) -> int:
        return sum(
            measurement.batch_size * measurement.output_length
            for rank in self.rank_reports
            for batch in rank.benchmark.batches
            for measurement in batch.measurements
        )

    @property
    def request_throughput(self) -> float:
        return self.num_prompts / self.duration_seconds

    @property
    def output_throughput(self) -> float:
        return self.output_tokens / self.duration_seconds

    @property
    def total_throughput(self) -> float:
        return (self.input_tokens + self.output_tokens) / self.duration_seconds

    def as_dict(self, *, configuration: dict[str, object] | None = None) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": 3,
            "model_type": self.model_type,
            "mode": self.mode,
            "num_prompts": self.num_prompts,
            "output_length": self.output_length,
            "duration_seconds": self.duration_seconds,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "request_throughput": self.request_throughput,
            "output_throughput": self.output_throughput,
            "total_throughput": self.total_throughput,
            "ranks": [rank.as_dict() for rank in self.rank_reports],
        }
        if configuration is not None:
            payload["configuration"] = configuration
        return payload


class BenchmarkLauncher(Protocol):
    """Execute a validated benchmark configuration."""

    def launch(self, config: BenchmarkConfig) -> GlobalBenchmarkReport: ...


def make_assignments(config: BenchmarkConfig) -> tuple[RankAssignment, ...]:
    """Partition prompts as evenly as possible in device-list order."""

    base, remainder = divmod(config.num_prompts, config.rank_count)
    assignments: list[RankAssignment] = []
    offset = 0
    for rank, device_id in enumerate(config.device_ids):
        count = base + (1 if rank < remainder else 0)
        assignments.append(
            RankAssignment(
                rank=rank,
                device_id=device_id,
                prompt_offset=offset,
                num_prompts=count,
            )
        )
        offset += count
    return tuple(assignments)


def execute_rank(
    config: BenchmarkConfig,
    assignment: RankAssignment,
    *,
    barrier: Any = None,
    clock: Callable[[], float] = time.perf_counter,
    on_progress: Callable[[int], None] | None = None,
) -> RankBenchmarkReport:
    """Load and measure one rank; this function is spawn-safe."""

    device = config.device_name(assignment.device_id)
    dataset = ShareGPTDataset(
        config.dataset_path,
        seed=config.seed,
        offset=assignment.prompt_offset,
    )
    measured_start: list[float | None] = [None]

    def start_measurement() -> None:
        if barrier is not None:
            barrier.wait()
        measured_start[0] = clock()

    with load_model(
        config.model_path,
        mode=config.mode.value,
        controller_endpoint=config.controller_endpoint or "",
        instance_id=config.instance_id,
        device=device,
        dtype=config.dtype.torch_name,
    ) as loaded:
        benchmark = run_benchmark(
            loaded.model,
            loaded.tokenizer,
            dataset,
            model_type=loaded.model_type,
            mode=config.mode.value,
            batch_sizes=(config.batch_size_per_rank,),
            num_prompts=assignment.num_prompts,
            output_length=config.output_length,
            warmup_runs=config.warmup_runs,
            device=device,
            before_measurement=start_measurement,
            on_progress=on_progress,
        )

    if measured_start[0] is None:
        raise RuntimeError("benchmark did not enter its measured phase")
    return RankBenchmarkReport(
        assignment=assignment,
        started_at=measured_start[0],
        ended_at=clock(),
        benchmark=benchmark,
    )


def _aggregate(reports: list[RankBenchmarkReport]) -> GlobalBenchmarkReport:
    if not reports:
        raise RuntimeError("benchmark produced no rank reports")
    ordered = tuple(sorted(reports, key=lambda report: report.assignment.rank))
    model_types = {report.benchmark.model_type for report in ordered}
    modes = {report.benchmark.mode for report in ordered}
    if len(model_types) != 1 or len(modes) != 1:
        raise RuntimeError("rank reports disagree about model type or mode")
    return GlobalBenchmarkReport(
        model_type=ordered[0].benchmark.model_type,
        mode=ordered[0].benchmark.mode,
        rank_reports=ordered,
    )


class InlineLauncher:
    """Execute one rank in the invoking process."""

    def launch(self, config: BenchmarkConfig) -> GlobalBenchmarkReport:
        assignment = make_assignments(config)
        if len(assignment) != 1:
            raise ValueError("inline launcher requires exactly one assignment")
        with tqdm(total=config.num_prompts, desc="Benchmark", unit="prompt") as progress:
            report = execute_rank(config, assignment[0], on_progress=progress.update)
        return _aggregate([report])


def _spawn_entry(
    config: BenchmarkConfig,
    assignment: RankAssignment,
    barrier: Any,
    result_queue: Any,
    rank_executor: Callable[..., RankBenchmarkReport] = execute_rank,
) -> None:
    try:

        def report_progress(completed: int) -> None:
            result_queue.put(("progress", assignment.rank, completed))

        result_queue.put(
            (
                "ok",
                rank_executor(
                    config,
                    assignment,
                    barrier=barrier,
                    on_progress=report_progress,
                ),
            )
        )
    except BaseException as error:  # propagate all child failures to the parent
        result_queue.put(
            (
                "error",
                assignment.rank,
                type(error).__name__,
                str(error),
                traceback.format_exc(),
            )
        )


class SpawnLauncher:
    """Run one independent frontend process per configured device."""

    def __init__(
        self,
        *,
        rank_executor: Callable[..., RankBenchmarkReport] = execute_rank,
        context_factory: Callable[[str], Any] = mp.get_context,
    ) -> None:
        self.rank_executor = rank_executor
        self.context_factory = context_factory

    def launch(self, config: BenchmarkConfig) -> GlobalBenchmarkReport:
        assignments = make_assignments(config)
        if len(assignments) < 2:
            raise ValueError("spawn launcher requires at least two assignments")

        context = self.context_factory("spawn")
        barrier = context.Barrier(len(assignments))
        result_queue = context.Queue()
        processes = [
            context.Process(
                target=_spawn_entry,
                args=(config, assignment, barrier, result_queue, self.rank_executor),
            )
            for assignment in assignments
        ]
        for process in processes:
            process.start()

        reports: list[RankBenchmarkReport] = []
        reported_ranks: set[int] = set()
        with tqdm(total=config.num_prompts, desc="Benchmark", unit="prompt") as progress:
            try:
                while len(reports) < len(processes):
                    try:
                        message = result_queue.get(timeout=0.5)
                    except Empty:
                        failed = next(
                            (
                                (assignment, process)
                                for assignment, process in zip(assignments, processes, strict=True)
                                if not process.is_alive()
                                and process.exitcode is not None
                                and assignment.rank not in reported_ranks
                            ),
                            None,
                        )
                        if failed is not None:
                            assignment, process = failed
                            raise RuntimeError(
                                f"benchmark rank {assignment.rank} exited with code "
                                f"{process.exitcode} without a report"
                            ) from None
                        continue

                    if message[0] == "progress":
                        progress.update(message[2])
                        continue
                    if message[0] == "error":
                        _, rank, error_type, error_message, error_traceback = message
                        raise RuntimeError(
                            f"benchmark rank {rank} failed with {error_type}: "
                            f"{error_message}\n{error_traceback}"
                        )
                    report = message[1]
                    reports.append(report)
                    reported_ranks.add(report.assignment.rank)
            except BaseException:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                raise
            finally:
                for process in processes:
                    process.join()

        return _aggregate(reports)


def select_launcher(config: BenchmarkConfig) -> BenchmarkLauncher:
    """Resolve explicit or automatic launcher selection."""

    kind = config.launcher
    if kind is LauncherKind.AUTO:
        kind = LauncherKind.INLINE if config.rank_count == 1 else LauncherKind.SPAWN
    if kind is LauncherKind.INLINE:
        return InlineLauncher()
    if kind is LauncherKind.SPAWN:
        return SpawnLauncher()
    raise ValueError(f"unsupported launcher: {kind}")
