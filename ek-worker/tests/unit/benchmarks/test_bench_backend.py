import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def benchmark(monkeypatch):
    path = Path(__file__).parents[3] / "benchmarks" / "bench_backend.py"
    spec = importlib.util.spec_from_file_location("test_bench_backend_module", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("distribution", ["balanced", "uniform", "hot"])
def test_backend_benchmark_runs_torch_cpu_smoke(distribution: str) -> None:
    worker_root = Path(__file__).parents[3]
    completed = subprocess.run(
        [
            sys.executable,
            str(worker_root / "benchmarks" / "bench_backend.py"),
            "--backend",
            "torch",
            "--device",
            "cpu",
            "--dtype",
            "fp32",
            "--tokens",
            "4",
            "--distribution",
            distribution,
            "--hidden-dim",
            "8",
            "--intermediate-dim",
            "12",
            "--top-k",
            "2",
            "--experts",
            "4",
            "--concurrency",
            "2",
            "--routing-cases",
            "3",
            "--warmup",
            "0",
            "--repeats",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )

    result = json.loads(completed.stdout)

    assert result["correctness"]["checked"] is True
    assert result["correctness"]["criterion"] == "allclose"
    assert result["correctness"]["max_abs_error"] == 0
    assert result["correctness"]["max_relative_l2"] == 0
    assert result["correctness"]["min_cosine_similarity"] > 0.99999
    assert result["execution"]["concurrency"] == 2
    assert result["execution"]["profile"] is False
    assert result["routing"]["distribution"] == distribution
    assert result["routing"]["cases"] == 3
    assert 1 <= result["routing"]["unique_lease_keys"] <= 6
    assert result["latency_ms"]["mean"] > 0
    assert result["throughput"]["tokens_per_second"] > 0


@pytest.mark.parametrize("distribution", ["balanced", "uniform", "hot"])
def test_id_generators_are_seeded_and_preserve_distinct_topk(benchmark, distribution):
    args = benchmark.BenchmarkOptions(
        tokens=257,
        top_k=3,
        experts=19,
        distribution=benchmark.Distribution(distribution),
    )
    generate = benchmark.ID_GENERATORS[args.distribution]
    ids = generate(args, torch.Generator().manual_seed(47))
    repeated = generate(args, torch.Generator().manual_seed(47))
    assert torch.equal(ids, repeated)
    assert ids.shape == (257, 3)
    assert ids.dtype == torch.int32
    assert torch.all((ids >= 0) & (ids < 19))
    assert torch.all(ids.sort(dim=1).values.diff(dim=1) > 0)


def test_balanced_generator_has_even_assignment_counts(benchmark):
    args = benchmark.BenchmarkOptions(tokens=7, top_k=3, experts=5)
    ids = benchmark.ID_GENERATORS[args.distribution](args, torch.Generator())
    counts = torch.bincount(ids.flatten(), minlength=5)
    assert int(counts.max() - counts.min()) <= 1


def test_hot_generator_concentrates_on_configured_subset(benchmark):
    args = benchmark.BenchmarkOptions(
        tokens=2048,
        top_k=2,
        experts=32,
        distribution=benchmark.Distribution.HOT,
        hot_experts=2,
        hot_multiplier=50,
    )
    ids = benchmark.ID_GENERATORS[args.distribution](
        args, torch.Generator().manual_seed(47)
    )
    counts = torch.bincount(ids.flatten(), minlength=32)
    assert int(counts.topk(2).values.sum()) > ids.numel() * 0.7


def test_routing_cases_are_seeded_and_vary_independently_from_concurrency(benchmark):
    args = benchmark.BenchmarkOptions(
        tokens=4,
        top_k=2,
        experts=32,
        concurrency=2,
        routing_cases=3,
        distribution=benchmark.Distribution.HOT,
    )
    batches = [
        benchmark._make_batch(
            args,
            dtype=torch.float32,
            device=torch.device("cpu"),
            slot=slot,
            routing_case=routing_case,
        )
        for routing_case in range(args.routing_cases)
        for slot in range(args.concurrency)
    ]
    repeated = benchmark._make_batch(
        args,
        dtype=torch.float32,
        device=torch.device("cpu"),
        slot=0,
        routing_case=0,
    )

    assert torch.equal(batches[0].expert_ids, repeated.expert_ids)
    assert len({tuple(batch.expert_ids.flatten().tolist()) for batch in batches}) == 6
    assert len({batch.distinct_expert_ids for batch in batches}) == 6


def test_routing_replay_loads_exact_routes_and_lease_keys(benchmark, tmp_path: Path):
    path = tmp_path / "routing-replay.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [
                    {
                        "expert_ids": [[3, 1], [1, -1]],
                        "lease_expert_ids": [1, 3],
                    },
                    {
                        "expert_ids": [[2, 0], [2, 1]],
                        "lease_expert_ids": [0, 1, 2],
                    },
                ],
            }
        )
    )
    args = benchmark.BenchmarkOptions(
        tokens=2,
        top_k=2,
        experts=4,
        routing_replay=path,
    )

    cases = benchmark._load_routing_replay(args)

    assert cases is not None
    assert [case.expert_ids.tolist() for case in cases] == [
        [[3, 1], [1, -1]],
        [[2, 0], [2, 1]],
    ]
    assert [case.lease_expert_ids for case in cases] == [(1, 3), (0, 1, 2)]


def test_routing_replay_rejects_lease_that_does_not_match_routes(
    benchmark, tmp_path: Path
):
    path = tmp_path / "routing-replay.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [
                    {
                        "expert_ids": [[3, 1], [1, -1]],
                        "lease_expert_ids": [1],
                    }
                ],
            }
        )
    )
    args = benchmark.BenchmarkOptions(
        tokens=2,
        top_k=2,
        experts=4,
        routing_replay=path,
    )

    with pytest.raises(ValueError, match="must equal the sorted distinct"):
        benchmark._load_routing_replay(args)


def test_backend_benchmark_replays_normalized_routing(benchmark, tmp_path: Path):
    path = tmp_path / "routing-replay.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [
                    {
                        "expert_ids": [[3, 1], [1, -1]],
                        "lease_expert_ids": [1, 3],
                    },
                    {
                        "expert_ids": [[2, 0], [2, 1]],
                        "lease_expert_ids": [0, 1, 2],
                    },
                ],
            }
        )
    )
    args = benchmark.BenchmarkOptions(
        backend=benchmark.BackendName.TORCH,
        device="cpu",
        dtype=benchmark.ActivationDType.FP32,
        tokens=2,
        hidden_dim=8,
        intermediate_dim=12,
        top_k=2,
        experts=4,
        routing_replay=path,
        warmup=1,
        repeats=1,
    )

    result = benchmark._run_benchmark(args)

    assert result["routing"]["source"] == "replay"
    assert result["routing"]["cases"] == 2
    assert result["routing"]["unique_lease_keys"] == 2


def test_submission_group_cycles_routing_cases_and_can_reset(benchmark):
    submitted = []

    class Completion:
        def wait_host(self):
            pass

        def close(self):
            pass

    class Backend:
        def submit(self, batch, output):
            submitted.append((batch, output))
            return Completion()

    setup = SimpleNamespace(
        concurrency=1,
        runtime=SimpleNamespace(device=torch.device("cpu")),
        backend=Backend(),
        batches=["case-0", "case-1", "case-2"],
        outputs=["output-0", "output-1", "output-2"],
        routing_cases=3,
    )
    runner = benchmark._SubmissionGroup(setup)
    try:
        for _ in range(4):
            runner.run()
        runner.reset_routing_case()
        runner.run()
    finally:
        runner.close()

    assert submitted == [
        ("case-0", "output-0"),
        ("case-1", "output-1"),
        ("case-2", "output-2"),
        ("case-0", "output-0"),
        ("case-0", "output-0"),
    ]


def test_masking_preserves_original_weights_and_distinct_metadata(benchmark):
    unmasked = benchmark._make_batch(
        benchmark.BenchmarkOptions(tokens=32, top_k=3, experts=19),
        dtype=torch.float32,
        device=torch.device("cpu"),
        slot=0,
    )
    masked = benchmark._make_batch(
        benchmark.BenchmarkOptions(tokens=32, top_k=3, experts=19, inactive_rate=0.8),
        dtype=torch.float32,
        device=torch.device("cpu"),
        slot=0,
    )
    active = masked.expert_ids >= 0
    assert torch.equal(masked.routing_weights[active], unmasked.routing_weights[active])
    assert torch.all(masked.routing_weights[~active] == 0)
    assert torch.any(~active.any(dim=1))
    assert masked.distinct_expert_ids == tuple(
        torch.unique(masked.expert_ids[active]).tolist()
    )


def test_all_inactive_batch_is_finite_and_has_no_distinct_experts(benchmark):
    batch = benchmark._make_batch(
        benchmark.BenchmarkOptions(tokens=4, top_k=2, experts=8, inactive_rate=1),
        dtype=torch.float32,
        device=torch.device("cpu"),
        slot=0,
    )
    assert torch.all(batch.expert_ids == -1)
    assert torch.all(batch.routing_weights == 0)
    assert batch.distinct_expert_ids == ()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"top_k": 9, "experts": 8},
        {"inactive_rate": 1.1},
        {"hot_experts": 0},
        {"hot_experts": 9, "experts": 8},
        {"hot_multiplier": float("nan")},
        {"routing_cases": 0},
        {"repeats": 0},
    ],
)
def test_invalid_benchmark_options_are_rejected(benchmark, kwargs):
    with pytest.raises(ValueError):
        benchmark.BenchmarkOptions(**kwargs)


def test_typer_rejects_invalid_distribution_without_running_backend(benchmark):
    from typer.testing import CliRunner

    result = CliRunner().invoke(benchmark.app, ["--distribution", "invalid"])
    assert result.exit_code == 2
    assert "invalid" in result.output
