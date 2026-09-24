import json
import subprocess
import sys
from pathlib import Path


def test_backend_benchmark_runs_torch_cpu_smoke() -> None:
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
    assert result["latency_ms"]["mean"] > 0
    assert result["throughput"]["tokens_per_second"] > 0
