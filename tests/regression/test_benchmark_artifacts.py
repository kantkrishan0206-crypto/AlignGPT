import csv
import json
from pathlib import Path


RESULT_DIR = Path("benchmarking/results/2026-05-20")


def test_committed_benchmark_artifacts_have_expected_contract():
    summary = json.loads((RESULT_DIR / "summary.json").read_text(encoding="utf-8"))

    assert summary["summary"]["pass_rate"] >= 0.9
    assert len(summary["suites"]) == 7

    rows = list(csv.DictReader((RESULT_DIR / "metrics.csv").open(encoding="utf-8")))
    assert {row["suite"] for row in rows} >= {"hallucination", "latency", "reproducibility"}
