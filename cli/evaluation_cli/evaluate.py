"""Evaluation CLI entry point for lightweight smoke checks."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from aligngpt.benchmarks import run_smoke_benchmark  # noqa: E402


def main() -> None:
    result = run_smoke_benchmark()
    print(f"{result.benchmark_name}: {'pass' if result.passed else 'fail'}")
    for metric in result.metrics:
        print(f"{metric.name}={metric.value:.4f}")
    if not result.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
