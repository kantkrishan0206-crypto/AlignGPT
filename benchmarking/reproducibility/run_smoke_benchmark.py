"""Run the lightweight AlignGPT smoke benchmark and write a JSON report."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from aligngpt.benchmarks import run_smoke_benchmark, write_benchmark_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/benchmarks/smoke.json")
    args = parser.parse_args()

    result = run_smoke_benchmark()
    write_benchmark_report(result, args.output)
    if not result.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
