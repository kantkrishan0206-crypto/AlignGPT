"""Small reproducible benchmark runner used by CI and example pipelines."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from aligngpt.evaluation import exact_contains, summarize_output
from aligngpt.schemas import BenchmarkResult, EvalMetric


@dataclass(frozen=True)
class BenchmarkCase:
    prompt: str
    expected_terms: tuple[str, ...]
    reference: str = ""


def run_smoke_benchmark(cases: tuple[BenchmarkCase, ...] | None = None) -> BenchmarkResult:
    cases = cases or DEFAULT_CASES
    per_case_scores = []
    metric_rows = []
    for case in cases:
        output = _deterministic_baseline(case.prompt)
        per_case_scores.append(exact_contains(output, case.expected_terms))
        metric_rows.append(summarize_output(output, case.reference or case.prompt))
    mean_required_term_score = sum(per_case_scores) / max(1, len(per_case_scores))
    lexical_mean = sum(row[0].value for row in metric_rows) / max(1, len(metric_rows))
    passed = mean_required_term_score >= 0.75
    return BenchmarkResult(
        benchmark_name="aligngpt_smoke",
        metrics=(
            EvalMetric("required_term_coverage", mean_required_term_score, True),
            EvalMetric("lexical_diversity_mean", lexical_mean, True),
        ),
        passed=passed,
        threshold_summary="required_term_coverage >= 0.75",
        metadata={"case_count": len(cases)},
    )


def write_benchmark_report(result: BenchmarkResult, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark_name": result.benchmark_name,
        "passed": result.passed,
        "threshold_summary": result.threshold_summary,
        "metrics": [metric.__dict__ for metric in result.metrics],
        "metadata": result.metadata,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _deterministic_baseline(prompt: str) -> str:
    return f"Alignment response: {prompt.strip()} safety evaluation benchmark"


DEFAULT_CASES = (
    BenchmarkCase("Explain reward modeling.", ("reward", "modeling", "safety")),
    BenchmarkCase("Describe benchmark reproducibility.", ("benchmark", "reproducibility")),
)
