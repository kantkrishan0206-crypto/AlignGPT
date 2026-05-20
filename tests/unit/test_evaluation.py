from aligngpt.benchmarks import run_smoke_benchmark
from aligngpt.evaluation import aggregate_metrics, exact_contains, lexical_diversity, summarize_output


def test_lexical_metrics_are_deterministic():
    assert lexical_diversity("alpha beta alpha") == 2 / 3
    assert exact_contains("Reward modeling improves alignment", ["reward", "alignment"]) == 1.0


def test_metric_aggregation():
    rows = [summarize_output("alpha beta"), summarize_output("alpha alpha beta")]
    aggregate = aggregate_metrics(rows)
    assert "lexical_diversity" in aggregate
    assert aggregate["token_count"] == 2.5


def test_smoke_benchmark_passes_default_cases():
    result = run_smoke_benchmark()
    assert result.passed
    assert result.benchmark_name == "aligngpt_smoke"
