.PHONY: install test lint format benchmark

install:
	python -m pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check src/aligngpt tests benchmarking/reproducibility cli/evaluation_cli evaluation/scientific_metrics

format:
	ruff format src/aligngpt tests benchmarking/reproducibility cli/evaluation_cli evaluation/scientific_metrics

benchmark:
	python benchmarking/reproducibility/run_smoke_benchmark.py --output artifacts/benchmarks/smoke.json
