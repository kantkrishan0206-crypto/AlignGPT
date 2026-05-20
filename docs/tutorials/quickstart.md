# Quickstart

1. Install the package with development extras.
2. Run `pytest` to verify import-safe platform contracts.
3. Start the API gateway with `uvicorn backend.api_gateway.app:app --reload`.
4. Run the smoke benchmark with `python benchmarking/reproducibility/run_smoke_benchmark.py`.

The legacy RLHF trainers remain under `src/training` and require optional ML dependencies.
