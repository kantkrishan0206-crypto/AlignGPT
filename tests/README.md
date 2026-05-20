# Tests

The default test suite is intentionally fast and network-free. It validates platform contracts, safety policies, deterministic metrics, and orchestration behavior.

Heavier model tests should use `pytest.mark.ml` and run only in GPU/model-artifact pipelines.
