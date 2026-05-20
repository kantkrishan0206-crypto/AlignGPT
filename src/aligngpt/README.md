# AlignGPT Core Package

This package contains import-safe platform primitives used by API scaffolds, SDK examples, tests, and future services.

It intentionally avoids importing heavy ML frameworks at module import time. Training, inference, and tokenizer dependencies should remain lazy and task-local.

Core responsibilities:

- `config.py`: environment-aware platform configuration.
- `schemas.py`: lightweight request, response, metric, and benchmark dataclasses.
- `safety.py`: prompt-injection and PII redaction starter policies.
- `evaluation.py`: deterministic lexical metrics and aggregation utilities.
- `benchmarks.py`: smoke benchmark runner for CI and reproducibility checks.
- `retrieval.py`: auditable in-memory retrieval contract for RAG development.
- `service.py`: orchestration boundary combining safety, retrieval, and placeholder inference.
- `observability.py`: structured run events and config fingerprints.
