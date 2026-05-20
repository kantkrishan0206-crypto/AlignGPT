# Inference Router

The router selects a model backend based on task, safety level, latency budget, cost constraints, model stage, and fallback policy.

The concrete import-safe implementation lives in `src/aligngpt/router.py`. It models GPU memory, context windows, quantization, health, cost, latency, microbatching, and fallback chains so local tests and production adapters share the same decision contract.
