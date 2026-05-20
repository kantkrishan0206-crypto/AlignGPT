# Changelog

## 0.3.0 - Operational SaaS Platform Layer

- Added GPU-aware inference router with capacity, latency, quantization, health, and fallback decisions.
- Added end-to-end alignment evaluation pipeline with retrieval, safety, routing, reward scoring, metrics, traces, and report export.
- Added operational benchmark generator and committed JSON/CSV/Markdown benchmark artifacts.
- Expanded FastAPI gateway with readiness, metrics, evaluation, status, and event-stream endpoints.
- Upgraded frontend into public landing, dashboard, benchmark, deployment, and trace pages.
- Added production diagrams, deployment runbook, and architecture rationale documentation.

## 0.2.0 - Platform Restructure

- Reorganized the project around a research-grade AI platform architecture.
- Added top-level governance docs: architecture, research, security, roadmap, model card, contributing guide, and changelog.
- Added CI/CD, test, security, and benchmark workflow scaffolds.
- Added backend, frontend, deployment, infrastructure, benchmark, security, evaluation, SDK, pipeline, and CLI subsystem scaffolds.
- Added lightweight `aligngpt` core package with import-safe config, schemas, safety, service, evaluation, benchmarking, retrieval, and observability utilities.
- Replaced network-heavy prototype tests with fast unit and integration tests.

## 0.1.0 - Prototype Baseline

- Initial RLHF prototype with prompt building, preference pairs, reward modeling, SFT, PPO, DPO, evaluation utilities, configs, toy data, and notebooks.
