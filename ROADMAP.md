# Roadmap

## Near Term

- Stabilize the core `aligngpt` package APIs.
- Add typed config validation for model, benchmark, safety, and deployment manifests.
- Convert the legacy RLHF trainers into package-level entry points with explicit optional dependencies.
- Add API contract tests for the FastAPI gateway.
- Implement dashboard data adapters for benchmark and experiment summaries.

## Research Milestones

- Reward-model calibration report with held-out preference pairs.
- DPO/PPO baseline comparison on a small reproducible instruction-following benchmark.
- Hallucination, refusal, robustness, and bias benchmark suites with confidence intervals.
- Ablation tracking for prompt templates, reward pooling, KL penalties, and retrieval settings.
- Model-card automation from evaluation manifests.

## Product Milestones

- Experiment dashboard with run status, metric deltas, safety flags, and artifact links.
- Admin panel for model registry and deployment promotion approvals.
- Benchmark viewer with regression detection and run-to-run comparison.
- SDK authentication and streaming inference examples.
- Human-feedback review queue for preference collection.

## Deployment Milestones

- Docker image build and vulnerability scan.
- Kubernetes deployment with autoscaling, resource limits, and readiness probes.
- Helm values for development, staging, and production.
- Terraform baseline for cloud networking, registry, and observability.
- GPU inference profile using vLLM or equivalent serving layer.
