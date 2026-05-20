# Architecture

AlignGPT is designed as a layered AI platform, not a single training script.

## Service Boundaries

- Frontend: dashboards for experiments, benchmarks, safety review, and deployment state.
- API Gateway: HTTP boundary for inference, evaluation, run metadata, and admin actions.
- Safety Layer: prompt-injection checks, jailbreak scoring, PII redaction, and access policy enforcement.
- Inference Router: routes requests to local checkpoints, hosted models, reward models, rerankers, or embedding services.
- Retrieval Layer: indexes curated corpora and returns auditable context with provenance.
- Evaluation Layer: computes scientific metrics, reward scores, ranking summaries, and benchmark reports.
- Experiment Registry: stores run manifests, config hashes, dataset fingerprints, artifact references, and model-card metadata.
- Observability: metrics, traces, structured logs, alerts, and benchmark regressions.

## Request Lifecycle

1. Client sends an inference or evaluation request through SDK, CLI, or frontend.
2. API Gateway validates schema, auth context, and rate limits.
3. Safety Layer normalizes input, detects injection patterns, redacts sensitive fields, and records policy decisions.
4. Retrieval Layer optionally fetches context with source provenance.
5. Inference Router selects model backend based on task, risk level, cost, latency budget, and deployment stage.
6. Response is scored by safety, reward, and evaluation hooks when configured.
7. Observability captures non-sensitive metrics, trace identifiers, latency, routing decision, and policy outcomes.

## Storage Layers

- Configs: versioned YAML/JSON files under `configs/`.
- Dataset Registry: dataset manifests, licenses, schema fingerprints, and quality gates under `datasets/dataset_registry/`.
- Experiment Tracking: run manifests and reproducibility protocols under `research/experiment_tracking/`.
- Feature Store: online/offline feature contracts under `backend/feature_store/`.
- Database: application schema and migration stubs under `backend/database/`.
- Artifacts: checkpoints, benchmark reports, generated model cards, and dashboard exports are expected to live in configured object storage for production deployments.

## Deployment Topology

Development runs locally with the FastAPI gateway, lightweight core package, and file-backed configs. Staging runs containerized services with Redis, Postgres-compatible storage, Prometheus, and a mock model backend. Production adds GPU inference workers, autoscaling, ingress controls, secret management, distributed tracing, and policy-enforced artifact promotion.

## Maintainability Decisions

- Domain folders mirror platform responsibilities so ownership is clear.
- Heavy ML code remains optional and lazy-loaded.
- Tests focus on contracts, deterministic utilities, and policy behavior first.
- TODOs are explicit where full implementations would require infrastructure, GPUs, private datasets, or hosted services.
