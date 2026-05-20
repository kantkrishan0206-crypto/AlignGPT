# Production Deployment Runbook

This runbook describes the path from local development to an internet-facing AlignGPT deployment.

## Target Topology

- Frontend: Vercel hosting `frontend/web/nextjs_app`.
- Backend API: Render, Railway, AWS ECS, or Google Cloud Run running `backend.api_gateway.app`.
- Database: PostgreSQL for users, runs, benchmark history, and model registry state.
- Redis: cache, rate-limit counters, event streaming, and queue coordination.
- Inference: vLLM GPU worker, hosted inference provider, and CPU safety fallback.
- Observability: Prometheus metrics, Grafana dashboard, structured logs, and trace events.

## Required Secrets

- `ALIGNGPT_SECRET_KEY`
- `ALIGNGPT_DATABASE_URL`
- `ALIGNGPT_REDIS_URL`
- `ALIGNGPT_INFERENCE_API_KEY`
- provider-specific deploy tokens for Vercel, Render, Railway, AWS, or Google Cloud

## Validation Steps

1. Build the API container.
2. Run unit and integration tests.
3. Generate the operational benchmark bundle.
4. Start the API and verify `/health`, `/ready`, `/metrics`, `/v1/status`.
5. Start the frontend and confirm dashboard routes render.
6. Configure deployment secrets in the hosting provider.
7. Deploy staging.
8. Run benchmark pipeline against staging endpoint.
9. Promote only if benchmark regressions are empty and security scans pass.

## Rollback

Rollback should pin the previous container image digest and frontend deployment ID. Benchmark and migration artifacts must be kept with the release record.
