# Render API Deployment

Render can deploy the FastAPI gateway from `render.yaml`.

## Steps

1. Connect the GitHub repository.
2. Create the web service from `render.yaml`.
3. Configure `ALIGNGPT_SECRET_KEY`, `ALIGNGPT_DATABASE_URL`, `ALIGNGPT_REDIS_URL`, and inference provider keys.
4. Verify `/health`, `/ready`, `/metrics`, `/v1/status`.
5. Run the benchmark pipeline against the Render URL before promotion.
