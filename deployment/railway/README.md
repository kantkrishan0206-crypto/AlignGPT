# Railway API Deployment

Railway can deploy the API using the root `railway.toml` and Dockerfile.

Provision PostgreSQL and Redis plugins, then set:

- `ALIGNGPT_DATABASE_URL`
- `ALIGNGPT_REDIS_URL`
- `ALIGNGPT_SECRET_KEY`
- `ALIGNGPT_INFERENCE_API_KEY`

After deployment, validate readiness and metrics endpoints before connecting the frontend.
