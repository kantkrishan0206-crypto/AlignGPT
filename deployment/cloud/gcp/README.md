# Google Cloud Deployment Path

Recommended first Google Cloud target:

- Cloud Run for the API container.
- Artifact Registry for images.
- Cloud SQL PostgreSQL for metadata.
- Memorystore Redis for queues and event streams.
- Cloud Monitoring for metrics and alerting.

GPU inference should run separately on GKE GPU nodes or Vertex AI endpoints and be registered in the inference router.
