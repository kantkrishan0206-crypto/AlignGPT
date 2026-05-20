# AWS Deployment Path

Recommended first AWS target:

- ECS Fargate for the FastAPI API container.
- ECR for images.
- RDS PostgreSQL for run and benchmark metadata.
- ElastiCache Redis for queues, cache, and events.
- Application Load Balancer with `/health` checks.
- Managed Prometheus/Grafana or self-hosted observability stack.

GPU-backed inference can run on ECS EC2 GPU capacity, EKS GPU nodes, or a separate vLLM service.
