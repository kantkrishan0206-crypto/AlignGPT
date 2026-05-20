# Backend

Backend services expose AI workflows through stable interfaces:

- `api_gateway/`: HTTP boundary and request validation.
- `grpc_services/`: future low-latency service interfaces.
- `auth/`: authentication and authorization policy.
- `rate_limiting/`: quotas and abuse protection.
- `websocket/`: streaming events and responses.
- `retrieval/`: RAG query service.
- `caching/`: cache policy and invalidation.
- `inference_router/`: model backend routing.
- `queue_system/`: asynchronous jobs.
- `feature_store/`: online/offline features.
- `database/`: schema and migrations.
