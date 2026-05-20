"""FastAPI gateway scaffold for AlignGPT.

FastAPI is optional. Importing this module requires the `api` extra.
"""

from __future__ import annotations

import json

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from aligngpt import AlignmentRequest, AlignmentService, PlatformConfig
from aligngpt.alignment_pipeline import AlignmentEvaluationPipeline
from aligngpt.observability import MetricsRegistry


class AlignPayload(BaseModel):
    prompt: str = Field(min_length=1, max_length=8000)
    task: str = "chat"
    metadata: dict[str, object] = Field(default_factory=dict)


app = FastAPI(title="AlignGPT API", version="0.2.0")
config = PlatformConfig.from_env()
service = AlignmentService(config)
metrics = MetricsRegistry()
pipeline = AlignmentEvaluationPipeline(metrics=metrics)


def require_operator(x_aligngpt_role: str | None = Header(default=None)) -> None:
    """Auth-ready boundary for protected operational endpoints."""

    if config.environment == "development":
        return
    if x_aligngpt_role not in {"researcher", "operator"}:
        raise HTTPException(status_code=403, detail="operator or researcher role required")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "aligngpt"}


@app.get("/ready")
def ready() -> dict[str, object]:
    manifest = pipeline.tracker_manifest()
    return {
        "status": "ready",
        "environment": config.environment,
        "model_backend": config.model_backend,
        "pipeline_config_hash": manifest["config_hash"],
    }


@app.get("/metrics", response_class=PlainTextResponse)
def prometheus_metrics() -> str:
    return metrics.to_prometheus()


@app.post("/v1/align")
def align(payload: AlignPayload) -> dict[str, object]:
    response = service.handle(
        AlignmentRequest(prompt=payload.prompt, task=payload.task, metadata=payload.metadata)
    )
    return {
        "request_id": response.request_id,
        "output": response.output,
        "model_backend": response.model_backend,
        "safety_findings": [finding.__dict__ for finding in response.safety_findings],
        "citations": list(response.citations),
        "metadata": response.metadata,
        "created_at": response.created_at,
    }


@app.post("/v1/evaluate")
def evaluate(payload: AlignPayload) -> dict[str, object]:
    result = pipeline.run(
        AlignmentRequest(prompt=payload.prompt, task=payload.task, metadata=payload.metadata)
    )
    return result.to_dict()


@app.get("/v1/status")
def platform_status() -> dict[str, object]:
    snapshot = metrics.snapshot()
    return {
        "service": "aligngpt",
        "environment": config.environment,
        "model_backend": config.model_backend,
        "metrics": snapshot,
        "pipeline": pipeline.tracker_manifest(),
    }


@app.get("/v1/events")
def event_stream() -> StreamingResponse:
    async def stream():
        events = [
            {"event": "health", "status": "ready"},
            {"event": "router", "backend": "vllm-a10g-primary"},
            {"event": "benchmark", "status": "passing"},
        ]
        for event in events:
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/v1/admin/reload")
def reload_runtime(x_aligngpt_role: str | None = Header(default=None)) -> dict[str, str]:
    require_operator(x_aligngpt_role)
    return {"status": "accepted", "message": "runtime reload hook ready"}
