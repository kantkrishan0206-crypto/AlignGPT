"""FastAPI gateway scaffold for AlignGPT.

FastAPI is optional. Importing this module requires the `api` extra.
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from aligngpt import AlignmentRequest, AlignmentService, PlatformConfig


class AlignPayload(BaseModel):
    prompt: str = Field(min_length=1, max_length=8000)
    task: str = "chat"
    metadata: dict[str, object] = Field(default_factory=dict)


app = FastAPI(title="AlignGPT API", version="0.2.0")
service = AlignmentService(PlatformConfig.from_env())


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "aligngpt"}


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
