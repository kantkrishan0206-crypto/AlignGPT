"""Orchestration boundary for AlignGPT request handling."""

from __future__ import annotations

from aligngpt.config import PlatformConfig
from aligngpt.retrieval import InMemoryRetriever
from aligngpt.safety import SafetyPolicy
from aligngpt.schemas import AlignmentRequest, AlignmentResponse


class AlignmentService:
    """Small service facade combining validation, safety, retrieval, and inference routing."""

    def __init__(
        self,
        config: PlatformConfig | None = None,
        safety_policy: SafetyPolicy | None = None,
        retriever: InMemoryRetriever | None = None,
    ) -> None:
        self.config = config or PlatformConfig()
        self.safety_policy = safety_policy or SafetyPolicy()
        self.retriever = retriever or InMemoryRetriever()

    def handle(self, request: AlignmentRequest) -> AlignmentResponse:
        request.validate(max_prompt_chars=self.config.max_prompt_chars)
        findings = self.safety_policy.assess_prompt(request.prompt)
        docs = self.retriever.search(request.prompt) if self.config.enable_retrieval else ()
        citations = tuple(doc.source for doc in docs)
        safe_prompt = self.safety_policy.redact(request.prompt)
        output = self._route_to_backend(safe_prompt, blocked=any(f.severity in {"high", "critical"} for f in findings))
        return AlignmentResponse(
            request_id=request.request_id,
            output=output,
            model_backend=self.config.model_backend,
            safety_findings=findings,
            citations=citations,
            metadata={"environment": self.config.environment, "retrieved_documents": len(docs)},
        )

    def _route_to_backend(self, prompt: str, blocked: bool) -> str:
        if blocked:
            return "Request requires safety review before model execution."
        if self.config.model_backend == "mock":
            return f"AlignGPT mock response for: {prompt}"
        return f"Routed to {self.config.model_backend}: {prompt}"
