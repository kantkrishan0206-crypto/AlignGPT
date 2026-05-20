"""Orchestration boundary for AlignGPT request handling."""

from __future__ import annotations

from aligngpt.config import PlatformConfig
from aligngpt.router import DEFAULT_BACKENDS, GpuAwareInferenceRouter, InferenceRequestProfile
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
        self.router = GpuAwareInferenceRouter(DEFAULT_BACKENDS)

    def handle(self, request: AlignmentRequest) -> AlignmentResponse:
        request.validate(max_prompt_chars=self.config.max_prompt_chars)
        findings = self.safety_policy.assess_prompt(request.prompt)
        docs = self.retriever.search(request.prompt) if self.config.enable_retrieval else ()
        citations = tuple(doc.source for doc in docs)
        safe_prompt = self.safety_policy.redact(request.prompt)
        route = self.router.select_backend(
            InferenceRequestProfile(
                task="rag" if request.task in {"chat", "evaluate"} else request.task,
                prompt_tokens=max(1, len(safe_prompt.split())),
                expected_output_tokens=220,
                required_capabilities=("rag",),
                risk_level="high" if any(f.severity in {"high", "critical"} for f in findings) else "low",
            )
        )
        output = self._route_to_backend(
            safe_prompt,
            blocked=any(f.severity in {"high", "critical"} for f in findings),
            backend_name=route.backend_name,
        )
        return AlignmentResponse(
            request_id=request.request_id,
            output=output,
            model_backend=route.backend_name,
            safety_findings=findings,
            citations=citations,
            metadata={
                "environment": self.config.environment,
                "retrieved_documents": len(docs),
                "route": route.__dict__,
            },
        )

    def _route_to_backend(self, prompt: str, blocked: bool, backend_name: str) -> str:
        if blocked:
            return "Request requires safety review before model execution."
        return f"Routed via {backend_name}: {prompt}"
