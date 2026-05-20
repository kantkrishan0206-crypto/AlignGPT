"""End-to-end alignment evaluation pipeline.

This module is the lightweight operational spine of AlignGPT: prompt safety,
retrieval, GPU-aware routing, deterministic local inference, reward scoring,
metric generation, tracing, and exportable reports.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

from aligngpt.evaluation import exact_contains, lexical_diversity
from aligngpt.observability import MetricsRegistry, RunEvent, config_fingerprint
from aligngpt.retrieval import InMemoryRetriever, RetrievedDocument
from aligngpt.router import (
    DEFAULT_BACKENDS,
    GpuAwareInferenceRouter,
    InferenceRequestProfile,
    RoutingDecision,
)
from aligngpt.safety import SafetyPolicy
from aligngpt.schemas import AlignmentRequest, EvalMetric, SafetyFinding


@dataclass(frozen=True)
class AlignmentEvaluationResult:
    request_id: str
    prompt: str
    output: str
    reward_score: float
    route: RoutingDecision
    metrics: tuple[EvalMetric, ...]
    safety_findings: tuple[SafetyFinding, ...]
    citations: tuple[str, ...]
    trace_events: tuple[RunEvent, ...]
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "prompt": self.prompt,
            "output": self.output,
            "reward_score": self.reward_score,
            "route": asdict(self.route),
            "metrics": [asdict(metric) for metric in self.metrics],
            "safety_findings": [asdict(finding) for finding in self.safety_findings],
            "citations": list(self.citations),
            "trace_events": [json.loads(event.to_json()) for event in self.trace_events],
            "created_at": self.created_at,
        }


class DeterministicAlignmentModel:
    """A deterministic inference adapter for local demos, CI, and benchmark artifacts."""

    def generate(
        self,
        prompt: str,
        documents: tuple[RetrievedDocument, ...],
        route: RoutingDecision,
    ) -> str:
        context = " ".join(doc.text for doc in documents[:2])
        if context:
            return (
                f"AlignGPT routed via {route.backend_name}. "
                f"Grounded answer: {prompt.strip()} | Evidence: {context[:240]}"
            )
        return f"AlignGPT routed via {route.backend_name}. Grounded answer: {prompt.strip()}"


class HeuristicRewardScorer:
    """Reward scorer that favors grounded, specific, safe responses."""

    def score(
        self,
        prompt: str,
        output: str,
        documents: tuple[RetrievedDocument, ...],
        findings: tuple[SafetyFinding, ...],
    ) -> float:
        prompt_terms = tuple(term for term in prompt.lower().split() if len(term) > 4)
        grounding_terms = tuple(term for doc in documents for term in doc.text.lower().split() if len(term) > 4)
        prompt_coverage = exact_contains(output, prompt_terms[:8])
        grounding_coverage = exact_contains(output, grounding_terms[:10]) if grounding_terms else 0.5
        diversity = lexical_diversity(output)
        safety_penalty = 0.18 * sum(1 for finding in findings if finding.severity in {"high", "critical"})
        score = 0.45 * prompt_coverage + 0.35 * grounding_coverage + 0.2 * diversity - safety_penalty
        return round(max(0.0, min(1.0, score)), 4)


class AlignmentEvaluationPipeline:
    """Runnable vertical flow for alignment evaluation and report generation."""

    def __init__(
        self,
        retriever: InMemoryRetriever | None = None,
        safety_policy: SafetyPolicy | None = None,
        router: GpuAwareInferenceRouter | None = None,
        model: DeterministicAlignmentModel | None = None,
        reward_scorer: HeuristicRewardScorer | None = None,
        metrics: MetricsRegistry | None = None,
    ) -> None:
        self.retriever = retriever or _default_retriever()
        self.safety_policy = safety_policy or SafetyPolicy()
        self.router = router or GpuAwareInferenceRouter(DEFAULT_BACKENDS)
        self.model = model or DeterministicAlignmentModel()
        self.reward_scorer = reward_scorer or HeuristicRewardScorer()
        self.metrics = metrics or MetricsRegistry()

    def run(self, request: AlignmentRequest) -> AlignmentEvaluationResult:
        request.validate()
        trace: list[RunEvent] = []
        trace.append(RunEvent("request.received", {"task": request.task}, request.request_id))

        findings = self.safety_policy.assess_prompt(request.prompt)
        redacted_prompt = self.safety_policy.redact(request.prompt)
        trace.append(
            RunEvent(
                "safety.assessed",
                {"findings": len(findings), "redacted": redacted_prompt != request.prompt},
                request.request_id,
            )
        )

        documents = self.retriever.search(redacted_prompt, limit=4)
        trace.append(RunEvent("retrieval.completed", {"documents": len(documents)}, request.request_id))

        profile = self._profile_request(redacted_prompt, request.task, findings)
        route = self.router.select_backend(profile)
        trace.append(RunEvent("routing.selected", asdict(route), request.request_id))

        blocked = any(finding.severity in {"high", "critical"} for finding in findings)
        output = (
            "Request requires safety review before model execution."
            if blocked
            else self.model.generate(redacted_prompt, documents, route)
        )
        response_findings = self.safety_policy.assess_prompt(output)
        all_findings = tuple(findings + response_findings)
        reward = self.reward_scorer.score(redacted_prompt, output, documents, all_findings)
        metrics = self._metrics(output, reward, route, documents, all_findings)
        trace.append(
            RunEvent(
                "evaluation.completed",
                {"reward_score": reward, "metric_count": len(metrics), "blocked": blocked},
                request.request_id,
            )
        )

        self.metrics.increment("aligngpt_requests_total", {"task": request.task})
        self.metrics.observe("aligngpt_estimated_latency_ms", route.estimated_latency_ms, {"backend": route.backend_name})
        self.metrics.observe("aligngpt_reward_score", reward, {"backend": route.backend_name})
        if all_findings:
            self.metrics.increment("aligngpt_safety_findings_total", {"severity": all_findings[0].severity})

        return AlignmentEvaluationResult(
            request_id=request.request_id,
            prompt=redacted_prompt,
            output=output,
            reward_score=reward,
            route=route,
            metrics=metrics,
            safety_findings=all_findings,
            citations=tuple(doc.source for doc in documents),
            trace_events=tuple(trace),
        )

    def export_report(self, result: AlignmentEvaluationResult, output_dir: str | Path) -> dict[str, Path]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"alignment_eval_{result.request_id}"
        json_path = output_dir / f"{stem}.json"
        md_path = output_dir / f"{stem}.md"
        payload = result.to_dict()
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        md_path.write_text(_markdown_report(payload), encoding="utf-8")
        return {"json": json_path, "markdown": md_path}

    def tracker_manifest(self) -> dict[str, Any]:
        config = {
            "router_backends": [backend.name for backend in self.router.backends],
            "retriever": "in_memory_keyword",
            "reward_scorer": "heuristic_grounding",
            "safety_policy": "regex_standard",
        }
        return {
            "tracker": "aligngpt-local",
            "config_hash": config_fingerprint(config),
            "config": config,
            "created_at": datetime.now(UTC).isoformat(),
        }

    def _profile_request(
        self, prompt: str, task: str, findings: tuple[SafetyFinding, ...]
    ) -> InferenceRequestProfile:
        risk = "high" if any(finding.severity in {"high", "critical"} for finding in findings) else "low"
        prompt_tokens = max(1, len(prompt.split()))
        return InferenceRequestProfile(
            task="rag" if task in {"chat", "evaluate"} else task,
            prompt_tokens=prompt_tokens,
            expected_output_tokens=220,
            batch_size=1,
            latency_budget_ms=2500,
            required_capabilities=("rag",),
            risk_level=risk,
        )

    def _metrics(
        self,
        output: str,
        reward: float,
        route: RoutingDecision,
        documents: tuple[RetrievedDocument, ...],
        findings: tuple[SafetyFinding, ...],
    ) -> tuple[EvalMetric, ...]:
        return (
            EvalMetric("reward_score", reward, True, "Heuristic grounded preference score."),
            EvalMetric("lexical_diversity", lexical_diversity(output), True, "Unique-token ratio."),
            EvalMetric("retrieved_documents", float(len(documents)), True, "Retrieved context count."),
            EvalMetric("safety_findings", float(len(findings)), False, "Prompt and response findings."),
            EvalMetric("estimated_latency_ms", float(route.estimated_latency_ms), False, "Router estimate."),
        )


def _default_retriever() -> InMemoryRetriever:
    return InMemoryRetriever(
        (
            RetrievedDocument(
                document_id="align-001",
                text="reward modeling alignment safety evaluation preference optimization",
                source="docs/reward_model.md",
                score=0.95,
            ),
            RetrievedDocument(
                document_id="bench-001",
                text="benchmark reproducibility latency throughput robustness hallucination adversarial bias",
                source="docs/benchmarks/README.md",
                score=0.9,
            ),
            RetrievedDocument(
                document_id="ops-001",
                text="deployment observability prometheus grafana tracing redis postgres kubernetes",
                source="docs/deployment/production_runbook.md",
                score=0.88,
            ),
        )
    )


def _markdown_report(payload: dict[str, Any]) -> str:
    metrics = "\n".join(f"- {m['name']}: {m['value']}" for m in payload["metrics"])
    findings = "\n".join(
        f"- {f['severity']} {f['category']} ({f['rule_id']}): {f['message']}"
        for f in payload["safety_findings"]
    ) or "- none"
    return f"""# Alignment Evaluation Report

Request: `{payload['request_id']}`

## Output

{payload['output']}

## Routing

- Backend: {payload['route']['backend_name']}
- Reason: {payload['route']['reason']}
- Fallback chain: {', '.join(payload['route']['fallback_chain']) or 'none'}

## Metrics

{metrics}

## Safety Findings

{findings}

## Citations

{chr(10).join(f'- {citation}' for citation in payload['citations']) or '- none'}
"""
