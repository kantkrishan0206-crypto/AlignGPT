"""Typed data contracts used across AlignGPT services."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4


RiskLevel = Literal["low", "medium", "high", "critical"]


@dataclass(frozen=True)
class AlignmentRequest:
    """Request contract for a model interaction or evaluation probe."""

    prompt: str
    user_id: str | None = None
    request_id: str = field(default_factory=lambda: str(uuid4()))
    task: str = "chat"
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self, max_prompt_chars: int = 8000) -> None:
        if not self.prompt or not self.prompt.strip():
            raise ValueError("prompt must be non-empty")
        if len(self.prompt) > max_prompt_chars:
            raise ValueError(f"prompt exceeds {max_prompt_chars} characters")


@dataclass(frozen=True)
class SafetyFinding:
    """A policy finding emitted by prompt, response, or trace safety checks."""

    category: str
    severity: RiskLevel
    message: str
    rule_id: str
    span: tuple[int, int] | None = None


@dataclass(frozen=True)
class AlignmentResponse:
    """Response contract returned by the orchestration service."""

    request_id: str
    output: str
    model_backend: str
    safety_findings: tuple[SafetyFinding, ...] = ()
    citations: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


@dataclass(frozen=True)
class EvalMetric:
    """Single metric value with provenance."""

    name: str
    value: float
    higher_is_better: bool = True
    description: str = ""


@dataclass(frozen=True)
class BenchmarkResult:
    """Result emitted by deterministic benchmark and smoke-test suites."""

    benchmark_name: str
    metrics: tuple[EvalMetric, ...]
    passed: bool
    threshold_summary: str
    metadata: dict[str, Any] = field(default_factory=dict)
