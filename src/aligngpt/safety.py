"""Starter safety policies for prompt handling and trace hygiene."""

from __future__ import annotations

from dataclasses import dataclass, field
import re

from aligngpt.schemas import SafetyFinding


DEFAULT_INJECTION_PATTERNS = {
    "ignore_previous": r"\b(ignore|forget|override)\b.{0,40}\b(previous|prior|system)\b",
    "secret_request": r"\b(api key|secret|token|password|private key)\b",
    "tool_override": r"\b(disable|bypass|turn off)\b.{0,40}\b(safety|guardrail|policy|filter)\b",
}

DEFAULT_PII_PATTERNS = {
    "email": r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
    "phone": r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b",
}


@dataclass(frozen=True)
class SafetyPolicy:
    """Regex-backed safety policy suitable for tests and early API scaffolds."""

    injection_patterns: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_INJECTION_PATTERNS))
    pii_patterns: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_PII_PATTERNS))

    def assess_prompt(self, prompt: str) -> tuple[SafetyFinding, ...]:
        findings: list[SafetyFinding] = []
        for rule_id, pattern in self.injection_patterns.items():
            match = re.search(pattern, prompt, flags=re.IGNORECASE | re.DOTALL)
            if match:
                findings.append(
                    SafetyFinding(
                        category="prompt_injection",
                        severity="high",
                        message="Prompt appears to request policy or instruction override.",
                        rule_id=rule_id,
                        span=match.span(),
                    )
                )
        for rule_id, pattern in self.pii_patterns.items():
            match = re.search(pattern, prompt, flags=re.IGNORECASE)
            if match:
                findings.append(
                    SafetyFinding(
                        category="pii",
                        severity="medium",
                        message="Prompt contains data that should be redacted from logs.",
                        rule_id=rule_id,
                        span=match.span(),
                    )
                )
        return tuple(findings)

    def redact(self, text: str) -> str:
        redacted = text
        for rule_id, pattern in self.pii_patterns.items():
            redacted = re.sub(pattern, f"[REDACTED_{rule_id.upper()}]", redacted, flags=re.IGNORECASE)
        return redacted
