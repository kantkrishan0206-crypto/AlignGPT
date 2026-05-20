"""Structured observability helpers for reproducible AI runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
import hashlib
import json
from typing import Any


@dataclass(frozen=True)
class RunEvent:
    event_type: str
    payload: dict[str, Any]
    trace_id: str
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_json(self) -> str:
        return json.dumps(
            {
                "event_type": self.event_type,
                "payload": self.payload,
                "trace_id": self.trace_id,
                "timestamp": self.timestamp,
            },
            sort_keys=True,
        )


def config_fingerprint(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


@dataclass
class MetricsRegistry:
    """Tiny in-memory metrics registry with Prometheus text exposition."""

    counters: dict[tuple[str, tuple[tuple[str, str], ...]], float] = field(default_factory=dict)
    observations: dict[tuple[str, tuple[tuple[str, str], ...]], list[float]] = field(default_factory=dict)

    def increment(self, name: str, labels: dict[str, str] | None = None, value: float = 1.0) -> None:
        key = self._key(name, labels)
        self.counters[key] = self.counters.get(key, 0.0) + value

    def observe(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        key = self._key(name, labels)
        self.observations.setdefault(key, []).append(float(value))

    def snapshot(self) -> dict[str, Any]:
        return {
            "counters": [
                {"name": name, "labels": dict(labels), "value": value}
                for (name, labels), value in sorted(self.counters.items())
            ],
            "observations": [
                {
                    "name": name,
                    "labels": dict(labels),
                    "count": len(values),
                    "avg": sum(values) / max(1, len(values)),
                    "max": max(values) if values else 0.0,
                }
                for (name, labels), values in sorted(self.observations.items())
            ],
        }

    def to_prometheus(self) -> str:
        lines: list[str] = []
        for (name, labels), value in sorted(self.counters.items()):
            lines.append(f"{name}{self._labels(labels)} {value}")
        for (name, labels), values in sorted(self.observations.items()):
            label_text = self._labels(labels)
            lines.append(f"{name}_count{label_text} {len(values)}")
            lines.append(f"{name}_sum{label_text} {sum(values)}")
            lines.append(f"{name}_max{label_text} {max(values) if values else 0.0}")
        return "\n".join(lines) + ("\n" if lines else "")

    def _key(self, name: str, labels: dict[str, str] | None) -> tuple[str, tuple[tuple[str, str], ...]]:
        return name, tuple(sorted((labels or {}).items()))

    def _labels(self, labels: tuple[tuple[str, str], ...]) -> str:
        if not labels:
            return ""
        return "{" + ",".join(f'{key}="{value}"' for key, value in labels) + "}"
