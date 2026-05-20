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
