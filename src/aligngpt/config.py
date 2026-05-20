"""Configuration loading for AlignGPT platform components."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PlatformConfig:
    """Minimal runtime configuration shared by API, CLI, and tests."""

    environment: str = "development"
    service_name: str = "aligngpt"
    log_level: str = "INFO"
    model_backend: str = "mock"
    safety_profile: str = "standard"
    enable_retrieval: bool = True
    request_timeout_seconds: float = 30.0
    max_prompt_chars: int = 8000
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "PlatformConfig":
        """Build config from environment variables without requiring external files."""

        return cls(
            environment=os.getenv("ALIGNGPT_ENV", "development"),
            service_name=os.getenv("ALIGNGPT_SERVICE_NAME", "aligngpt"),
            log_level=os.getenv("ALIGNGPT_LOG_LEVEL", "INFO"),
            model_backend=os.getenv("ALIGNGPT_MODEL_BACKEND", "mock"),
            safety_profile=os.getenv("ALIGNGPT_SAFETY_PROFILE", "standard"),
            enable_retrieval=_parse_bool(os.getenv("ALIGNGPT_ENABLE_RETRIEVAL", "true")),
            request_timeout_seconds=float(os.getenv("ALIGNGPT_REQUEST_TIMEOUT_SECONDS", "30")),
            max_prompt_chars=int(os.getenv("ALIGNGPT_MAX_PROMPT_CHARS", "8000")),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "PlatformConfig":
        """Load JSON or YAML config files with a clear error for unsupported formats."""

        path = Path(path)
        payload = _read_config_file(path)
        known = {field.name for field in cls.__dataclass_fields__.values()}
        kwargs = {key: value for key, value in payload.items() if key in known}
        extra = {key: value for key, value in payload.items() if key not in known}
        metadata = dict(kwargs.get("metadata") or {})
        metadata.update(extra)
        kwargs["metadata"] = metadata
        return cls(**kwargs)


def _read_config_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - depends on optional env state
            raise RuntimeError("Install PyYAML to load YAML configuration files.") from exc
        loaded = yaml.safe_load(text) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Expected mapping in {path}")
        return loaded
    raise ValueError(f"Unsupported config format: {path.suffix}")


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}
