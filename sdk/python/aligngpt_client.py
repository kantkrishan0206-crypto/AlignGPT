"""Minimal Python SDK client for AlignGPT API."""

from __future__ import annotations

import json
from typing import Any
from urllib import request


class AlignGPTClient:
    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        self.base_url = base_url.rstrip("/")

    def align(self, prompt: str, task: str = "chat", metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = json.dumps({"prompt": prompt, "task": task, "metadata": metadata or {}}).encode("utf-8")
        req = request.Request(
            f"{self.base_url}/v1/align",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=30) as response:  # noqa: S310 - caller controls base_url.
            return json.loads(response.read().decode("utf-8"))
