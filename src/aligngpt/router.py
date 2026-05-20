"""GPU-aware inference routing for AlignGPT.

The router is intentionally import-safe: it models capacity, quantization,
health, latency, and fallback decisions without importing CUDA or model-serving
libraries. Production adapters can feed it live telemetry from vLLM, TGI,
Ollama, hosted APIs, or custom GPU workers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


BackendHealth = Literal["healthy", "degraded", "offline"]
Quantization = Literal["fp16", "bf16", "int8", "int4", "cpu"]


@dataclass(frozen=True)
class InferenceRequestProfile:
    """Routing inputs derived from a request before backend selection."""

    task: str = "chat"
    prompt_tokens: int = 0
    expected_output_tokens: int = 256
    batch_size: int = 1
    latency_budget_ms: int = 2500
    required_capabilities: tuple[str, ...] = ("chat",)
    risk_level: Literal["low", "medium", "high"] = "low"

    @property
    def estimated_total_tokens(self) -> int:
        return self.prompt_tokens + self.expected_output_tokens


@dataclass(frozen=True)
class ModelBackend:
    """Serving backend capacity profile."""

    name: str
    backend_type: Literal["local", "vllm", "hosted", "cpu", "mock"]
    capabilities: tuple[str, ...]
    gpu_memory_gb: float
    reserved_memory_gb: float
    max_batch_size: int
    context_window: int
    quantization: Quantization
    average_latency_ms: int
    tokens_per_second: float
    health: BackendHealth = "healthy"
    priority: int = 100
    cost_per_1k_tokens: float = 0.0
    fallback_for: tuple[str, ...] = ()

    @property
    def available_memory_gb(self) -> float:
        return max(0.0, self.gpu_memory_gb - self.reserved_memory_gb)


@dataclass(frozen=True)
class RoutingDecision:
    """Explainable backend decision emitted for tracing and dashboards."""

    backend_name: str
    fallback_chain: tuple[str, ...]
    reason: str
    score: float
    estimated_latency_ms: int
    estimated_memory_gb: float
    microbatch_size: int
    quantization: Quantization
    degraded: bool = False


@dataclass
class GpuAwareInferenceRouter:
    """Selects a backend using capacity, health, cost, and risk constraints."""

    backends: tuple[ModelBackend, ...] = field(default_factory=tuple)
    memory_per_1k_tokens_gb: float = 0.18

    def select_backend(self, profile: InferenceRequestProfile) -> RoutingDecision:
        candidates = [backend for backend in self.backends if self._supports(backend, profile)]
        if not candidates:
            raise ValueError(f"No backend supports task={profile.task!r} with {profile.required_capabilities}")

        scored = [(self._score_backend(backend, profile), backend) for backend in candidates]
        scored.sort(key=lambda item: item[0], reverse=True)
        best_score, best = scored[0]
        estimated_memory = self.estimate_memory_gb(profile)
        estimated_latency = self.estimate_latency_ms(best, profile)
        microbatch_size = max(1, min(profile.batch_size, best.max_batch_size))
        fallback_chain = self._fallback_chain(best, candidates)
        reason = self._reason(best, profile, estimated_memory, estimated_latency)
        return RoutingDecision(
            backend_name=best.name,
            fallback_chain=fallback_chain,
            reason=reason,
            score=round(best_score, 4),
            estimated_latency_ms=estimated_latency,
            estimated_memory_gb=round(estimated_memory, 4),
            microbatch_size=microbatch_size,
            quantization=best.quantization,
            degraded=best.health == "degraded",
        )

    def create_microbatches(self, total_items: int, microbatch_size: int) -> tuple[tuple[int, int], ...]:
        if total_items <= 0:
            return ()
        if microbatch_size <= 0:
            raise ValueError("microbatch_size must be positive")
        ranges = []
        for start in range(0, total_items, microbatch_size):
            ranges.append((start, min(total_items, start + microbatch_size)))
        return tuple(ranges)

    def estimate_memory_gb(self, profile: InferenceRequestProfile) -> float:
        return (
            profile.estimated_total_tokens
            / 1000
            * self.memory_per_1k_tokens_gb
            * max(1, profile.batch_size)
        )

    def estimate_latency_ms(self, backend: ModelBackend, profile: InferenceRequestProfile) -> int:
        generation_ms = int((profile.expected_output_tokens / max(1.0, backend.tokens_per_second)) * 1000)
        batch_penalty = max(0, profile.batch_size - backend.max_batch_size) * 200
        health_penalty = 350 if backend.health == "degraded" else 0
        return backend.average_latency_ms + generation_ms + batch_penalty + health_penalty

    def _supports(self, backend: ModelBackend, profile: InferenceRequestProfile) -> bool:
        if backend.health == "offline":
            return False
        if profile.task not in backend.capabilities and not set(profile.required_capabilities).issubset(
            set(backend.capabilities)
        ):
            return False
        if profile.estimated_total_tokens > backend.context_window:
            return False
        if self.estimate_memory_gb(profile) > backend.available_memory_gb and backend.backend_type != "hosted":
            return False
        if profile.risk_level == "high" and backend.backend_type == "mock":
            return False
        return True

    def _score_backend(self, backend: ModelBackend, profile: InferenceRequestProfile) -> float:
        latency = self.estimate_latency_ms(backend, profile)
        latency_fit = max(0.0, 1.0 - (latency / max(1, profile.latency_budget_ms * 2)))
        memory_headroom = 1.0
        if backend.backend_type != "hosted":
            memory_headroom = min(1.0, backend.available_memory_gb / max(0.1, self.estimate_memory_gb(profile)))
        quantization_bonus = {"bf16": 0.08, "fp16": 0.06, "int8": 0.04, "int4": 0.02, "cpu": -0.05}[
            backend.quantization
        ]
        health_penalty = 0.15 if backend.health == "degraded" else 0.0
        dev_penalty = 0.65 if backend.backend_type == "mock" else 0.0
        cost_penalty = min(0.2, backend.cost_per_1k_tokens / 10)
        priority_bonus = max(0.0, (100 - backend.priority) / 1000)
        return (
            latency_fit
            + memory_headroom
            + quantization_bonus
            + priority_bonus
            - health_penalty
            - cost_penalty
            - dev_penalty
        )

    def _fallback_chain(
        self, selected: ModelBackend, candidates: list[ModelBackend]
    ) -> tuple[str, ...]:
        fallbacks = [name for name in selected.fallback_for if any(candidate.name == name for candidate in candidates)]
        additional = [
            candidate.name
            for candidate in candidates
            if candidate.name != selected.name and candidate.name not in fallbacks
        ]
        return tuple(fallbacks + additional)

    def _reason(
        self,
        backend: ModelBackend,
        profile: InferenceRequestProfile,
        estimated_memory: float,
        estimated_latency: int,
    ) -> str:
        return (
            f"{backend.name} selected for {profile.task}: "
            f"{backend.quantization} backend, {estimated_memory:.2f}GB estimated memory, "
            f"{estimated_latency}ms estimated latency, health={backend.health}"
        )


DEFAULT_BACKENDS = (
    ModelBackend(
        name="vllm-a10g-primary",
        backend_type="vllm",
        capabilities=("chat", "rag", "reward_scoring"),
        gpu_memory_gb=24.0,
        reserved_memory_gb=7.0,
        max_batch_size=8,
        context_window=8192,
        quantization="fp16",
        average_latency_ms=420,
        tokens_per_second=96,
        priority=10,
        cost_per_1k_tokens=0.35,
        fallback_for=("hosted-alignment-api", "cpu-safety-fallback"),
    ),
    ModelBackend(
        name="hosted-alignment-api",
        backend_type="hosted",
        capabilities=("chat", "rag", "reward_scoring"),
        gpu_memory_gb=0.0,
        reserved_memory_gb=0.0,
        max_batch_size=16,
        context_window=16384,
        quantization="bf16",
        average_latency_ms=900,
        tokens_per_second=64,
        priority=30,
        cost_per_1k_tokens=2.4,
        fallback_for=("cpu-safety-fallback",),
    ),
    ModelBackend(
        name="cpu-safety-fallback",
        backend_type="cpu",
        capabilities=("chat", "rag"),
        gpu_memory_gb=0.0,
        reserved_memory_gb=0.0,
        max_batch_size=2,
        context_window=4096,
        quantization="cpu",
        average_latency_ms=1800,
        tokens_per_second=18,
        priority=80,
        cost_per_1k_tokens=0.05,
    ),
    ModelBackend(
        name="mock-local-dev",
        backend_type="mock",
        capabilities=("chat", "rag", "reward_scoring"),
        gpu_memory_gb=1.0,
        reserved_memory_gb=0.1,
        max_batch_size=4,
        context_window=4096,
        quantization="cpu",
        average_latency_ms=80,
        tokens_per_second=240,
        priority=95,
        cost_per_1k_tokens=0.0,
    ),
)
