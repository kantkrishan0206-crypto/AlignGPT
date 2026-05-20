"""Deterministic evaluation utilities for lightweight CI and smoke tests."""

from __future__ import annotations

import statistics
from typing import Iterable

from aligngpt.schemas import EvalMetric


def lexical_diversity(text: str) -> float:
    tokens = _tokens(text)
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def length_ratio(output: str, reference: str) -> float:
    ref_len = max(1, len(_tokens(reference)))
    return len(_tokens(output)) / ref_len


def exact_contains(output: str, required_terms: Iterable[str]) -> float:
    normalized = output.casefold()
    terms = [term.casefold() for term in required_terms]
    if not terms:
        return 1.0
    hits = sum(1 for term in terms if term in normalized)
    return hits / len(terms)


def summarize_output(output: str, reference: str | None = None) -> tuple[EvalMetric, ...]:
    metrics = [
        EvalMetric("lexical_diversity", lexical_diversity(output), True, "Unique-token ratio."),
        EvalMetric("token_count", float(len(_tokens(output))), False, "Whitespace token count."),
    ]
    if reference is not None:
        metrics.append(
            EvalMetric("length_ratio", length_ratio(output, reference), False, "Output/reference token ratio.")
        )
    return tuple(metrics)


def aggregate_metrics(rows: Iterable[tuple[EvalMetric, ...]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        for metric in row:
            grouped.setdefault(metric.name, []).append(metric.value)
    return {name: statistics.mean(values) for name, values in grouped.items()}


def _tokens(text: str) -> list[str]:
    return [token for token in text.strip().split() if token]
