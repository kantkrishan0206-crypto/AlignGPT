"""Scientific metric helpers for alignment experiments."""

from __future__ import annotations


def win_rate(chosen_wins: int, total_pairs: int) -> float:
    if total_pairs <= 0:
        raise ValueError("total_pairs must be positive")
    return chosen_wins / total_pairs


def calibration_error(predicted: list[float], observed: list[float]) -> float:
    if len(predicted) != len(observed):
        raise ValueError("predicted and observed must have the same length")
    if not predicted:
        raise ValueError("inputs must be non-empty")
    return sum(abs(p - o) for p, o in zip(predicted, observed, strict=True)) / len(predicted)
