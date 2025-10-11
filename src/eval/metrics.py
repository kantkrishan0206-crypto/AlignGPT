#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluation metrics for language model outputs.
- Perplexity (approximate via causal LM loss)
- BLEU/ROUGE against references (optional)
- Diversity metrics (distinct-n, type-token ratio)
- Length and repetition heuristics
- Reward scoring integration (optional)
- Batch utilities for scoring lists of generations

This module is framework-light and can be extended to include:
- Toxicity filters (placeholder hooks)
- Factuality checks (stub)
- Preference score aggregation across ensembles
"""

import math
import statistics
from typing import List, Dict, Optional, Any, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Optional rouge_bleu imports guarded
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    _BLEU_AVAILABLE = True
except Exception:
    _BLEU_AVAILABLE = False

try:
    from rouge_score import rouge_scorer
    _ROUGE_AVAILABLE = True
except Exception:
    _ROUGE_AVAILABLE = False


def _safe_len(text: str) -> int:
    return len(text.strip().split())


def _distinct_n(tokens: List[str], n: int) -> float:
    if len(tokens) < n:
        return 0.0
    ngrams = set(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))
    return len(ngrams) / max(1, len(tokens) - n + 1)


def _type_token_ratio(tokens: List[str]) -> float:
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def _repeat_heuristics(tokens: List[str], window: int = 5) -> float:
    """
    Simple repetition score: fraction of windows that repeat exactly.
    """
    if len(tokens) < window * 2:
        return 0.0
    repeats = 0
    total = 0
    for i in range(len(tokens) - 2 * window + 1):
        total += 1
        if tokens[i:i+window] == tokens[i+window:i+2*window]:
            repeats += 1
    return repeats / max(1, total)


def compute_perplexity(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: Optional[List[str]],
    sample_size: int = 128,
    max_length: int = 512,
) -> float:
    """
    Approximate perplexity by computing average negative log likelihood over a sample of texts.
    If texts is None, caller should pass a representative dataset elsewhere.
    """
    device = next(model.parameters()).device
    if not texts or len(texts) == 0:
        raise ValueError("compute_perplexity requires non-empty texts list or a dataset integration")

    sub = texts[: min(sample_size, len(texts))]
    nlls = []
    model.eval()
    with torch.no_grad():
        for t in sub:
            enc = tokenizer(t, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt").to(device)
            labels = enc["input_ids"]
            out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], labels=labels)
            # loss is mean over tokens; convert to NLL per token
            nll = float(out.loss.item())
            nlls.append(nll)
    avg_nll = statistics.mean(nlls) if nlls else float("inf")
    # Perplexity = exp(NLL)
    return math.exp(avg_nll)


def compute_bleu(hyp: str, ref: str) -> Optional[float]:
    if not _BLEU_AVAILABLE:
        return None
    smoothie = SmoothingFunction().method1
    hyp_tokens = hyp.strip().split()
    ref_tokens = ref.strip().split()
    try:
        score = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothie)
        return float(score)
    except Exception:
        return None


def compute_rouge(hyp: str, ref: str) -> Optional[Dict[str, float]]:
    if not _ROUGE_AVAILABLE:
        return None
    try:
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = scorer.score(ref, hyp)
        return {k: float(v.fmeasure) for k, v in scores.items()}
    except Exception:
        return None


def compute_diversity(text: str) -> Dict[str, float]:
    """
    Return distinct-1, distinct-2, and type-token ratio.
    """
    toks = text.strip().split()
    return {
        "distinct1": _distinct_n(toks, 1),
        "distinct2": _distinct_n(toks, 2),
        "ttr": _type_token_ratio(toks),
    }


def compute_length_and_repetition(text: str) -> Dict[str, float]:
    toks = text.strip().split()
    return {
        "length": float(len(toks)),
        "repeat_win5": _repeat_heuristics(toks, window=5),
    }


def score_with_reward_model(
    reward_model: Any,
    tokenizer: AutoTokenizer,
    texts: List[str],
    device: torch.device,
    max_length: int = 512,
) -> List[float]:
    """
    Batch score outputs with the reward model head used in RM/PPO.
    """
    reward_model.eval()
    scores = []
    with torch.no_grad():
        for t in texts:
            enc = tokenizer(t, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt").to(device)
            s = reward_model(enc["input_ids"], enc["attention_mask"])
            if isinstance(s, torch.Tensor):
                s = s.squeeze().item()
            scores.append(float(s))
    return scores


def summarize_metrics_per_output(hyp: str, ref: Optional[str] = None) -> Dict[str, Any]:
    """
    Compute basic metrics for a single hypothesis (and optional reference).
    """
    out = {}
    out.update(compute_diversity(hyp))
    out.update(compute_length_and_repetition(hyp))
    if ref is not None:
        bleu = compute_bleu(hyp, ref)
        rouge = compute_rouge(hyp, ref)
        if bleu is not None:
            out["bleu"] = bleu
        if rouge is not None:
            out.update({f"{k}": v for k, v in rouge.items()})
    return out


def aggregate_metric_table(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Aggregate a list of metric dicts into means for reporting.
    """
    if not rows:
        return {}
    agg: Dict[str, List[float]] = {}
    for r in rows:
        for k, v in r.items():
            if isinstance(v, (int, float)) and not math.isnan(v):
                agg.setdefault(k, []).append(float(v))
    return {k: float(statistics.mean(vs)) for k, vs in agg.items()}


def compare_models_summary(
    baseline_rows: List[Dict[str, Any]],
    aligned_rows: List[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """
    Return mean metrics for baseline vs aligned.
    """
    return {
        "baseline": aggregate_metric_table(baseline_rows),
        "aligned": aggregate_metric_table(aligned_rows),
    }