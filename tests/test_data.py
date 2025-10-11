import os
import json
import tempfile
from src.data.build_prompts import normalize_prompt, dedupe
from src.data.make_pref_pairs import score_heuristic, pair_by_heuristic

def test_normalize_prompt():
    raw = "   What is RLHF?   \n\n"
    norm = normalize_prompt(raw)
    assert norm == "What is RLHF?"

def test_dedupe_prompts():
    rows = [{"prompt": "A"}, {"prompt": "B"}, {"prompt": "A"}]
    deduped = dedupe(rows)
    assert len(deduped) == 2

def test_score_heuristic():
    text = "This is a test sentence with some diversity."
    score = score_heuristic(text)
    assert isinstance(score, float)

def test_pair_by_heuristic():
    prompt = "Explain transformers"
    candidates = ["Short answer", "Detailed explanation", "Another one"]
    pairs = pair_by_heuristic(prompt, candidates)
    assert all(len(p) == 2 for p in pairs)