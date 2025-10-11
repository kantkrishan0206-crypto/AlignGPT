#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Builds pref_pairs.jsonl from candidates.jsonl.
- Strategies:
  1) Rule-based ranking (length, lexical diversity, simple heuristics)
  2) Scoring using your reward model (if provided)
- Outputs JSONL lines: {"prompt": "...", "chosen": "...", "rejected": "..."}

Usage (rule-based):
  python src/data/make_pref_pairs.py \
    --candidates data/candidates.jsonl \
    --output data/pref_pairs.jsonl \
    --strategy heuristic

Usage (reward-scored):
  python src/data/make_pref_pairs.py \
    --candidates data/candidates.jsonl \
    --output data/pref_pairs.jsonl \
    --strategy reward \
    --reward-ckpt ./checkpoints/rm_gpt2 \
    --model-name gpt2
"""

import os, sys, json, argparse
from typing import List, Dict, Tuple
import math

def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def write_jsonl(path: str, rows: List[Dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def score_heuristic(text: str) -> float:
    # Simple composite: length within range + unique tokens ratio
    tokens = text.strip().split()
    n = len(tokens)
    if n == 0:
        return -1e9
    uniq = len(set(tokens))
    diversity = uniq / max(1, n)
    # Penalize too short/too long
    length_score = -abs(n - 64) / 64.0
    return diversity + length_score

def pair_by_heuristic(prompt: str, candidates: List[str]) -> List[Tuple[str, str]]:
    # Rank and pair adjacent best vs worst
    scored = sorted([(c, score_heuristic(c)) for c in candidates], key=lambda x: x[1], reverse=True)
    pairs = []
    i, j = 0, len(scored) - 1
    while i < j:
        chosen, rejected = scored[i][0], scored[j][0]
        pairs.append((chosen, rejected))
        i += 1
        j -= 1
    return pairs

def load_reward_and_tokenizer(reward_ckpt: str, model_name: str):
    import torch
    from transformers import AutoTokenizer
    from src.models.reward import RewardModel
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None and tok.eos_token:
        tok.pad_token = tok.eos_token
    rm = RewardModel.from_pretrained(reward_ckpt, base_path_or_name=model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rm.to(device)
    rm.eval()
    return rm, tok, device

def score_with_reward(rm, tok, device, text: str, max_len: int = 512) -> float:
    import torch
    enc = tok(text, truncation=True, max_length=max_len, padding="max_length", return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    with torch.no_grad():
        score = rm(input_ids, attn)
    return float(score.squeeze().item()) if hasattr(score, "item") else float(score)

def pair_by_reward(prompt: str, candidates: List[str], rm, tok, device) -> List[Tuple[str, str]]:
    scored = [(c, score_with_reward(rm, tok, device, c)) for c in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)
    pairs = []
    i, j = 0, len(scored) - 1
    while i < j:
        pairs.append((scored[i][0], scored[j][0]))
        i += 1
        j -= 1
    return pairs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--strategy", choices=["heuristic", "reward"], default="heuristic")
    ap.add_argument("--reward-ckpt", default=None)
    ap.add_argument("--model-name", default="gpt2")
    args = ap.parse_args()

    use_reward = args.strategy == "reward"
    rm, tok, device = (None, None, None)
    if use_reward:
        assert args.reward_ckpt is not None, "--reward-ckpt required for reward strategy"
        rm, tok, device = load_reward_and_tokenizer(args.reward_ckpt, args.model_name)

    out_rows = []
    for row in read_jsonl(args.candidates):
        prompt = row.get("prompt")
        cands = row.get("candidates", [])
        if not prompt or len(cands) < 2:
            continue
        pairs = pair_by_reward(prompt, cands, rm, tok, device) if use_reward else pair_by_heuristic(prompt, cands)
        for chosen, rejected in pairs:
            out_rows.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})

    write_jsonl(args.output, out_rows)
    print(f"Wrote {len(out_rows)} preference pairs to {args.output}")

if __name__ == "__main__":
    main()