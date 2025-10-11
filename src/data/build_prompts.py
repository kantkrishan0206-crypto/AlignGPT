#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Builds a clean prompts.jsonl from raw sources.
- Validates schema and length
- Deduplicates and shuffles
- Splits train/val/test if requested

Usage:
  python src/data/build_prompts.py \
    --input raw_corpus.jsonl \
    --output data/prompts.jsonl \
    --prompt-field prompt \
    --min-len 5 --max-len 1024 \
    --shuffle --seed 42 \
    --split-ratios 0.9 0.05 0.05
"""

import os, sys, json, argparse, random
from typing import List, Dict

def read_jsonl(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def write_jsonl(path: str, rows: List[Dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def normalize_prompt(text: str) -> str:
    return " ".join(text.strip().split())

def filter_prompts(rows: List[Dict], field: str, min_len: int, max_len: int) -> List[Dict]:
    out = []
    for r in rows:
        p = r.get(field) or r.get("instruction") or r.get("text")
        if not p:
            continue
        p = normalize_prompt(p)
        if min_len <= len(p) <= max_len:
            out.append({"prompt": p})
    return out

def dedupe(rows: List[Dict]) -> List[Dict]:
    seen, out = set(), []
    for r in rows:
        p = r["prompt"]
        if p in seen:
            continue
        seen.add(p)
        out.append(r)
    return out

def split(rows: List[Dict], ratios: List[float]):
    assert abs(sum(ratios) - 1.0) < 1e-6, "Split ratios must sum to 1.0"
    n = len(rows)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    train = rows[:n_train]
    val = rows[n_train:n_train + n_val]
    test = rows[n_train + n_val:]
    return train, val, test

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--prompt-field", default="prompt")
    ap.add_argument("--min-len", type=int, default=5)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-ratios", nargs=3, type=float, default=None, help="train val test")
    args = ap.parse_args()

    rows = read_jsonl(args.input)
    rows = filter_prompts(rows, args.prompt_field, args.min_len, args.max_len)
    rows = dedupe(rows)
    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(rows)

    if args.split_ratios:
        train, val, test = split(rows, args.split_ratios)
        base = os.path.splitext(args.output)[0]
        write_jsonl(base + ".train.jsonl", train)
        write_jsonl(base + ".val.jsonl", val)
        write_jsonl(base + ".test.jsonl", test)
        print(f"Wrote splits: {len(train)} train, {len(val)} val, {len(test)} test")
    else:
        write_jsonl(args.output, rows)
        print(f"Wrote {len(rows)} prompts to {args.output}")

if __name__ == "__main__":
    main()