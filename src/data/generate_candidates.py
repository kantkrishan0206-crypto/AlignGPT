#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generates candidate responses for prompts using a policy checkpoint.
- Streams prompts.jsonl and writes candidates.jsonl with multiple samples per prompt
- Validates, deduplicates per prompt, and tracks metadata

Schema: {"prompt": "...", "candidates": ["...", "..."], "model": "sft_gpt2", "n": 4}

Usage:
  python src/data/generate_candidates.py \
    --prompts data/prompts.jsonl \
    --policy-ckpt ./checkpoints/sft_gpt2 \
    --model-name gpt2 \
    --output data/candidates.jsonl \
    --num-samples 4 \
    --max-prompt-length 256 \
    --max-new-tokens 128 \
    --top-p 0.9 --temperature 1.0
"""

import os, sys, json, argparse
from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

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

def build_gen_cfg(args, tokenizer):
    return GenerationConfig(
        do_sample=True,
        top_p=args.top_p,
        top_k=args.top_k,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        return_dict_in_generate=True,
        output_scores=False,
        num_return_sequences=args.num_samples,
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--policy-ckpt", required=True)
    ap.add_argument("--model-name", default="gpt2")
    ap.add_argument("--output", required=True)
    ap.add_argument("--num-samples", type=int, default=4)
    ap.add_argument("--max-prompt-length", type=int, default=256)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None and tokenizer.eos_token:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.policy_ckpt)
    model.to(device)
    model.eval()

    gen_cfg = build_gen_cfg(args, tokenizer)

    results = []
    for row in read_jsonl(args.prompts):
        prompt = row.get("prompt")
        if not prompt:
            continue
        inputs = tokenizer(
            prompt, truncation=True, max_length=args.max_prompt_length, padding=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, generation_config=gen_cfg)
        seqs = out.sequences.view(args.num_samples, -1) if out.sequences.dim() == 2 else out.sequences
        texts = [tokenizer.decode(seqs[i], skip_special_tokens=True) for i in range(args.num_samples)]
        # Deduplicate per-prompt
        uniq = []
        seen = set()
        for t in texts:
            tt = t.strip()
            if tt and tt not in seen:
                seen.add(tt)
                uniq.append(tt)
        results.append({"prompt": prompt, "candidates": uniq, "model": os.path.basename(args.policy_ckpt), "n": len(uniq)})

    write_jsonl(args.output, results)
    print(f"Wrote {len(results)} prompts with candidates to {args.output}")

if __name__ == "__main__":
    main()