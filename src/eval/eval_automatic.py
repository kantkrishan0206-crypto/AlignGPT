#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Automatic evaluation pipeline:
- Load a policy checkpoint and tokenizer
- Generate outputs for prompts
- Compute metrics per-output and aggregate summaries
- Optional: score with reward model
- Compare baseline vs aligned models if both are provided
- Write a JSON report with full details and summary

Usage:
  python src/eval/eval_automatic.py \
    --prompts data/prompts.jsonl \
    --policy-ckpt ./checkpoints/ppo_gpt2 \
    --model-name gpt2 \
    --output reports/ppo_eval.json \
    --max-prompt-length 256 \
    --max-new-tokens 128 \
    --top-p 0.9 --temperature 1.0 \
    --reward-ckpt ./checkpoints/rm_gpt2

Comparison mode:
  python src/eval/eval_automatic.py \
    --prompts data/prompts.jsonl \
    --baseline-ckpt ./checkpoints/sft_gpt2 \
    --aligned-ckpt ./checkpoints/ppo_gpt2 \
    --model-name gpt2 \
    --output reports/compare_sft_vs_ppo.json
"""

import os
import json
import argparse
from typing import List, Dict, Any, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

from src.utils.logging import init_logging, get_logger, log_dict
from src.utils.seed import set_global_seed
from src.eval.metrics import (
    summarize_metrics_per_output,
    aggregate_metric_table,
    compare_models_summary,
    score_with_reward_model,
)
from src.models.reward import RewardModel


def read_prompts(path: str) -> List[str]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            p = obj.get("prompt") or obj.get("instruction") or obj.get("text")
            if p:
                rows.append(p)
    return rows


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
        num_return_sequences=1,
    )


def generate_outputs(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    max_prompt_length: int,
    gen_cfg: GenerationConfig,
) -> List[str]:
    device = next(model.parameters()).device
    model.eval()
    outs = []
    with torch.no_grad():
        for p in prompts:
            enc = tokenizer(p, truncation=True, max_length=max_prompt_length, padding=True, return_tensors="pt").to(device)
            gen = model.generate(**enc, generation_config=gen_cfg)
            text = tokenizer.decode(gen.sequences[0], skip_special_tokens=True)
            outs.append(text)
    return outs


def evaluate_single_model(
    prompts: List[str],
    outputs: List[str],
    references: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for i, out in enumerate(outputs):
        ref = references[i] if references and i < len(references) else None
        rows.append(summarize_metrics_per_output(out, ref))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model-name", default="gpt2")
    ap.add_argument("--seed", type=int, default=42)

    # Single-model mode
    ap.add_argument("--policy-ckpt", default=None)

    # Comparison mode
    ap.add_argument("--baseline-ckpt", default=None)
    ap.add_argument("--aligned-ckpt", default=None)

    # Generation config
    ap.add_argument("--max-prompt-length", type=int, default=256)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--repetition-penalty", type=float, default=1.0)

    # Optional reward scoring
    ap.add_argument("--reward-ckpt", default=None)

    args = ap.parse_args()
    init_logging("./logs", run_name="eval_automatic")
    logger = get_logger(__name__)
    set_global_seed(args.seed)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    prompts = read_prompts(args.prompts)
    logger.info(f"Loaded {len(prompts)} prompts from {args.prompts}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None and tokenizer.eos_token:
        tokenizer.pad_token = tokenizer.eos_token

    gen_cfg = build_gen_cfg(args, tokenizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    report: Dict[str, Any] = {"prompts_path": args.prompts, "model_name": args.model_name}

    # Optional reward model
    reward_model = None
    if args.reward_ckpt:
        reward_model = RewardModel.from_pretrained(args.reward_ckpt, base_path_or_name=args.model_name).to(device)
        logger.info(f"Loaded reward model from {args.reward_ckpt}")

    # Single-model mode
    if args.policy_ckpt:
        logger.info(f"Evaluating single model: {args.policy_ckpt}")
        model = AutoModelForCausalLM.from_pretrained(args.policy_ckpt).to(device)
        outs = generate_outputs(model, tokenizer, prompts, args.max_prompt_length, gen_cfg)
        per_rows = evaluate_single_model(prompts, outs)
        agg = aggregate_metric_table(per_rows)
        log_dict(logger, "single_model_metrics", agg)

        report.update({
            "mode": "single",
            "policy_ckpt": args.policy_ckpt,
            "outputs_count": len(outs),
            "metrics_mean": agg,
            "metrics_per_output": per_rows,
        })

        if reward_model is not None:
            rewards = score_with_reward_model(reward_model, tokenizer, outs, device)
            report["reward_scores"] = rewards
            report["reward_mean"] = float(sum(rewards) / max(1, len(rewards)))

    # Comparison mode
    elif args.baseline_ckpt and args.aligned_ckpt:
        logger.info(f"Comparing baseline={args.baseline_ckpt} vs aligned={args.aligned_ckpt}")
        base = AutoModelForCausalLM.from_pretrained(args.baseline_ckpt).to(device)
        aligned = AutoModelForCausalLM.from_pretrained(args.aligned_ckpt).to(device)

        base_outs = generate_outputs(base, tokenizer, prompts, args.max_prompt_length, gen_cfg)
        aligned_outs = generate_outputs(aligned, tokenizer, prompts, args.max_prompt_length, gen_cfg)

        base_rows = evaluate_single_model(prompts, base_outs)
        aligned_rows = evaluate_single_model(prompts, aligned_outs)
        summary = compare_models_summary(base_rows, aligned_rows)
        log_dict(logger, "compare_summary", {"baseline_mean": summary["baseline"], "aligned_mean": summary["aligned"]})

        report.update({
            "mode": "compare",
            "baseline_ckpt": args.baseline_ckpt,
            "aligned_ckpt": args.aligned_ckpt,
            "baseline_outputs_count": len(base_outs),
            "aligned_outputs_count": len(aligned_outs),
            "baseline_metrics_mean": summary["baseline"],
            "aligned_metrics_mean": summary["aligned"],
            "baseline_metrics_per_output": base_rows,
            "aligned_metrics_per_output": aligned_rows,
        })

        if reward_model is not None:
            base_rewards = score_with_reward_model(reward_model, tokenizer, base_outs, device)
            aligned_rewards = score_with_reward_model(reward_model, tokenizer, aligned_outs, device)
            report["baseline_reward_mean"] = float(sum(base_rewards) / max(1, len(base_rewards)))
            report["aligned_reward_mean"] = float(sum(aligned_rewards) / max(1, len(aligned_rewards)))

    else:
        raise ValueError("Provide either --policy-ckpt for single model or both --baseline-ckpt and --aligned-ckpt for comparison")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"Wrote evaluation report to {args.output}")