#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DPO Trainer (Direct Preference Optimization)
File: src/training/dpo_trainer.py

Purpose:
- Optimize a supervised-fine-tuned policy directly using paired preferences (chosen vs rejected)
- Implements stable DPO loss with KL regularization to a frozen reference model
- Config-driven (configs/dpo_gpt2.yaml) with CLI overrides
- Integrated logging, checkpointing, and evaluation hooks
- Cleanly fits into the rlhf-lab project structure

Data expectation (data/pref_pairs.jsonl):
- Each JSONL line: {"prompt": "...", "chosen": "...", "rejected": "..."}
"""

import os
import sys
import time
import math
import json
import yaml
import argparse
from dataclasses import dataclass
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_scheduler,
    AdamW,
)

# Local modules per your project structure
try:
    from src.utils.seed import set_global_seed
    from src.utils.logging import init_logging, get_logger
    from src.utils.checkpoints import save_checkpoint_safe, load_checkpoint_safe
    from src.models.policy import load_policy_model
    from src.models.tokenizer import get_tokenizer
    from src.eval.metrics import compute_perplexity
except Exception as e:
    print(f"[ImportError] Verify your project structure and PYTHONPATH. Details: {e}")
    sys.exit(1)


# ------------------------------
# Config dataclass and helpers
# ------------------------------

@dataclass
class DPORunConfig:
    # Models
    model_name: str = "gpt2"
    policy_ckpt: str = "./checkpoints/sft_gpt2"     # path to SFT policy as initialization
    ref_ckpt: Optional[str] = None                  # optional explicit ref model; default: load from model_name
    trust_remote_code: bool = False

    # Data
    data_path: str = "data/pref_pairs.jsonl"
    prompt_field: str = "prompt"
    chosen_field: str = "chosen"
    rejected_field: str = "rejected"
    max_length: int = 512

    # Training
    batch_size: int = 8
    epochs: int = 3
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    warmup_ratio: float = 0.05
    grad_clip_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    dataloader_num_workers: int = 0

    # DPO loss coefficients
    beta: float = 0.1         # temperature for preference strength
    kl_coef: float = 0.02     # KL regularization weight

    # Infra
    seed: int = 42
    device: str = "auto"      # auto, cuda, cpu
    fp16: bool = True
    bf16: bool = False

    # Logging & checkpoints
    output_dir: str = "./checkpoints/dpo_gpt2"
    log_dir: str = "./logs"
    save_steps: int = 200
    save_total_limit: int = 3
    report_to: Optional[list] = None
    run_name: Optional[str] = None

    # Eval hooks
    eval_sample_size: int = 128
    eval_every_steps: int = 500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direct Preference Optimization Trainer")
    parser.add_argument("--config", type=str, default="configs/dpo_gpt2.yaml", help="Path to YAML config")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output dir")
    parser.add_argument("--log_dir", type=str, default=None, help="Override log dir")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint if available")
    parser.add_argument("--override", type=str, nargs="*", default=None, help="Key=Value overrides for config")
    parser.add_argument("--run_name", type=str, default=None, help="Experiment/run name")
    return parser.parse_args()


def load_yaml_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def coerce_value(value: str):
    lv = value.lower()
    if lv in ["true", "false"]:
        return lv == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def apply_overrides(cfg: Dict[str, Any], overrides: Optional[list]):
    if not overrides:
        return cfg
    for item in overrides:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        value = coerce_value(value)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            if p not in node or not isinstance(node[p], dict):
                node[p] = {}
            node = node[p]
        node[parts[-1]] = value
    return cfg


def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)


def pick_device(spec: str) -> torch.device:
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(spec)


# ------------------------------
# Dataset and batching
# ------------------------------

def build_dataloader(
    data_path: str,
    tokenizer: AutoTokenizer,
    prompt_field: str,
    chosen_field: str,
    rejected_field: str,
    max_length: int,
    batch_size: int,
    num_workers: int = 0,
) -> DataLoader:
    """
    Loads preference pairs and tokenizes prompt + completions.
    Produces a DataLoader with tensors for chosen and rejected completions.
    """
    ds = load_dataset("json", data_files=data_path, split="train")

    def tokenize_row(row):
        # We compute logprobs for chosen and rejected conditioned on prompt.
        prompt = row.get(prompt_field, "")
        chosen = row.get(chosen_field, "")
        rejected = row.get(rejected_field, "")

        # Concatenate prompt + completion for causal LM likelihoods
        chosen_full = prompt + chosen
        rejected_full = prompt + rejected

        chosen_enc = tokenizer(
            chosen_full, truncation=True, max_length=max_length, padding="max_length"
        )
        rejected_enc = tokenizer(
            rejected_full, truncation=True, max_length=max_length, padding="max_length"
        )
        return {
            "chosen_input_ids": chosen_enc["input_ids"],
            "chosen_attention_mask": chosen_enc["attention_mask"],
            "rejected_input_ids": rejected_enc["input_ids"],
            "rejected_attention_mask": rejected_enc["attention_mask"],
        }

    ds_tok = ds.map(tokenize_row, batched=False)
    ds_tok.set_format(type="torch")
    return DataLoader(ds_tok, batch_size=batch_size, shuffle=True, num_workers=num_workers)


# ------------------------------
# DPO objective
# ------------------------------

class DPOLoss(nn.Module):
    """
    DPO loss:
        L = -beta * [ log σ( (log pθ(c|x) - log pθ(r|x)) - (log pref_ref(c) - log pref_ref(r)) ) ]
            + kl_coef * KL[pθ || p_ref] (optional stabilizer)

    Implementation detail:
    - We compute log-likelihoods for chosen/rejected under current policy and a fixed reference model.
    - The difference of differences is passed through a log-sigmoid scaled by beta.
    - Optional KL regularization stabilizes policy against ref drift.

    For clarity, we approximate KL via token-level cross-entropy difference on the batch.
    """

    def __init__(self, beta: float = 0.1, kl_coef: float = 0.02):
        super().__init__()
        self.beta = beta
        self.kl_coef = kl_coef
        self.log_sigmoid = nn.LogSigmoid()

    def forward(
        self,
        logp_chosen: torch.Tensor,
        logp_rejected: torch.Tensor,
        logp_ref_chosen: torch.Tensor,
        logp_ref_rejected: torch.Tensor,
        token_kl: Optional[torch.Tensor] = None,
    ):
        # preference term
        diff_policy = logp_chosen - logp_rejected
        diff_ref = logp_ref_chosen - logp_ref_rejected
        margin = diff_policy - diff_ref  # larger => stronger preference alignment

        pref_loss = -self.beta * self.log_sigmoid(margin).mean()

        if token_kl is not None:
            kl_loss = token_kl.mean()
            total = pref_loss + self.kl_coef * kl_loss
            return total, {"pref_loss": pref_loss.item(), "kl_loss": kl_loss.item()}
        else:
            return pref_loss, {"pref_loss": pref_loss.item(), "kl_loss": None}


# ------------------------------
# Log-likelihood utilities
# ------------------------------

def causal_logprob(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Returns scalar log-likelihood per sequence by summing token logprobs.
    If you want per-token, modify to return the vector. We use sum for DPO pairwise differences.
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
    # Hugging Face returns loss averaged; we want token logprobs sum. Use logits to compute log-softmax.
    logits = outputs.logits  # [B, T, V]
    # Shift to predict next token
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_mask = attention_mask[:, 1:]

    log_probs = torch.log_softmax(shift_logits, dim=-1)  # [B, T-1, V]
    # gather logprobs at gold tokens
    gold_logp = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)  # [B, T-1]
    gold_logp = gold_logp * shift_mask  # mask pad tokens

    # Sum across tokens to get sequence log-likelihood
    seq_logp = gold_logp.sum(dim=-1)  # [B]
    return seq_logp


def token_kl_div(
    policy: AutoModelForCausalLM,
    ref: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Approximate token-level KL(policy || ref) averaged per sequence on the batch.
    """
    with torch.no_grad():
        ref_logits = ref(input_ids=input_ids, attention_mask=attention_mask).logits
    pol_logits = policy(input_ids=input_ids, attention_mask=attention_mask).logits

    # Shift to align next-token prediction
    pol = torch.log_softmax(pol_logits[:, :-1, :], dim=-1)
    ref = torch.softmax(ref_logits[:, :-1, :], dim=-1)

    # KL = sum_v pol_logprob * (pol_prob - ref_prob)  [using log-softmax for policy and softmax for ref]
    # More standard is KL = sum_v pol_prob * (log(pol_prob) - log(ref_prob))
    pol_prob = torch.exp(pol)  # [B, T-1, V]
    kl = pol_prob * (pol - torch.log(ref + 1e-12))
    # mask padding
    mask = attention_mask[:, 1:].unsqueeze(-1)
    kl = (kl * mask).sum(dim=-1)  # [B, T-1]
    seq_kl = kl.mean(dim=-1)      # [B] average per sequence
    return seq_kl


# ------------------------------
# Training routine
# ------------------------------

def train(cfg: DPORunConfig, resume: bool):
    ensure_dirs(cfg.output_dir, cfg.log_dir)
    init_logging(log_dir=cfg.log_dir)
    logger = get_logger(__name__)

    set_global_seed(cfg.seed)
    device = pick_device(cfg.device)
    logger.info(f"Starting DPO training device={device}, seed={cfg.seed}")

    # Tokenizer setup
    tokenizer = get_tokenizer(cfg.model_name, use_fast=True, trust_remote_code=cfg.trust_remote_code)
    if tokenizer.pad_token is None and tokenizer.eos_token:
        tokenizer.pad_token = tokenizer.eos_token

    # Policy and reference models
    logger.info("Loading policy model (initialized from SFT checkpoint or base model)...")
    policy = load_policy_model(
        cfg.policy_ckpt if os.path.isdir(cfg.policy_ckpt) else cfg.model_name,
        trust_remote_code=cfg.trust_remote_code,
    )
    policy.to(device)
    policy.train()

    logger.info("Loading reference model (frozen)...")
    if cfg.ref_ckpt and os.path.isdir(cfg.ref_ckpt):
        ref_model = AutoModelForCausalLM.from_pretrained(cfg.ref_ckpt)
    else:
        ref_model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
    ref_model.to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    # Dataloader
    loader = build_dataloader(
        data_path=cfg.data_path,
        tokenizer=tokenizer,
        prompt_field=cfg.prompt_field,
        chosen_field=cfg.chosen_field,
        rejected_field=cfg.rejected_field,
        max_length=cfg.max_length,
        batch_size=cfg.batch_size,
        num_workers=cfg.dataloader_num_workers,
    )

    # Optimizer and scheduler
    optimizer = AdamW(
        policy.parameters(),
        lr=cfg.learning_rate,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        eps=cfg.adam_eps,
        weight_decay=cfg.weight_decay,
    )

    total_steps = math.ceil(len(loader) * cfg.epochs / max(1, cfg.gradient_accumulation_steps))
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # Loss
    dpo_loss = DPOLoss(beta=cfg.beta, kl_coef=cfg.kl_coef)

    # Resume
    start_global_step = 0
    if resume:
        ckpt = load_checkpoint_safe(cfg.output_dir)
        if ckpt:
            logger.info(f"Resuming from checkpoint: {ckpt}")
            # If you saved state_dict, load here. For simplicity, we skip in this skeleton.
            # policy.load_state_dict(torch.load(...))
            # optimizer.load_state_dict(...)
            # scheduler.load_state_dict(...)
            # start_global_step = ...

    # Training loop
    global_step = start_global_step
    start_time = time.time()
    logger.info("Beginning DPO optimization...")

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.fp16 and torch.cuda.is_available())

    for epoch in range(cfg.epochs):
        for step, batch in enumerate(loader):
            chosen_ids = batch["chosen_input_ids"].to(device)
            chosen_mask = batch["chosen_attention_mask"].to(device)
            rejected_ids = batch["rejected_input_ids"].to(device)
            rejected_mask = batch["rejected_attention_mask"].to(device)

            with torch.cuda.amp.autocast(enabled=cfg.fp16 and torch.cuda.is_available()):
                # Log-likelihoods under policy
                logp_chosen = causal_logprob(policy, chosen_ids, chosen_mask)
                logp_rejected = causal_logprob(policy, rejected_ids, rejected_mask)

                # Log-likelihoods under reference (no grad)
                with torch.no_grad():
                    logp_ref_chosen = causal_logprob(ref_model, chosen_ids, chosen_mask)
                    logp_ref_rejected = causal_logprob(ref_model, rejected_ids, rejected_mask)

                # Token-level KL approximation on chosen sequences (stabilizer)
                token_kl = token_kl_div(policy, ref_model, chosen_ids, chosen_mask)

                loss, loss_parts = dpo_loss(
                    logp_chosen=logp_chosen,
                    logp_rejected=logp_rejected,
                    logp_ref_chosen=logp_ref_chosen,
                    logp_ref_rejected=logp_ref_rejected,
                    token_kl=token_kl,
                )

            # Backward
            scaler.scale(loss).backward()

            # Grad accumulation
            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                # Optional grad clipping
                if cfg.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip_norm)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

                # Logging
                if global_step % 10 == 0:
                    logger.info(
                        f"epoch={epoch} step={global_step} "
                        f"loss={loss.item():.4f} "
                        f"pref_loss={loss_parts['pref_loss']:.4f} "
                        f"kl_loss={loss_parts['kl_loss'] if loss_parts['kl_loss'] is not None else 'n/a'}"
                    )

                # Checkpointing
                if cfg.save_steps and global_step % cfg.save_steps == 0:
                    logger.info(f"Saving checkpoint at step {global_step}...")
                    save_checkpoint_safe(policy, cfg.output_dir, f"step_{global_step}")

                # Evaluation hook
                if cfg.eval_every_steps and global_step % cfg.eval_every_steps == 0:
                    try:
                        ppl = compute_perplexity(policy, tokenizer, None, sample_size=cfg.eval_sample_size)
                        logger.info(f"Approx perplexity (sample): {ppl:.2f}")
                    except Exception as e:
                        logger.warning(f"Eval skipped at step {global_step}: {e}")

    duration = time.time() - start_time
    logger.info(f"DPO training finished in {duration:.2f}s total_steps={global_step}")

    # Final save
    logger.info("Saving final policy...")
    save_checkpoint_safe(policy, cfg.output_dir, "final")

    # Manifest
    manifest = {
        "model_name": cfg.model_name,
        "policy_ckpt": cfg.policy_ckpt,
        "ref_ckpt": cfg.ref_ckpt,
        "output_dir": cfg.output_dir,
        "log_dir": cfg.log_dir,
        "seed": cfg.seed,
        "duration_sec": duration,
        "total_steps": global_step,
        "batch_size": cfg.batch_size,
        "lr": cfg.learning_rate,
        "beta": cfg.beta,
        "kl_coef": cfg.kl_coef,
        "max_length": cfg.max_length,
        "epochs": cfg.epochs,
    }
    with open(os.path.join(cfg.output_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info("All done. Checkpoints and logs are available.")
    return {"duration_sec": duration, "total_steps": global_step}


def main():
    args = parse_args()

    cfg_dict = load_yaml_config(args.config)
    cfg_dict = apply_overrides(cfg_dict, args.override)
    if args.output_dir:
        cfg_dict["output_dir"] = args.output_dir
    if args.log_dir:
        cfg_dict["log_dir"] = args.log_dir
    if args.run_name:
        cfg_dict["run_name"] = args.run_name

    cfg = DPORunConfig(
        model_name=cfg_dict.get("model_name", "gpt2"),
        policy_ckpt=cfg_dict.get("policy_ckpt", "./checkpoints/sft_gpt2"),
        ref_ckpt=cfg_dict.get("ref_ckpt", None),
        trust_remote_code=cfg_dict.get("trust_remote_code", False),
        data_path=cfg_dict.get("data", {}).get("path", "data/pref_pairs.jsonl"),
        prompt_field=cfg_dict.get("data", {}).get("prompt_field", "prompt"),
        chosen_field=cfg_dict.get("data", {}).get("chosen_field", "chosen"),
        rejected_field=cfg_dict.get("data", {}).get("rejected_field", "rejected"),
        max_length=cfg_dict.get("data", {}).get("max_length", 512),
        batch_size=cfg_dict.get("training", {}).get("batch_size", 8),
        epochs=cfg_dict.get("training", {}).get("epochs", 3),
        learning_rate=cfg_dict.get("training", {}).get("learning_rate", 1e-5),
        weight_decay=cfg_dict.get("training", {}).get("weight_decay", 0.0),
        adam_beta1=cfg_dict.get("optim", {}).get("beta1", 0.9),
        adam_beta2=cfg_dict.get("optim", {}).get("beta2", 0.95),
        adam_eps=cfg_dict.get("optim", {}).get("eps", 1e-8),
        warmup_ratio=cfg_dict.get("training", {}).get("warmup_ratio", 0.05),
        grad_clip_norm=cfg_dict.get("training", {}).get("grad_clip_norm", 1.0),
        gradient_accumulation_steps=cfg_dict.get("training", {}).get("gradient_accumulation_steps", 1),
        dataloader_num_workers=cfg_dict.get("training", {}).get("dataloader_num_workers", 0),
        beta=cfg_dict.get("dpo", {}).get("beta", 0.1),
        kl_coef=cfg_dict.get("dpo", {}).get("kl_coef", 0.02),
        seed=cfg_dict.get("seed", 42),
        device=cfg_dict.get("device", "auto"),
        fp16=cfg_dict.get("fp16", True),
        bf16=cfg_dict.get("bf16", False),
        output_dir=cfg_dict.get("output_dir", "./checkpoints/dpo_gpt2"),
        log_dir=cfg_dict.get("log_dir", "./logs"),
        save_steps=cfg_dict.get("training", {}).get("save_steps", 200),
        save_total_limit=cfg_dict.get("training", {}).get("save_total_limit", 3),
        report_to=cfg_dict.get("report_to", []),
        run_name=cfg_dict.get("run_name", None),
        eval_sample_size=cfg_dict.get("eval", {}).get("sample_size", 128),
        eval_every_steps=cfg_dict.get("eval", {}).get("every_steps", 500),
    )

    train(cfg, resume=args.resume)


if __name__ == "__main__":
    main()