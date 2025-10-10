#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train RLHF Orchestrator (SFT mode)
File: src/training/sft/train_rlhf.py

Purpose:
- Orchestrates supervised fine-tuning (SFT) for a causal LM (e.g., GPT-2)
- Modular integration with tokenizer, policy model, utils for logging/seed/checkpoints
- Config-driven via YAML + CLI overrides
- Ready to extend with reward/DPO/PPO modes later via --mode switch

Requirements:
- environment.yml should include: transformers, datasets, accelerate, trl (optional), pyyaml, wandb (optional)
"""

import os
import sys
import time
import json
import shutil
import logging
import argparse
from typing import Any, Dict, Optional

import yaml
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# Local modules (must exist per your project structure)
# If these imports fail, verify your repo structure and PYTHONPATH.
try:
    from src.utils.seed import set_global_seed
    from src.utils.logging import get_logger, init_logging
    from src.utils.checkpoints import save_checkpoint_safe, load_checkpoint_safe
    from src.models.policy import load_policy_model
    from src.models.tokenizer import get_tokenizer
    from src.eval.metrics import compute_perplexity
except Exception as e:
    print(f"[ImportError] Ensure your project structure and PYTHONPATH are correct: {e}")
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RLHF - SFT orchestrator")
    parser.add_argument("--config", type=str, default="configs/sft_gpt2.yaml", help="Path to YAML config")
    parser.add_argument("--mode", type=str, default="sft", choices=["sft"], help="Training mode")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/sft_gpt2", help="Checkpoint output dir")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Logging directory")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint in output_dir")
    parser.add_argument("--run_name", type=str, default=None, help="Experiment/run name (for wandb/mlflow)")
    parser.add_argument("--override", type=str, nargs="*", default=None, help="Key=Value overrides for config")
    return parser.parse_args()


def load_config(path: str, overrides: Optional[list] = None) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # Simple key=value CLI overrides (supports flat keys, e.g., 'training.batch_size=8')
    if overrides:
        for item in overrides:
            if "=" not in item:
                continue
            key, value = item.split("=", 1)
            # Try to coerce basic types
            if value.lower() in ["true", "false"]:
                value = value.lower() == "true"
            else:
                try:
                    if "." in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    pass
            # Support nested keys via dot notation
            node = cfg
            parts = key.split(".")
            for p in parts[:-1]:
                if p not in node or not isinstance(node[p], dict):
                    node[p] = {}
                node = node[p]
            node[parts[-1]] = value

    return cfg


def ensure_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # macOS MPS
        return torch.device("mps")
    return torch.device("cpu")


def build_dataset(data_cfg: Dict[str, Any], tokenizer: AutoTokenizer) -> Dict[str, Any]:
    """
    Assumes data/sft.jsonl has records like:
    {"prompt": "...", "response": "..."} or {"text": "..."}.

    Produces tokenized train dataset for LM training.
    """
    data_path = data_cfg.get("path", "data/sft.jsonl")
    split = data_cfg.get("split", "train")
    text_field = data_cfg.get("text_field", None)
    prompt_field = data_cfg.get("prompt_field", "prompt")
    response_field = data_cfg.get("response_field", "response")
    max_length = data_cfg.get("max_length", 1024)
    add_bos = data_cfg.get("add_bos", True)

    raw = load_dataset("json", data_files=data_path, split=split)

    def to_text(example):
        if text_field and text_field in example:
            text = example[text_field]
        else:
            prompt = example.get(prompt_field, "")
            response = example.get(response_field, "")
            text = prompt + response
        if add_bos and tokenizer.bos_token:
            text = tokenizer.bos_token + text
        return {"text": text}

    processed = raw.map(to_text, remove_columns=[c for c in raw.column_names if c not in ["text"]])

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    tokenized = processed.map(tokenize, batched=True)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return {"train": tokenized}


def build_data_collator(tokenizer: AutoTokenizer, mlm: bool = False) -> DataCollatorForLanguageModeling:
    return DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=mlm)


def setup_trainer(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    datasets: Dict[str, Any],
    train_cfg: Dict[str, Any],
    output_dir: str,
    log_dir: str,
    run_name: Optional[str] = None,
) -> Trainer:
    args = TrainingArguments(
        output_dir=output_dir,
        report_to=train_cfg.get("report_to", []),  # e.g., ["wandb"]
        run_name=run_name,
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 4),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 1),
        num_train_epochs=train_cfg.get("num_train_epochs", 1),
        learning_rate=train_cfg.get("learning_rate", 5e-5),
        weight_decay=train_cfg.get("weight_decay", 0.0),
        warmup_steps=train_cfg.get("warmup_steps", 0),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "linear"),
        logging_dir=log_dir,
        logging_steps=train_cfg.get("logging_steps", 10),
        save_strategy=train_cfg.get("save_strategy", "steps"),
        save_steps=train_cfg.get("save_steps", 200),
        save_total_limit=train_cfg.get("save_total_limit", 3),
        evaluation_strategy=train_cfg.get("evaluation_strategy", "no"),  # change to "steps" if you add eval
        eval_steps=train_cfg.get("eval_steps", 500),
        fp16=train_cfg.get("fp16", True),
        bf16=train_cfg.get("bf16", False),
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 2),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", False),
        torch_compile=train_cfg.get("torch_compile", False),
        deepspeed=train_cfg.get("deepspeed", None),  # path to ds config if needed
    )

    data_collator = build_data_collator(tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=datasets["train"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    return trainer


def save_training_manifest(output_dir: str, manifest: Dict[str, Any]) -> None:
    path = os.path.join(output_dir, "manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def maybe_resume(trainer: Trainer, resume_flag: bool, output_dir: str) -> Optional[str]:
    if not resume_flag:
        return None
    last_ckpt = load_checkpoint_safe(output_dir)
    return last_ckpt  # return path or None


def main():
    args = parse_args()

    # Initialize logging
    ensure_dirs(args.output_dir, args.log_dir)
    init_logging(log_dir=args.log_dir)
    logger = get_logger(__name__)
    logger.info("Starting Train RLHF (SFT mode)")

    # Load config
    cfg = load_config(args.config, overrides=args.override)
    logger.info(f"Loaded config from {args.config}")

    # Seed for reproducibility
    seed_val = cfg.get("seed", 42)
    set_global_seed(seed_val)
    logger.info(f"Global seed set: {seed_val}")

    # Device info
    device = get_device()
    logger.info(f"Using device: {device}")

    # Tokenizer & Model
    model_name = cfg.get("model_name", "gpt2")
    tokenizer_cfg = cfg.get("tokenizer", {})
    policy_cfg = cfg.get("policy", {})

    tokenizer = get_tokenizer(model_name, **tokenizer_cfg)
    model = load_policy_model(model_name, **policy_cfg)
    model.to(device)
    logger.info(f"Loaded tokenizer & policy model: {model_name}")

    # Dataset
    data_cfg = cfg.get("data", {"path": "data/sft.jsonl"})
    datasets = build_dataset(data_cfg, tokenizer)
    logger.info(f"Dataset loaded and tokenized from {data_cfg.get('path')}")

    # Trainer
    train_cfg = cfg.get("training", {})
    trainer = setup_trainer(
        model=model,
        tokenizer=tokenizer,
        datasets=datasets,
        train_cfg=train_cfg,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        run_name=args.run_name,
    )

    # Resume checkpoint (optional)
    resume_ckpt = maybe_resume(trainer, args.resume, args.output_dir)
    if resume_ckpt:
        logger.info(f"Resuming from checkpoint: {resume_ckpt}")

    # Train
    start = time.time()
    logger.info("Beginning training...")
    trainer.train(resume_from_checkpoint=resume_ckpt)
    duration = time.time() - start
    logger.info(f"Training completed in {duration:.2f}s")

    # Save final checkpoint safely
    logger.info("Saving final checkpoint...")
    save_checkpoint_safe(trainer, args.output_dir)

    # Compute a quick metric (e.g., perplexity on a small slice)
    try:
        ppl = compute_perplexity(model, tokenizer, datasets["train"], sample_size=128)
        logger.info(f"Approx perplexity (train sample): {ppl:.2f}")
    except Exception as e:
        logger.warning(f"Perplexity computation skipped: {e}")

    # Save manifest
    manifest = {
        "config_path": args.config,
        "model_name": model_name,
        "output_dir": args.output_dir,
        "log_dir": args.log_dir,
        "seed": seed_val,
        "duration_sec": duration,
        "resume_from": resume_ckpt,
        "train_cfg": train_cfg,
        "data_cfg": data_cfg,
    }
    save_training_manifest(args.output_dir, manifest)

    logger.info("All done. Checkpoints and logs are available.")


if __name__ == "__main__":
    main()