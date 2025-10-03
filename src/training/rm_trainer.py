"""
src/training/rm_trainer.py

Command-line trainer for the Reward Model component of the RLHF pipeline.

Features:
- Load configuration from YAML or CLI.
- Load preference pairs JSONL data (prompt, chosen, rejected).
- Build PrefPairDataset (tokenized) and train using RewardTrainer from src/models/reward.py.
- Support for resume from checkpoint, evaluation, small-sample smoke test, and saving best checkpoint.
- Helpful logging and deterministic seeding for reproducibility.

Usage:
    python src/training/rm_trainer.py --config configs/rm_gpt2.yaml

Make sure you have installed requirements in environment.yml and run from the project root.
"""
from __future__ import annotations

import os
import sys
import json
import time
import argparse
import logging
from dataclasses import dataclass
from typing import Optional, List

# allow running from project root when invoked directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml
import torch
import random
import numpy as np

# Local imports
try:
    from src.models.reward import RewardConfig, RewardModel, RewardTrainer, load_pref_pairs_from_jsonl, PrefPairDataset
    from src.models.tokenizer import TokenizerConfig, TokenizerWrapper
except Exception:
    # fallback for different working directory layouts
    from models.reward import RewardConfig, RewardModel, RewardTrainer, load_pref_pairs_from_jsonl, PrefPairDataset
    from models.tokenizer import TokenizerConfig, TokenizerWrapper

logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)


DEFAULT_CONFIG = {
    "model_name_or_path": "distilbert-base-uncased",
    "tokenizer_name_or_path": None,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "max_length": 128,
    "hidden_pool": "cls",
    "dropout": 0.1,
    "head_hidden": 256,
    "head_activation": "tanh",
    "lr": 2e-5,
    "weight_decay": 0.0,
    "adam_eps": 1e-8,
    "use_amp": True,
    "gradient_accumulation_steps": 1,
    "max_epochs": 3,
    "train_batch_size": 16,
    "eval_batch_size": 32,
    "save_every": 1000,
    "out_dir": "./checkpoints/rm",
    "seed": 42,
}


def parse_args():
    p = argparse.ArgumentParser(description="Reward Model Trainer (RM)")
    p.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    p.add_argument("--train_file", type=str, default="data/pref_pairs.jsonl", help="JSONL file with preference pairs")
    p.add_argument("--val_file", type=str, default=None, help="Optional validation JSONL file")
    p.add_argument("--out_dir", type=str, default=None, help="Override output dir")
    p.add_argument("--max_items", type=int, default=None, help="Limit number of pref pairs (for smoke tests)")
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    p.add_argument("--local_files_only", action="store_true", help="Only load HF models/tokenizers from local files")
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: Optional[str]) -> RewardConfig:
    if path is None:
        raw = DEFAULT_CONFIG
    else:
        with open(path, "r") as fh:
            raw = yaml.safe_load(fh) or {}
        # merge defaults
        merged = DEFAULT_CONFIG.copy()
        merged.update(raw)
        raw = merged
    return RewardConfig(**raw)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.out_dir:
        cfg.out_dir = args.out_dir
    os.makedirs(cfg.out_dir, exist_ok=True)

    set_seed(cfg.seed)

    logger.info("Reward Model Training - config: %s", cfg)

    # load preference data
    train_pairs = load_pref_pairs_from_jsonl(args.train_file, max_items=args.max_items)
    if len(train_pairs) == 0:
        logger.warning("No training preference pairs found at %s. Running smoke test with synthetic data.", args.train_file)
        train_pairs = [
            {"prompt": "Q: Define AI.", "chosen": "AI is the field of computer science...", "rejected": "AI is a kind of sandwich."},
            {"prompt": "Write a friendly greeting.", "chosen": "Hello! How can I help you today?", "rejected": "Go away."},
        ]

    val_pairs = None
    if args.val_file:
        val_pairs = load_pref_pairs_from_jsonl(args.val_file, max_items=None)

    # tokenizer config
    tokenizer_name = cfg.tokenizer_name_or_path or cfg.model_name_or_path
    tok_cfg = TokenizerConfig(model_name_or_path=tokenizer_name, max_length=cfg.max_length)
    tokenizer = TokenizerWrapper.from_pretrained(tok_cfg, local_files_only=args.local_files_only)

    # build datasets
    train_ds = PrefPairDataset(train_pairs, tokenizer=tokenizer._tokenizer if hasattr(tokenizer, "_tokenizer") else tokenizer, cfg=cfg)
    val_ds = PrefPairDataset(val_pairs, tokenizer=tokenizer._tokenizer if (val_pairs is not None and hasattr(tokenizer, "_tokenizer")) else tokenizer, cfg=cfg) if val_pairs is not None else None

    # instantiate or load model
    model = RewardModel.from_pretrained(cfg)

    # prepare trainer
    trainer = RewardTrainer(model=model, cfg=cfg)

    # if resume from checkpoint
    if args.resume:
        logger.info("Resuming from checkpoint %s", args.resume)
        try:
            # assume checkpoint dir contains reward_head.pt and backbone
            model = RewardModel.load_pretrained(args.resume, device=cfg.device)
            trainer.model = model
        except Exception as e:
            logger.warning("Failed to load checkpoint from %s: %s", args.resume, e)

    # train
    logger.info("Starting training on %d pairs", len(train_ds))
    trainer.train(train_dataset=train_ds, val_dataset=val_ds)

    # final evaluation if val dataset provided
    if val_ds is not None:
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=cfg.eval_batch_size, shuffle=False, collate_fn=lambda b: b)
        val_loss = trainer.evaluate(val_loader)
        logger.info("Final validation loss: %.4f", val_loss)

    logger.info("Reward model training complete; saving final model to %s", os.path.join(cfg.out_dir, "final"))
    trainer.model.save_pretrained(os.path.join(cfg.out_dir, "final"))


if __name__ == "__main__":
    main()
