#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Reward Model Trainer
- Trains a reward model from preference pairs (A vs B)
- Uses Bradley–Terry style pairwise loss
- Config-driven (configs/rm_gpt2.yaml)
"""

import os, sys, yaml, argparse, logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, AdamW, get_scheduler

# Local imports
from src.utils.seed import set_global_seed
from src.utils.logging import init_logging, get_logger
from src.utils.checkpoints import save_checkpoint_safe
from src.models.reward import RewardModel

class PairwiseLoss(nn.Module):
    """Bradley–Terry loss for preference pairs."""
    def forward(self, chosen_scores, rejected_scores):
        return -torch.mean(torch.log(torch.sigmoid(chosen_scores - rejected_scores)))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/rm_gpt2.yaml")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/rm_gpt2")
    parser.add_argument("--log_dir", type=str, default="./logs")
    return parser.parse_args()

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def build_dataloader(data_path, tokenizer, batch_size, max_length=512):
    dataset = load_dataset("json", data_files=data_path, split="train")

    def tokenize(example):
        chosen = tokenizer(example["chosen"], truncation=True, max_length=max_length, padding="max_length")
        rejected = tokenizer(example["rejected"], truncation=True, max_length=max_length, padding="max_length")
        return {
            "chosen_input_ids": chosen["input_ids"],
            "chosen_attention_mask": chosen["attention_mask"],
            "rejected_input_ids": rejected["input_ids"],
            "rejected_attention_mask": rejected["attention_mask"],
        }

    tokenized = dataset.map(tokenize, batched=False)
    tokenized.set_format(type="torch")
    return DataLoader(tokenized, batch_size=batch_size, shuffle=True)

def train(cfg, args):
    set_global_seed(cfg.get("seed", 42))
    init_logging(args.log_dir)
    logger = get_logger(__name__)

    model_name = cfg["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModel.from_pretrained(model_name)
    model = RewardModel(base_model).to("cuda" if torch.cuda.is_available() else "cpu")

    dataloader = build_dataloader(cfg["data"]["path"], tokenizer, cfg["training"]["batch_size"])
    optimizer = AdamW(model.parameters(), lr=cfg["training"]["lr"])
    scheduler = get_scheduler("linear", optimizer, num_warmup_steps=0, num_training_steps=len(dataloader)*cfg["training"]["epochs"])
    criterion = PairwiseLoss()

    logger.info("Starting reward model training...")
    model.train()
    for epoch in range(cfg["training"]["epochs"]):
        for step, batch in enumerate(dataloader):
            chosen_ids = batch["chosen_input_ids"].to(model.device)
            chosen_mask = batch["chosen_attention_mask"].to(model.device)
            rejected_ids = batch["rejected_input_ids"].to(model.device)
            rejected_mask = batch["rejected_attention_mask"].to(model.device)

            chosen_scores = model(chosen_ids, chosen_mask)
            rejected_scores = model(rejected_ids, rejected_mask)

            loss = criterion(chosen_scores, rejected_scores)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if step % 10 == 0:
                logger.info(f"Epoch {epoch} Step {step} Loss {loss.item():.4f}")

        save_checkpoint_safe(model, args.output_dir, f"epoch_{epoch}")

    logger.info("Training complete.")

if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    train(cfg, args)