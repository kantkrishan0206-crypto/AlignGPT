"""
src/models/reward.py

Production-ready RewardModel and lightweight training utilities for RLHF.

Features:
- Flexible backbones: can wrap any HF encoder (bert/roberta) or causal LM (gpt-style) and pool hidden states.
- Pairwise preference loss (Bradley-Terry / logistic), optional margin ranking loss.
- Regression/BCE options for scalar or probabilistic targets.
- Torch-native training loop with mixed precision (AMP), gradient accumulation, and scheduler support.
- Utilities to load preference data from JSONL, batching/collation helpers, and evaluation metrics (accuracy, AUC).
- Save/load helpers for model + tokenizer + config.

Design notes:
- This module avoids heavy external dependencies (no Lightning) so it's easy to integrate into research pipelines or production
  training loops while still being robust for larger-scale training.

Expected input formats for preference pairs (pref_pairs.jsonl):
{ "prompt": "Write a short recipe for tea.",
  "chosen": "Boil water...",
  "rejected": "First do nothing..." }


Example usage (high-level):
    cfg = RewardConfig(model_name_or_path="bert-base-uncased", max_length=256)
    rm = RewardModel.from_pretrained(cfg)
    trainer = RewardTrainer(rm, cfg)
    trainer.train(pref_pairs_jsonl_path="data/pref_pairs.jsonl")

"""
from __future__ import annotations

import os
import json
import math
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoConfig,
    get_linear_schedule_with_warmup,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(h)


# ---------------------------
# Config dataclass
# ---------------------------

@dataclass
class RewardConfig:
    model_name_or_path: str = "bert-base-uncased"
    tokenizer_name_or_path: Optional[str] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_length: int = 256
    hidden_pool: str = "cls"  # 'cls', 'mean', 'last' pooling
    dropout: float = 0.1
    head_hidden: int = 256
    head_activation: str = "tanh"
    lr: float = 1e-5
    weight_decay: float = 0.0
    adam_eps: float = 1e-8
    use_amp: bool = True
    gradient_accumulation_steps: int = 1
    max_epochs: int = 3
    train_batch_size: int = 16
    eval_batch_size: int = 32
    save_every: int = 1000  # steps
    out_dir: str = "./checkpoints/rm"
    seed: int = 42


# ---------------------------
# Dataset helpers
# ---------------------------

class PrefPairDataset(Dataset):
    """Dataset of preference pairs: each item is (prompt, chosen, rejected).

    Optionally the dataset can store tokenized inputs to speed training if a tokenizer
    is provided before DataLoader creation.
    """

    def __init__(self, pairs: List[Dict[str, str]], tokenizer: Optional[AutoTokenizer] = None, cfg: Optional[RewardConfig] = None):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.cfg = cfg

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item = self.pairs[idx]
        prompt = item["prompt"]
        chosen = item["chosen"]
        rejected = item["rejected"]
        if self.tokenizer is None:
            return {"prompt": prompt, "chosen": chosen, "rejected": rejected}
        # For reward training we typically concatenate prompt + response and feed into encoder
        max_length = self.cfg.max_length if self.cfg is not None else 256
        c_enc = self.tokenizer(prompt + self.tokenizer.sep_token + chosen, truncation=True, padding=False, max_length=max_length, return_tensors="pt")
        r_enc = self.tokenizer(prompt + self.tokenizer.sep_token + rejected, truncation=True, padding=False, max_length=max_length, return_tensors="pt")
        # squeeze tensors
        c_enc = {k: v.squeeze(0) for k, v in c_enc.items()}
        r_enc = {k: v.squeeze(0) for k, v in r_enc.items()}
        return {"chosen": c_enc, "rejected": r_enc}


def load_pref_pairs_from_jsonl(path: str, max_items: Optional[int] = None) -> List[Dict[str, str]]:
    pairs = []
    with open(path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if max_items is not None and i >= max_items:
                break
            obj = json.loads(line)
            # minimal validation
            if not all(k in obj for k in ("prompt", "chosen", "rejected")):
                continue
            pairs.append({"prompt": obj["prompt"], "chosen": obj["chosen"], "rejected": obj["rejected"]})
    return pairs


def collate_pref_pairs(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    # batch is list of {chosen: enc_dict, rejected: enc_dict}
    # we will pad chosen and rejected separately and return combined batch
    chosen_list = [b["chosen"] for b in batch]
    rejected_list = [b["rejected"] for b in batch]

    def pad_and_stack(enc_list):
        keys = enc_list[0].keys()
        out = {}
        for k in keys:
            tensors = [e[k] for e in enc_list]
            out[k] = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=0)
        return out

    chosen_batch = pad_and_stack(chosen_list)
    rejected_batch = pad_and_stack(rejected_list)
    return {"chosen": chosen_batch, "rejected": rejected_batch}


# ---------------------------
# Reward model
# ---------------------------

class RewardHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256, dropout: float = 0.1, activation: str = "tanh"):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        if activation == "tanh":
            self.act = nn.Tanh()
        elif activation == "relu":
            self.act = nn.ReLU()
        else:
            raise ValueError("Unsupported activation: %s" % activation)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.act(x)
        return self.out(x).squeeze(-1)


class RewardModel(nn.Module):
    """RewardModel wraps a Transformer backbone and a scalar head that predicts a reward score.

    Use `forward` to obtain scalar scores for a batch of tokenized inputs.

    The module supports multiple pooling strategies: 'cls' (first token), 'mean' (attention-masked mean), 'last' (last token).
    """

    def __init__(self, backbone: AutoModel, cfg: RewardConfig):
        super().__init__()
        self.backbone = backbone
        self.cfg = cfg
        hidden_size = getattr(backbone.config, "hidden_size", None)
        if hidden_size is None:
            # try last hidden size from config
            hidden_size = getattr(backbone.config, "d_model", None)
        if hidden_size is None:
            raise ValueError("Could not infer hidden size from backbone config")
        self.head = RewardHead(in_dim=hidden_size, hidden_dim=cfg.head_hidden, dropout=cfg.dropout, activation=cfg.head_activation)

    @classmethod
    def from_pretrained(cls, cfg: RewardConfig, local_files_only: bool = False) -> "RewardModel":
        tokenizer_name = cfg.tokenizer_name_or_path or cfg.model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True, local_files_only=local_files_only)
        backbone = AutoModel.from_pretrained(cfg.model_name_or_path, local_files_only=local_files_only)
        model = cls(backbone=backbone, cfg=cfg)
        model.tokenizer = tokenizer
        model.to(cfg.device)
        return model

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute scalar reward scores for inputs.

        Args:
            input_ids: (B, T)
            attention_mask: (B, T)
        Returns:
            scores: (B,) tensor of floats
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # some models return last_hidden_state as outputs[0]
        last_hidden = outputs.last_hidden_state
        if self.cfg.hidden_pool == "cls":
            pooled = last_hidden[:, 0, :]
        elif self.cfg.hidden_pool == "mean":
            mask = attention_mask.unsqueeze(-1) if attention_mask is not None else None
            if mask is not None:
                summed = (last_hidden * mask).sum(dim=1)
                lengths = mask.sum(dim=1).clamp(min=1)
                pooled = summed / lengths
            else:
                pooled = last_hidden.mean(dim=1)
        elif self.cfg.hidden_pool == "last":
            # gather the last non-padding token's hidden state per example
            if attention_mask is None:
                pooled = last_hidden[:, -1, :]
            else:
                lengths = attention_mask.sum(dim=1) - 1
                lengths = lengths.clamp(min=0)
                pooled = last_hidden[torch.arange(last_hidden.size(0)), lengths]
        else:
            raise ValueError("Unsupported pooling: %s" % self.cfg.hidden_pool)

        scores = self.head(pooled)
        return scores

    def save_pretrained(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        # save backbone via HF API
        try:
            self.backbone.save_pretrained(path)
        except Exception as e:
            logger.warning("backbone.save_pretrained failed: %s", e)
            torch.save(self.backbone.state_dict(), os.path.join(path, "backbone_state.bin"))
        # save head and cfg
        torch.save({"head_state": self.head.state_dict()}, os.path.join(path, "reward_head.pt"))
        with open(os.path.join(path, "reward_cfg.json"), "w") as fh:
            json.dump(self.cfg.__dict__, fh)
        # save tokenizer if present
        if hasattr(self, "tokenizer"):
            self.tokenizer.save_pretrained(path)
        logger.info("Saved RewardModel to %s", path)

    @classmethod
    def load_pretrained(cls, path: str, device: Optional[str] = None) -> "RewardModel":
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        cfg_path = os.path.join(path, "reward_cfg.json")
        if os.path.exists(cfg_path):
            with open(cfg_path, "r") as fh:
                raw = json.load(fh)
            cfg = RewardConfig(**raw)
        else:
            # fallback: try to load HF config
            cfg = RewardConfig(model_name_or_path=path)
        tokenizer = AutoTokenizer.from_pretrained(path)
        backbone = AutoModel.from_pretrained(path)
        model = cls(backbone=backbone, cfg=cfg)
        model.tokenizer = tokenizer
        head_state = torch.load(os.path.join(path, "reward_head.pt"), map_location=device)
        model.head.load_state_dict(head_state["head_state"]) if "head_state" in head_state else model.head.load_state_dict(head_state)
        model.to(device)
        return model


# ---------------------------
# Loss functions
# ---------------------------

class PairwiseLoss(nn.Module):
    """Logistic (Bradley-Terry) loss for preference pairs.

    L = -log sigmoid(s_chosen - s_rejected)
    """
    def __init__(self):
        super().__init__()

    def forward(self, s_chosen: torch.Tensor, s_rejected: torch.Tensor) -> torch.Tensor:
        diff = s_chosen - s_rejected
        loss = -F.logsigmoid(diff)
        return loss.mean()


class MarginRankingLoss(nn.Module):
    """Simple margin ranking loss: max(0, margin - (s_chosen - s_rejected))"""

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, s_chosen: torch.Tensor, s_rejected: torch.Tensor) -> torch.Tensor:
        diff = s_chosen - s_rejected
        loss = F.relu(self.margin - diff)
        return loss.mean()


# ---------------------------
# Trainer
# ---------------------------

class RewardTrainer:
    """Minimal trainer for reward models. Keeps training loop explicit so you can customize.

    Methods:
        - train: train from a pref_pairs.jsonl or PrefPairDataset
        - evaluate: compute metrics on validation set
    """

    def __init__(self, model: RewardModel, cfg: RewardConfig):
        self.model = model
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model.to(self.device)
        self.amp_scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp and self.device.type == "cuda")

    def _make_optimizer(self):
        param_groups = [
            {"params": [p for n, p in self.model.named_parameters() if p.requires_grad], "weight_decay": self.cfg.weight_decay}
        ]
        optim = torch.optim.AdamW(param_groups, lr=self.cfg.lr, eps=self.cfg.adam_eps)
        return optim

    def train(
        self,
        train_dataset: PrefPairDataset,
        val_dataset: Optional[PrefPairDataset] = None,
        resume_from: Optional[str] = None,
    ) -> None:
        os.makedirs(self.cfg.out_dir, exist_ok=True)
        train_dataloader = DataLoader(train_dataset, batch_size=self.cfg.train_batch_size, shuffle=True, collate_fn=collate_pref_pairs)
        if val_dataset is not None:
            val_dataloader = DataLoader(val_dataset, batch_size=self.cfg.eval_batch_size, shuffle=False, collate_fn=collate_pref_pairs)
        else:
            val_dataloader = None

        optimizer = self._make_optimizer()
        total_steps = math.ceil(len(train_dataloader) * self.cfg.max_epochs / max(1, self.cfg.gradient_accumulation_steps))
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.03 * total_steps), num_training_steps=total_steps)

        criterion = PairwiseLoss()

        global_step = 0
        best_val_loss = float("inf")

        for epoch in range(self.cfg.max_epochs):
            self.model.train()
            running_loss = 0.0
            for step, batch in enumerate(train_dataloader):
                chosen = batch["chosen"]
                rejected = batch["rejected"]
                # move to device
                for k in chosen:
                    chosen[k] = chosen[k].to(self.device)
                    rejected[k] = rejected[k].to(self.device)

                with torch.cuda.amp.autocast(enabled=self.amp_scaler.is_enabled()):
                    s_chosen = self.model(input_ids=chosen["input_ids"], attention_mask=chosen.get("attention_mask", None))
                    s_rejected = self.model(input_ids=rejected["input_ids"], attention_mask=rejected.get("attention_mask", None))
                    loss = criterion(s_chosen, s_rejected)
                    loss = loss / max(1, self.cfg.gradient_accumulation_steps)

                self.amp_scaler.scale(loss).backward()

                if (step + 1) % self.cfg.gradient_accumulation_steps == 0:
                    self.amp_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.amp_scaler.step(optimizer)
                    self.amp_scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1

                    running_loss += loss.item() * self.cfg.gradient_accumulation_steps

                    if global_step % 50 == 0:
                        avg = running_loss / 50
                        logger.info(f"Epoch {epoch} | step {global_step}/{total_steps} | avg loss {avg:.4f}")
                        running_loss = 0.0

                    if global_step % self.cfg.save_every == 0:
                        ckpt_path = os.path.join(self.cfg.out_dir, f"step_{global_step}")
                        self.save_checkpoint(ckpt_path, optimizer=optimizer, scheduler=scheduler, global_step=global_step)

            # end epoch
            if val_dataloader is not None:
                val_loss = self.evaluate(val_dataloader)
                logger.info(f"Epoch {epoch} finished. Val loss: {val_loss:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(os.path.join(self.cfg.out_dir, "best"), optimizer=optimizer, scheduler=scheduler, global_step=global_step)

        # final save
        self.model.save_pretrained(os.path.join(self.cfg.out_dir, "final"))

    def evaluate(self, dataloader: DataLoader) -> float:
        """Evaluate average pairwise loss on dataloader."""
        self.model.eval()
        criterion = PairwiseLoss()
        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for batch in dataloader:
                chosen = batch["chosen"]
                rejected = batch["rejected"]
                for k in chosen:
                    chosen[k] = chosen[k].to(self.device)
                    rejected[k] = rejected[k].to(self.device)
                s_chosen = self.model(input_ids=chosen["input_ids"], attention_mask=chosen.get("attention_mask", None))
                s_rejected = self.model(input_ids=rejected["input_ids"], attention_mask=rejected.get("attention_mask", None))
                loss = criterion(s_chosen, s_rejected)
                total_loss += loss.item() * s_chosen.size(0)
                count += s_chosen.size(0)
        return total_loss / max(1, count)

    def save_checkpoint(self, path: str, optimizer: Optional[torch.optim.Optimizer] = None, scheduler: Optional[Any] = None, global_step: Optional[int] = None) -> None:
        os.makedirs(path, exist_ok=True)
        # model
        self.model.save_pretrained(path)
        # optimizer & scheduler
        if optimizer is not None:
            torch.save(optimizer.state_dict(), os.path.join(path, "optimizer.pt"))
        if scheduler is not None:
            torch.save(scheduler.state_dict(), os.path.join(path, "scheduler.pt"))
        meta = {"global_step": global_step, "cfg": self.cfg.__dict__}
        with open(os.path.join(path, "meta.json"), "w") as fh:
            json.dump(meta, fh)
        logger.info("Saved checkpoint to %s", path)


# ---------------------------
# Evaluation metrics
# ---------------------------

def pairwise_accuracy(scores_chosen: np.ndarray, scores_rejected: np.ndarray) -> float:
    assert scores_chosen.shape == scores_rejected.shape
    return float((scores_chosen > scores_rejected).astype(int).mean())


# ---------------------------
# Simple command-line helpers
# ---------------------------

if __name__ == "__main__":
    # quick smoke test: load small sample and run one forward
    sample_pairs = [
        {"prompt": "Q: What is AI?", "chosen": "AI is ... good.", "rejected": "AI is bad."},
        {"prompt": "Write a haiku.", "chosen": "Autumn moonlight...", "rejected": "random text"},
    ]
    cfg = RewardConfig(model_name_or_path="distilbert-base-uncased", max_length=128, train_batch_size=2)
    rm = RewardModel.from_pretrained(cfg)
    ds = PrefPairDataset(sample_pairs, tokenizer=rm.tokenizer, cfg=cfg)
    batch = [ds[i] for i in range(len(ds))]
    collated = collate_pref_pairs(batch)
    scores_c = rm(input_ids=collated["chosen"]["input_ids"], attention_mask=collated["chosen"].get("attention_mask", None))
    scores_r = rm(input_ids=collated["rejected"]["input_ids"], attention_mask=collated["rejected"].get("attention_mask", None))
    print("scores chosen", scores_c)
    print("scores rejected", scores_r)
