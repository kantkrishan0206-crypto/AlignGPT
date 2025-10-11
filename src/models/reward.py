#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Reward Model
- Pairwise preference scoring head on top of a base LM encoder/hidden states
- Supports forward scoring for single sequences and batched inputs
- Clean save/load interfaces for integration with trainers
"""

import os
import json
import torch
import torch.nn as nn
from typing import Optional, Dict, Any

from transformers import AutoModel, AutoConfig

DEFAULT_MODEL_NAME = "gpt2"


class RMHead(nn.Module):
    """
    Simple scalar head over hidden states.
    - Pooling: last token hidden or mean pooling (configurable)
    - Projection: linear -> scalar
    """

    def __init__(self, hidden_size: int, pooling: str = "last", dropout: float = 0.0):
        super().__init__()
        self.pooling = pooling
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # hidden_states: [B, T, H]
        if self.pooling == "last":
            # Use the last non-pad token per sequence
            if attention_mask is not None:
                lengths = attention_mask.sum(dim=1) - 1  # [B]
                idx = lengths.clamp(min=0).unsqueeze(-1).unsqueeze(-1).expand(-1, 1, hidden_states.size(-1))
                pooled = hidden_states.gather(1, idx).squeeze(1)  # [B, H]
            else:
                pooled = hidden_states[:, -1, :]  # last token
        elif self.pooling == "mean":
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1)  # [B, T, 1]
                summed = (hidden_states * mask).sum(dim=1)
                denom = mask.sum(dim=1).clamp(min=1e-6)
                pooled = summed / denom
            else:
                pooled = hidden_states.mean(dim=1)
        else:
            raise ValueError(f"Unsupported pooling: {self.pooling}")

        pooled = self.dropout(pooled)
        return self.score(pooled).squeeze(-1)  # [B]


class RewardModel(nn.Module):
    """
    RewardModel wraps a frozen or trainable base LM (AutoModel) and a scalar head.
    - The base model provides hidden states; we use last_hidden_state.
    - reward(x) yields a scalar preference score per input.

    Usage:
        rm = RewardModel.from_pretrained("./checkpoints/rm_gpt2")
        scores = rm(input_ids, attention_mask)
    """

    def __init__(
        self,
        base_model: nn.Module,
        hidden_size: int,
        pooling: str = "last",
        dropout: float = 0.0,
        freeze_base: bool = True,
    ):
        super().__init__()
        self.base_model = base_model
        if freeze_base:
            for p in self.base_model.parameters():
                p.requires_grad_(False)
        self.head = RMHead(hidden_size=hidden_size, pooling=pooling, dropout=dropout)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # [B, T, H]
        return self.head(hidden, attention_mask=attention_mask)

    def save_pretrained(self, output_dir: str, manifest_extra: Optional[Dict[str, Any]] = None):
        os.makedirs(output_dir, exist_ok=True)
        # Save head state dict and a small manifest; base model should be saved externally or referenced
        torch.save(self.head.state_dict(), os.path.join(output_dir, "rm_head.pt"))
        manifest = {"type": "reward_model"}
        if manifest_extra:
            manifest.update(manifest_extra)
        with open(os.path.join(output_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    @staticmethod
    def from_pretrained(
        path_or_name: str,
        base_path_or_name: Optional[str] = None,
        trust_remote_code: bool = False,
        pooling: str = "last",
        dropout: float = 0.0,
        freeze_base: bool = True,
    ) -> "RewardModel":
        """
        Load reward head and base model.
        - path_or_name: directory containing rm_head.pt OR same as base if co-located
        - base_path_or_name: HF ID or local path for base LM (AutoModel)
        """
        # Load base model and determine hidden size from config
        base_id = base_path_or_name or DEFAULT_MODEL_NAME
        config = AutoConfig.from_pretrained(base_id, trust_remote_code=trust_remote_code)
        base = AutoModel.from_pretrained(base_id, trust_remote_code=trust_remote_code)

        rm = RewardModel(
            base_model=base,
            hidden_size=config.hidden_size,
            pooling=pooling,
            dropout=dropout,
            freeze_base=freeze_base,
        )

        # Load head weights if present
        head_path = os.path.join(path_or_name, "rm_head.pt")
        if os.path.isfile(head_path):
            state = torch.load(head_path, map_location="cpu")
            rm.head.load_state_dict(state)

        return rm