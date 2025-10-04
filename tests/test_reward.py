import os
import json
import tempfile
import random
import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import project modules (ensure tests run from project root)
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.reward import (
    RewardHead,
    RewardModel,
    RewardConfig,
    PairwiseLoss,
    MarginRankingLoss,
    PrefPairDataset,
    collate_pref_pairs,
    RewardTrainer,
    load_pref_pairs_from_jsonl,
)


# -----------------------------
# Helpers for tests
# -----------------------------

class DummyBackbone(nn.Module):
    """A tiny backbone that mimics HF models for unit tests.

    - Accepts input_ids (B, T) and returns an object with `last_hidden_state` of shape (B, T, H).
    - Exposes `.config.hidden_size` used by RewardModel.
    """

    def __init__(self, vocab_size=32, hidden_size=32):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.emb = nn.Embedding(vocab_size, hidden_size)

    def forward(self, input_ids, attention_mask=None):
        # input_ids: (B, T)
        x = self.emb(input_ids)
        # return object with last_hidden_state attr
        return SimpleNamespace(last_hidden_state=x)


class FakeTokenizer:
    """A tiny tokenizer used for tests.

    Behavior:
      - Splits string into whitespace tokens
    - Maps token -> small integer via a growing vocab
    - Returns dicts similar to HF tokenizers with 'input_ids' and 'attention_mask'
    - Supports return_tensors='pt'
    """

    def __init__(self, max_length=64):
        self.vocab = {"<pad>": 0, "<sep>": 1}
        self.vocab_size = len(self.vocab)
        self.sep_token = "<sep>"
        self.pad_token_id = 0
        self.max_length = max_length

    def _token_to_id(self, token: str) -> int:
        if token not in self.vocab:
            self.vocab[token] = len(self.vocab)
            self.vocab_size = len(self.vocab)
        return self.vocab[token]

    def __call__(self, text: str, truncation=True, padding=False, max_length=None, return_tensors=None):
        max_length = max_length or self.max_length
        parts = text.strip().split()
        ids = [self._token_to_id(p) for p in parts][:max_length]
        if return_tensors == "pt":
            import torch
            ids_t = torch.tensor([ids], dtype=torch.long)
            attention = torch.ones_like(ids_t)
            return {"input_ids": ids_t, "attention_mask": attention}
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def encode(self, text: str, add_special_tokens=True):
        return self(text, return_tensors=None)["input_ids"]

    def save_pretrained(self, path: str):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "fake_tokenizer.json"), "w") as fh:
            json.dump({"vocab": self.vocab}, fh)


# -----------------------------
# Tests
# -----------------------------

def test_reward_head_forward_shapes():
    in_dim = 16
    head = RewardHead(in_dim, hidden_dim=8, dropout=0.0, activation="tanh")
    x = torch.randn(4, in_dim)
    out = head(x)
    assert out.shape == (4,), f"Expected (4,), got {out.shape}"


def test_reward_model_forward_dummy_backbone():
    cfg = RewardConfig(model_name_or_path="dummy", max_length=32)
    backbone = DummyBackbone(vocab_size=50, hidden_size=16)
    rm = RewardModel(backbone=backbone, cfg=cfg)
    # create fake input ids
    input_ids = torch.randint(0, 50, (3, 10), dtype=torch.long)
    attn = torch.ones_like(input_ids)
    scores = rm(input_ids=input_ids, attention_mask=attn)
    assert isinstance(scores, torch.Tensor)
    assert scores.shape == (3,), f"Expected (3,), got {scores.shape}"


def test_pairwise_and_margin_loss_values():
    s_chosen = torch.tensor([2.0, 3.0, 0.5])
    s_rejected = torch.tensor([1.0, 1.5, 0.4])
    pl = PairwiseLoss()
    ml = MarginRankingLoss(margin=0.5)
    loss_p = pl(s_chosen, s_rejected)
    loss_m = ml(s_chosen, s_rejected)
    # losses should be positive scalars
    assert loss_p.item() > 0
    assert loss_m.item() >= 0


def test_prefpairdataset_and_collate(tmp_path):
    # create a small pref pairs jsonl
    pairs = [
        {"prompt": "Q1", "chosen": "A good answer", "rejected": "A bad answer"},
        {"prompt": "Q2", "chosen": "Yes", "rejected": "No"},
    ]
    pfile = tmp_path / "pairs.jsonl"
    with open(pfile, "w", encoding="utf-8") as fh:
        for p in pairs:
            fh.write(json.dumps(p) + "\n")

    tok = FakeTokenizer(max_length=32)
    ds = PrefPairDataset(str(pfile), tokenizer=tok, cfg=RewardConfig(model_name_or_path="dummy", max_length=32))
    assert len(ds) == 2
    # get items and collate
    batch = [ds[i] for i in range(len(ds))]
    collated = collate_pref_pairs(batch)
    assert "chosen" in collated and "rejected" in collated
    # check tensor shapes
    chosen_ids = collated["chosen"]["input_ids"]
    rejected_ids = collated["rejected"]["input_ids"]
    assert chosen_ids.shape[0] == 2
    assert rejected_ids.shape[0] == 2


def test_reward_trainer_smoke_train(tmp_path):
    # small dataset
    pairs = [
        {"prompt": "Q1", "chosen": "good answer", "rejected": "bad"},
        {"prompt": "Q2", "chosen": "better reply", "rejected": "worse reply"},
        {"prompt": "Q3", "chosen": "positive", "rejected": "negative"},
        {"prompt": "Q4", "chosen": "correct", "rejected": "incorrect"},
    ]
    pfile = tmp_path / "pairs.jsonl"
    with open(pfile, "w", encoding="utf-8") as fh:
        for p in pairs:
            fh.write(json.dumps(p) + "\n")

    # build fake tokenizer and dummy backbone
    tok = FakeTokenizer(max_length=32)
    cfg = RewardConfig(model_name_or_path="dummy", max_length=32, train_batch_size=2, eval_batch_size=2, max_epochs=1, lr=1e-3, use_amp=False)
    backbone = DummyBackbone(vocab_size=tok.vocab_size + 200, hidden_size=16)

    rm = RewardModel(backbone=backbone, cfg=cfg)
    rm.tokenizer = tok

    # build dataset
    ds = PrefPairDataset(str(pfile), tokenizer=tok, cfg=cfg)

    trainer = RewardTrainer(model=rm, cfg=cfg)

    # run a short training (should not crash)
    trainer.train(train_dataset=ds, val_dataset=None)

    # after training, check that checkpoint saved final model exists
    final_dir = os.path.join(cfg.out_dir, "final")
    assert os.path.exists(final_dir), f"Final checkpoint directory not found at {final_dir}"


if __name__ == "__main__":
    # run tests manually
    test_reward_head_forward_shapes()
    test_reward_model_forward_dummy_backbone()
    test_pairwise_and_margin_loss_values()
    from tempfile import TemporaryDirectory
    td = TemporaryDirectory()
    test_prefpairdataset_and_collate(tempfile.TemporaryDirectory().name)
    print("All smoke tests passed")
