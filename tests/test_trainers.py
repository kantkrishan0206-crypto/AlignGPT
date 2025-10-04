"""
tests/test_trainers.py

Integration-style unit tests for the trainer scripts in rlhf-lab.

These tests are designed to be lightweight and fast for local VS Code runs by
mocking heavy external dependencies (Hugging Face Transformers, large models)
with small, deterministic stub implementations.

They verify that the trainers' high-level control flow runs, that checkpoint
folders are created, and that basic logging and save hooks execute.

Run with:
    pytest -q

Note: these tests intentionally monkeypatch `from_pretrained` and other heavy
calls so they do not download large models or require GPUs.
"""

import os
import sys
import json
import shutil
import tempfile
import types
from unittest import mock

import pytest

# Ensure project src is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Import trainer modules
from src.training import sft_trainer as sft_trainer_module
from src.training import rm_trainer as rm_trainer_module
from src.training import ppo_trainer as ppo_trainer_module
from src.training import dpo_trainer as dpo_trainer_module

# ------------------
# Fake lightweight components used to patch heavy HF dependencies
# ------------------

class FakeTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.sep_token = "[SEP]"
        self.cfg = types.SimpleNamespace(bos_token="", eos_token="")

    def __call__(self, text, truncation=True, max_length=512, return_tensors=None, **kwargs):
        # simple whitespace tokenization
        if isinstance(text, list):
            ids = [[hash(w) % 1000 for w in t.split()] for t in text]
            return {"input_ids": ids, "attention_mask": [[1]*len(x) for x in ids]}
        else:
            ids = [hash(w) % 1000 for w in text.split()]
            return {"input_ids": ids, "attention_mask": [1]*len(ids)}

    def encode(self, text, add_special_tokens=False):
        return [hash(w) % 1000 for w in text.split()]

    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "tokenizer.json"), "w") as fh:
            json.dump({"note": "fake tokenizer"}, fh)

class FakeCausalLM:
    def __init__(self, *args, **kwargs):
        self.config = types.SimpleNamespace(hidden_size=16, n_embd=16)
        self.device = 'cpu'

    def generate(self, **kwargs):
        # return a tensor-like structure; trainers expect list/str
        inputs = kwargs.get('input_ids') if 'input_ids' in kwargs else kwargs
        # simple echo
        return ["generated text"]

    def __call__(self, input_ids=None, attention_mask=None, return_dict=True, labels=None):
        # Return object with logits and last_hidden_state
        class O:
            pass
        o = O()
        batch = 1
        seq = 4
        hid = 16
        import torch
        o.logits = torch.zeros((batch, seq, hid))
        o.last_hidden_state = torch.zeros((batch, seq, hid))
        return o

    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "pytorch_model.bin"), "w") as fh:
            fh.write("fake model")

    def to(self, device):
        self.device = device
        return self

class FakePolicyModel:
    def __init__(self, *args, **kwargs):
        self.model = FakeCausalLM()
        self.cfg = types.SimpleNamespace(max_length=128)

    @classmethod
    def from_pretrained(cls, cfg, local_files_only=False, **kwargs):
        return cls()

    def generate(self, prompts, max_new_tokens=32, do_sample=False, temperature=1.0, top_k=None, top_p=None):
        # return a response per prompt
        if isinstance(prompts, list):
            return [f"resp for: {p}" for p in prompts]
        return f"resp for: {prompts}"

    def score(self, prompts, responses):
        # simple deterministic logprob: -len(response)
        if isinstance(prompts, list):
            return [-(len(r.split())) for r in responses]
        return -(len(responses.split()))

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "policy_saved.txt"), "w") as fh:
            fh.write("saved")

class FakeRewardModel:
    def __init__(self, *args, **kwargs):
        self.cfg = types.SimpleNamespace(max_length=256)
        self.tokenizer = FakeTokenizer()

    @classmethod
    def from_pretrained(cls, cfg, local_files_only=False):
        return cls()

    def __call__(self, input_ids=None, attention_mask=None):
        # returns a scalar tensor-like object
        import torch
        return torch.tensor([1.0])

    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "reward_saved.txt"), "w") as fh:
            fh.write("saved")

# ------------------
# Fixtures
# ------------------

@pytest.fixture
def tmp_dir(tmp_path):
    d = tmp_path / "rlhf_test"
    d.mkdir()
    yield str(d)

# ------------------
# SFT trainer test
# ------------------

def test_sft_trainer_runs_smoke(tmp_dir, monkeypatch):
    # prepare small sft.jsonl
    sft_path = os.path.join(tmp_dir, "sft.jsonl")
    data = [
        {"prompt": "Q: hello\nA:", "response": "Hi there."},
        {"prompt": "Q: sum 2 and 3\nA:", "response": "5."}
    ]
    with open(sft_path, "w") as fh:
        for obj in data:
            fh.write(json.dumps(obj) + "\n")

    # monkeypatch heavy HF components
    monkeypatch.setattr(sft_trainer_module, 'AutoModelForCausalLM', lambda *a, **k: FakeCausalLM())
    monkeypatch.setattr(sft_trainer_module, 'AutoTokenizer', lambda *a, **k: FakeTokenizer())

    # patch Trainer to a fake that just records train called
    class FakeTrainer:
        def __init__(self, model, args, train_dataset, data_collator, tokenizer):
            self.model = model
            self.args = args
            self.train_dataset = train_dataset
            self.called = False
        def train(self):
            # simulate a training pass by iterating dataset and ensuring collator works
            for i in range(min(2, len(self.train_dataset))):
                _ = self.train_dataset[i]
            self.called = True
    monkeypatch.setattr(sft_trainer_module, 'Trainer', FakeTrainer)

    # run main with overrides
    # monkeypatch parse_args to set config values
    Args = types.SimpleNamespace
    def fake_parse_args():
        return Args(config=None, train_file=sft_path, output_dir=os.path.join(tmp_dir, "out_sft"), local_files_only=True)
    monkeypatch.setattr(sft_trainer_module, 'parse_args', fake_parse_args)

    # run
    sft_trainer_module.main()

    # assert output dir created
    assert os.path.exists(os.path.join(tmp_dir, "out_sft"))

# ------------------
# RM trainer test
# ------------------

def test_rm_trainer_runs_smoke(tmp_dir, monkeypatch):
    # prepare pref_pairs.jsonl
    pr_path = os.path.join(tmp_dir, "pref_pairs.jsonl")
    pairs = [
        {"prompt": "Q: define hi", "chosen": "Hi is greeting.", "rejected": "No."},
        {"prompt": "Write a poem", "chosen": "Roses are red.", "rejected": "Spam."}
    ]
    with open(pr_path, "w") as fh:
        for p in pairs:
            fh.write(json.dumps(p) + "\n")

    # monkeypatch TokenizerWrapper and RewardModel
    monkeypatch.setattr(rm_trainer_module, 'TokenizerWrapper', types.SimpleNamespace(from_pretrained=lambda cfg, local_files_only=False: types.SimpleNamespace(_tokenizer=FakeTokenizer())))
    monkeypatch.setattr(rm_trainer_module, 'RewardModel', types.SimpleNamespace(from_pretrained=lambda cfg, local_files_only=False: FakeRewardModel()))

    # fake parse_args
    Args = types.SimpleNamespace
    def fake_parse_args():
        return Args(config=None, train_file=pr_path, val_file=None, out_dir=os.path.join(tmp_dir, "out_rm"), max_items=None, resume=None, local_files_only=True)
    monkeypatch.setattr(rm_trainer_module, 'parse_args', fake_parse_args)

    # run
    rm_trainer_module.main()

    # check final saved model
    assert os.path.exists(os.path.join(tmp_dir, "out_rm", "final"))

# ------------------
# PPO trainer test
# ------------------

def test_ppo_trainer_smoke(tmp_dir, monkeypatch):
    # prepare prompts.jsonl
    prompts_path = os.path.join(tmp_dir, "prompts.jsonl")
    prompts = [{"prompt": "Say hello."}, {"prompt": "Write a haiku."}]
    with open(prompts_path, "w") as fh:
        for p in prompts:
            fh.write(json.dumps(p) + "\n")

    # monkeypatch PolicyModel, RewardModel, TokenizerWrapper
    monkeypatch.setattr(ppo_trainer_module, 'PolicyModel', FakePolicyModel)
    monkeypatch.setattr(ppo_trainer_module, 'RewardModel', types.SimpleNamespace(from_pretrained=lambda cfg, local_files_only=False: FakeRewardModel()))
    monkeypatch.setattr(ppo_trainer_module, 'TokenizerWrapper', types.SimpleNamespace(from_pretrained=lambda cfg, local_files_only=False: FakeTokenizer()))

    # fake config loader
    def fake_load_config(path):
        cfg = ppo_trainer_module.PPOConfig()
        cfg.prompts_file = prompts_path
        cfg.reward_model_path = "unused"
        cfg.total_steps = 4
        cfg.rollout_size = 4
        cfg.batch_size = 2
        cfg.save_every = 2
        cfg.out_dir = os.path.join(tmp_dir, "out_ppo")
        return cfg
    monkeypatch.setattr(ppo_trainer_module, 'load_config', fake_load_config)

    # run main
    ppo_trainer_module.main()

    # check checkpoints
    assert os.path.exists(os.path.join(tmp_dir, "out_ppo"))

# ------------------
# DPO trainer test
# ------------------

def test_dpo_trainer_smoke(tmp_dir, monkeypatch):
    # prepare pref pairs
    pr_path = os.path.join(tmp_dir, "pref_pairs.jsonl")
    pairs = [
        {"prompt": "Q: hi", "chosen": "Hi.", "rejected": "No."},
    ]
    with open(pr_path, "w") as fh:
        for p in pairs:
            fh.write(json.dumps(p) + "\n")

    # monkeypatch TokenizerWrapper and PolicyModel
    monkeypatch.setattr(dpo_trainer_module, 'TokenizerWrapper', types.SimpleNamespace(from_pretrained=lambda cfg, local_files_only=False: FakeTokenizer()))
    monkeypatch.setattr(dpo_trainer_module, 'PolicyModel', FakePolicyModel)

    # fake load_config
    def fake_load_config(path):
        cfg = dpo_trainer_module.DPOConfig()
        cfg.train_file = pr_path
        cfg.out_dir = os.path.join(tmp_dir, "out_dpo")
        cfg.num_train_epochs = 1
        cfg.per_device_train_batch_size = 1
        cfg.max_items = None
        cfg.local_files_only = True
        cfg.logging_steps = 1
        cfg.save_steps = 1
        return cfg
    monkeypatch.setattr(dpo_trainer_module, 'load_config', fake_load_config)

    # run main
    dpo_trainer_module.main()

    assert os.path.exists(os.path.join(tmp_dir, "out_dpo"))
