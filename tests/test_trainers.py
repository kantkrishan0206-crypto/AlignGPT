import os
import yaml
import tempfile
from src.training.sft.train_rlhf import train as sft_train
from src.training.rm_trainer import train as rm_train
from src.training.ppo_trainer import train as ppo_train
from src.training.dpo_trainer import train as dpo_train

def test_sft_trainer_runs():
    cfg = {
        "model_name": "gpt2",
        "data": {"path": "data/sft.jsonl"},
        "training": {"batch_size": 2, "epochs": 1, "lr": 1e-5}
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "sft.yaml")
        with open(path, "w") as f:
            yaml.dump(cfg, f)
        sft_train(cfg, args=type("Args", (), {"config": path, "output_dir": tmp, "log_dir": tmp}))

def test_rm_trainer_runs():
    # Similar structure: create dummy pref_pairs.jsonl and config, then call rm_train()

def test_ppo_trainer_runs():
    # Create dummy prompts.jsonl and config, then call ppo_train()

def test_dpo_trainer_runs():
    # Create dummy pref_pairs.jsonl and config, then call dpo_train()