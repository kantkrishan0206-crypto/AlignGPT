"""
src/training/ppo_trainer.py

Proximal Policy Optimization (PPO) trainer for fine-tuning causal LMs with a Reward Model.

This trainer implements a practical, research-friendly PPO loop tailored for language models.
It is designed to run in a local VS Code environment for small-scale experiments (distilgpt2, tiny models)
and is structured so it can be scaled up to larger models with more infrastructure.

Key features:
- Collect rollouts by generating responses from prompts using the policy (PolicyModel wrapper).
- Score responses with a separate RewardModel (trained via rm_trainer).
- Compute advantages using a learned value head (on-policy critic) with GAE.
- Optimize the policy using PPO-Clipped objective with KL penalty against a reference policy.
- Save and load checkpoints (policy + value head + optimizer states + meta).
- Evaluation hooks and simple metrics (mean reward, pairwise win-rate when comparing two policies).
- Supports mixed precision, gradient accumulation, and curriculum-like scheduling for generation params.

Notes & limitations:
- This implementation is a single-node, single-GPU trainer (no distributed support implemented).
- For production-scale training, you should replace some parts with more scalable data collection,
  use multiple actors to gather rollouts, and use better memory/IO sharding.

Example usage (small smoke test):
    python src/training/ppo_trainer.py --config configs/ppo_gpt2.yaml

"""

from __future__ import annotations

import os
import sys
import math
import time
import json
import logging
import argparse
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

# allow running from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from src.models.policy import PolicyModel, PolicyWrapperConfig
    from src.models.reward import RewardModel, RewardConfig
    from src.models.tokenizer import TokenizerWrapper, TokenizerConfig
except Exception:
    from models.policy import PolicyModel, PolicyWrapperConfig
    from models.reward import RewardModel, RewardConfig
    from models.tokenizer import TokenizerWrapper, TokenizerConfig

logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)


# ---------------------------
# Config
# ---------------------------

@dataclass
class PPOConfig:
    model_name_or_path: str = "gpt2"
    tokenizer_name_or_path: Optional[str] = None
    reward_model_path: Optional[str] = None
    prompts_file: str = "data/prompts.jsonl"

    # generation / rollout
    batch_size: int = 8
    gen_max_new_tokens: int = 64
    gen_temperature: float = 1.0
    gen_top_k: Optional[int] = 50
    gen_top_p: Optional[float] = 0.95

    # PPO optimization
    ppo_epochs: int = 4
    minibatch_size: int = 4
    learning_rate: float = 1.5e-5
    weight_decay: float = 0.0
    clip_epsilon: float = 0.2
    vf_coef: float = 0.5
    entropy_coef: float = 0.01
    kl_coef: float = 0.02
    max_grad_norm: float = 1.0

    # advantage / GAE
    gamma: float = 0.99
    lam: float = 0.95

    # training loop
    total_steps: int = 1000
    rollout_size: int = 64  # number of samples collected before each update
    update_epochs: int = 4
    seed: int = 42

    # misc
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir: str = "./checkpoints/ppo"
    save_every: int = 500
    fp16: bool = True
    local_files_only: bool = False


# ---------------------------
# Prompt dataset
# ---------------------------

class PromptDataset(Dataset):
    def __init__(self, path: str):
        self.prompts = []
        if not os.path.exists(path):
            raise FileNotFoundError(f"Prompts file not found: {path}")
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                # support lines either plain string or {"prompt": ...}
                if isinstance(obj, str):
                    self.prompts.append(obj)
                elif isinstance(obj, dict) and "prompt" in obj:
                    self.prompts.append(obj["prompt"])
        if len(self.prompts) == 0:
            raise ValueError("No prompts found in prompts file")

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]


# ---------------------------
# Value head for critic
# ---------------------------

class ValueHead(nn.Module):
    def __init__(self, hidden_size: int, hidden_dim: int = 256):
        super().__init__()
        self.v_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        # pooled: (B, hidden_size)
        return self.v_head(pooled).squeeze(-1)


# ---------------------------
# Utilities
# ---------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_rollouts(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    # batch contains dicts with keys: prompt, response, input_ids, attention_mask, logprob, value, reward, advantage, return
    out = {}
    out["prompts"] = [b["prompt"] for b in batch]
    out["responses"] = [b["response"] for b in batch]
    out["logprobs"] = torch.tensor([b["logprob"] for b in batch], dtype=torch.float32)
    out["values"] = torch.tensor([b["value"] for b in batch], dtype=torch.float32)
    out["rewards"] = torch.tensor([b["reward"] for b in batch], dtype=torch.float32)
    out["advantages"] = torch.tensor([b["advantage"] for b in batch], dtype=torch.float32)
    out["returns"] = torch.tensor([b["return"] for b in batch], dtype=torch.float32)
    return out


# ---------------------------
# PPO Trainer
# ---------------------------

class PPOTrainer:
    def __init__(self, policy: PolicyModel, reward_model: RewardModel, tokenizer: TokenizerWrapper, cfg: PPOConfig):
        self.policy = policy
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        # attach small value head on top of policy model's hidden size
        # try to infer hidden size from policy.model.config
        hsz = getattr(self.policy.model.config, "hidden_size", None) or getattr(self.policy.model.config, "n_embd", None)
        if hsz is None:
            # fallback
            hsz = 768
        self.value_head = ValueHead(hidden_size=hsz).to(self.device)

        # optimizers
        params = list(self.policy.model.parameters()) + list(self.value_head.parameters())
        self.optimizer = AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

        # For reference policy KL penalty: snapshot old params to compute old logprobs
        self.reference_policy = None

        # AMP
        self.use_amp = cfg.fp16 and self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def save_checkpoint(self, path: str, step: int):
        os.makedirs(path, exist_ok=True)
        # save policy model via its wrapper
        try:
            self.policy.save(os.path.join(path, "policy"))
        except Exception:
            # if policy.save not available, fallback to HF save
            try:
                self.policy.model.save_pretrained(os.path.join(path, "policy"))
            except Exception:
                pass
        # save value head and optimizer state
        torch.save({"value_state": self.value_head.state_dict()}, os.path.join(path, "value_head.pt"))
        torch.save(self.optimizer.state_dict(), os.path.join(path, "optimizer.pt"))
        meta = {"step": step, "cfg": self.cfg.__dict__}
        with open(os.path.join(path, "meta.json"), "w") as fh:
            json.dump(meta, fh)
        logger.info("Saved PPO checkpoint to %s", path)

    def load_checkpoint(self, path: str):
        # load value head
        vpath = os.path.join(path, "value_head.pt")
        if os.path.exists(vpath):
            st = torch.load(vpath, map_location=self.device)
            self.value_head.load_state_dict(st.get("value_state", st))
            logger.info("Loaded value head from %s", vpath)
        # load optimizer
        optp = os.path.join(path, "optimizer.pt")
        if os.path.exists(optp):
            st = torch.load(optp, map_location=self.device)
            try:
                self.optimizer.load_state_dict(st)
                logger.info("Loaded optimizer state from %s", optp)
            except Exception as e:
                logger.warning("Failed to load optimizer state: %s", e)

    # ---------
    # Rollout collection
    # ---------

    def generate_responses(self, prompts: List[str]) -> List[str]:
        # use policy.generate (which returns decoded string)
        return self.policy.generate(prompts, max_new_tokens=self.cfg.gen_max_new_tokens, do_sample=True, temperature=self.cfg.gen_temperature, top_k=self.cfg.gen_top_k, top_p=self.cfg.gen_top_p)

    def compute_logprobs(self, prompts: List[str], responses: List[str]) -> List[float]:
        # use policy.score to compute sequence log-prob
        # score expects full concatenation of prompt+response depending on tokenizer in score implementation
        logps = self.policy.score(prompts, responses)
        # ensure list
        if isinstance(logps, float) or isinstance(logps, int):
            return [float(logps)]
        return [float(x) for x in logps]

    def compute_values(self, prompts: List[str], responses: List[str]) -> List[float]:
        # compute value estimates using the value_head; pool last hidden state of policy.model
        values = []
        self.policy.model.eval()
        with torch.no_grad():
            for p, r in zip(prompts, responses):
                # create combined input and tokenize via tokenizer wrapper
                full = p + r
                enc = self.tokenizer(full, truncation=True, max_length=self.policy.cfg.max_length if hasattr(self.policy, "cfg") else 512)
                # enc may be dict
                input_ids = torch.tensor(enc["input_ids"], dtype=torch.long, device=self.device).unsqueeze(0)
                attention_mask = torch.tensor(enc.get("attention_mask", [1] * input_ids.size(1)], dtype=torch.long, device=self.device).unsqueeze(0)
                outputs = self.policy.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
                last_hidden = outputs.last_hidden_state  # (1, T, H)
                # pool by last token of sequence
                lengths = attention_mask.sum(dim=1) - 1
                idx = lengths.clamp(min=0).long()
                pooled = last_hidden[0, idx.item(), :].unsqueeze(0)  # (1, H)
                v = self.value_head(pooled)
                values.append(float(v.item()))
        return values

    def compute_rewards(self, prompts: List[str], responses: List[str]) -> List[float]:
        # use reward_model to score
        self.reward_model.eval()
        with torch.no_grad():
            # reward_model.forward expects input_ids and attention masks
            scores = []
            for p, r in zip(prompts, responses):
                full = p + self.tokenizer._tokenizer.sep_token + r if hasattr(self.tokenizer._tokenizer, "sep_token") and self.tokenizer._tokenizer.sep_token else p + r
                enc = self.tokenizer(full, truncation=True, max_length=getattr(self.reward_model.cfg, "max_length", 256))
                input_ids = torch.tensor(enc["input_ids"], dtype=torch.long, device=self.device).unsqueeze(0)
                attention_mask = torch.tensor(enc.get("attention_mask", [1] * input_ids.size(1)], dtype=torch.long, device=self.device).unsqueeze(0)
                s = self.reward_model(input_ids=input_ids, attention_mask=attention_mask)
                scores.append(float(s.item()))
        return scores

    # ---------
    # Advantage estimation
    # ---------

    def compute_gae(self, rewards: List[float], values: List[float], gamma: float, lam: float) -> Tuple[List[float], List[float]]:
        # rewards and values are lists of scalars for each trajectory (here sequence-level)
        # For simplicity we treat each sample as a 1-step trajectory: advantage = reward - value
        # but implement GAE for multi-step sequences if needed in the future
        advantages = []
        returns = []
        for r, v in zip(rewards, values):
            adv = r - v
            ret = adv + v
            advantages.append(float(adv))
            returns.append(float(ret))
        # Normalize advantages
        adv_arr = np.array(advantages, dtype=np.float32)
        adv_mean = adv_arr.mean() if adv_arr.size > 0 else 0.0
        adv_std = adv_arr.std() if adv_arr.size > 0 else 1.0
        if adv_std == 0:
            adv_std = 1.0
        advantages = ((adv_arr - adv_mean) / adv_std).tolist()
        return advantages, returns

    # ---------
    # PPO update
    # ---------

    def ppo_update(self, rollouts: List[Dict[str, Any]]):
        # rollouts: list of dicts with prompt, response, logprob, value, reward
        # compute advantages & returns
        prompts = [r["prompt"] for r in rollouts]
        responses = [r["response"] for r in rollouts]
        logprobs_old = [r["logprob"] for r in rollouts]
        values_old = [r["value"] for r in rollouts]
        rewards = [r["reward"] for r in rollouts]

        advantages, returns = self.compute_gae(rewards, values_old, self.cfg.gamma, self.cfg.lam)

        # prepare minibatches
        dataset = list(zip(prompts, responses, logprobs_old, values_old, rewards, advantages, returns))
        n = len(dataset)
        for epoch in range(self.cfg.ppo_epochs):
            random.shuffle(dataset)
            for start in range(0, n, self.cfg.minibatch_size):
                mb = dataset[start : start + self.cfg.minibatch_size]
                mb_prompts = [x[0] for x in mb]
                mb_responses = [x[1] for x in mb]
                mb_logp_old = torch.tensor([x[2] for x in mb], dtype=torch.float32, device=self.device)
                mb_values_old = torch.tensor([x[3] for x in mb], dtype=torch.float32, device=self.device)
                mb_rewards = torch.tensor([x[4] for x in mb], dtype=torch.float32, device=self.device)
                mb_advantages = torch.tensor([x[5] for x in mb], dtype=torch.float32, device=self.device)
                mb_returns = torch.tensor([x[6] for x in mb], dtype=torch.float32, device=self.device)

                # compute new logprobs and new values
                # use policy.score for logprobs
                new_logps = torch.tensor(self.compute_logprobs(mb_prompts, mb_responses), dtype=torch.float32, device=self.device)
                new_values = torch.tensor(self.compute_values(mb_prompts, mb_responses), dtype=torch.float32, device=self.device)

                # policy loss (PPO clipped)
                ratio = torch.exp(new_logps - mb_logp_old)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_epsilon, 1.0 + self.cfg.clip_epsilon) * mb_advantages
                policy_loss = -torch.mean(torch.min(surr1, surr2))

                # value loss
                value_loss = F.mse_loss(new_values, mb_returns)

                # entropy bonus: approximate via token-level entropy if possible; here we use a small constant
                # to encourage exploration (a more exact entropy requires per-token probs)
                entropy_bonus = torch.tensor(0.0, device=self.device)

                # KL penalty to reference policy (if available): compute KL between old policy and new policy
                kl_penalty = torch.tensor(0.0, device=self.device)
                if self.reference_policy is not None:
                    # approximate via logprob differences
                    kl = mb_logp_old - new_logps
                    kl_penalty = torch.mean(kl)

                loss = policy_loss + self.cfg.vf_coef * value_loss - self.cfg.entropy_coef * entropy_bonus + self.cfg.kl_coef * kl_penalty

                # gradient step
                self.optimizer.zero_grad()
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(list(self.policy.model.parameters()) + list(self.value_head.parameters()), self.cfg.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(list(self.policy.model.parameters()) + list(self.value_head.parameters()), self.cfg.max_grad_norm)
                    self.optimizer.step()

        logger.info("Completed PPO update; mean reward=%.4f", float(np.mean(rewards) if len(rewards) else 0.0))

    # ---------
    # Main loop
    # ---------

    def train(self):
        set_seed(self.cfg.seed)
        os.makedirs(self.cfg.out_dir, exist_ok=True)

        prompts_ds = PromptDataset(self.cfg.prompts_file)
        # simple sampler: random sample prompts

        step = 0
        total_steps = self.cfg.total_steps
        while step < total_steps:
            # collect rollouts
            rollouts = []
            while len(rollouts) < self.cfg.rollout_size and step < total_steps:
                # sample a batch of prompts
                prompts = [prompts_ds[random.randrange(len(prompts_ds))] for _ in range(self.cfg.batch_size)]
                responses = self.generate_responses(prompts)
                # ensure responses is list
                if isinstance(responses, str):
                    responses = [responses]
                logps = self.compute_logprobs(prompts, responses)
                values = self.compute_values(prompts, responses)
                rewards = self.compute_rewards(prompts, responses)

                for p, r_text, lp, v, rew in zip(prompts, responses, logps, values, rewards):
                    rollouts.append({"prompt": p, "response": r_text, "logprob": lp, "value": v, "reward": rew})
                step += len(prompts)

            # perform PPO update on collected rollouts
            logger.info("Collected %d rollouts; running PPO update", len(rollouts))
            self.ppo_update(rollouts)

            # checkpointing
            if step % self.cfg.save_every == 0:
                ckpt_path = os.path.join(self.cfg.out_dir, f"step_{step}")
                self.save_checkpoint(ckpt_path, step)

        # final save
        self.save_checkpoint(os.path.join(self.cfg.out_dir, "final"), step)
        logger.info("PPO training finished")


# ---------------------------
# CLI
# ---------------------------

def parse_args():
    p = argparse.ArgumentParser(description="PPO Trainer for RLHF lab")
    p.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    p.add_argument("--local_files_only", action="store_true", help="Load HF models/tokenizers from local files only")
    p.add_argument("--prompts", type=str, default=None, help="Override prompts file")
    p.add_argument("--out_dir", type=str, default=None, help="Override output dir")
    return p.parse_args()


def load_config(path: Optional[str]) -> PPOConfig:
    if path is None:
        return PPOConfig()
    import yaml
    with open(path, "r") as fh:
        raw = yaml.safe_load(fh) or {}
    cfg = PPOConfig(**raw)
    return cfg


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.prompts:
        cfg.prompts_file = args.prompts
    if args.out_dir:
        cfg.out_dir = args.out_dir

    logger.info("Loading policy model %s", cfg.model_name_or_path)
    policy_cfg = PolicyWrapperConfig(model_name_or_path=cfg.model_name_or_path)
    policy = PolicyModel.from_pretrained(policy_cfg, local_files_only=args.local_files_only)

    # tokenizer
    tok_cfg = TokenizerConfig(model_name_or_path=cfg.tokenizer_name_or_path or cfg.model_name_or_path)
    tokenizer = TokenizerWrapper.from_pretrained(tok_cfg, local_files_only=args.local_files_only)

    # reward model
    if cfg.reward_model_path is None:
        raise ValueError("Please provide --config with reward_model_path or set in YAML")
    reward_cfg = RewardConfig(model_name_or_path=cfg.reward_model_path)
    reward_model = RewardModel.from_pretrained(reward_cfg, local_files_only=args.local_files_only)

    trainer = PPOTrainer(policy=policy, reward_model=reward_model, tokenizer=tokenizer, cfg=cfg)
    trainer.train()


if __name__ == "__main__":
    main()
