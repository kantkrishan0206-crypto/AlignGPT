#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PPO Trainer (RLHF)
File: src/training/ppo_trainer.py

Purpose:
- Optimize a supervised-fine-tuned policy via PPO against a trained reward model
- Config-driven with YAML + CLI overrides
- Supports batched generation, KL control, mixed precision, logging, checkpointing
- Clean integration with existing project modules (utils/, models/, eval/)
"""

import os
import sys
import time
import math
import json
import yaml
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import torch
from torch.nn.utils import clip_grad_norm_
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
)
from trl import PPOTrainer, PPOConfig

# Local modules
try:
    from src.utils.seed import set_global_seed
    from src.utils.logging import init_logging, get_logger
    from src.utils.checkpoints import save_checkpoint_safe, load_checkpoint_safe
    from src.models.policy import load_policy_model
    from src.models.reward import RewardModel
    from src.models.tokenizer import get_tokenizer
    from src.eval.metrics import compute_perplexity
except Exception as e:
    print(f"[ImportError] Verify your project structure and PYTHONPATH. Details: {e}")
    sys.exit(1)


# ------------------------------
# Config dataclasses and helpers
# ------------------------------

@dataclass
class PPORunConfig:
    # Model and tokenizer
    model_name: str = "gpt2"
    policy_ckpt: str = "./checkpoints/sft_gpt2"     # path to SFT policy
    reward_ckpt: str = "./checkpoints/rm_gpt2"      # path to trained reward model
    trust_remote_code: bool = False

    # Data
    prompts_path: str = "data/prompts.jsonl"
    prompt_field: str = "prompt"
    max_prompt_length: int = 256
    max_response_length: int = 256

    # PPO training
    batch_size: int = 8
    learning_rate: float = 1e-5
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    weight_decay: float = 0.0
    target_kl: float = 0.1
    init_kl_coef: float = 0.2
    kl_penalty: str = "kl"  # "kl" or "abs"
    ppo_epochs: int = 1
    mini_batch_size: int = 8
    cliprange: float = 0.2

    # Generation
    do_sample: bool = True
    top_p: float = 0.9
    top_k: int = 0
    temperature: float = 1.0
    repetition_penalty: float = 1.0
    eos_token_id: Optional[int] = None

    # Run / infra
    seed: int = 42
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = False
    grad_clip_norm: float = 1.0
    device: str = "auto"  # "auto", "cuda", "cpu"
    dataloader_num_workers: int = 0

    # Logging & checkpoints
    output_dir: str = "./checkpoints/ppo_gpt2"
    log_dir: str = "./logs"
    save_steps: int = 200
    save_total_limit: int = 3
    report_to: List[str] = None  # e.g., ["wandb"]
    run_name: Optional[str] = None

    # Evaluation
    eval_sample_size: int = 128
    eval_every_steps: int = 500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RLHF PPO Trainer")
    parser.add_argument("--config", type=str, default="configs/ppo_gpt2.yaml", help="Path to YAML config")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output dir for checkpoints")
    parser.add_argument("--log_dir", type=str, default=None, help="Override logging directory")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint if available")
    parser.add_argument("--override", type=str, nargs="*", default=None, help="Key=Value overrides for config")
    parser.add_argument("--run_name", type=str, default=None, help="Experiment/run name")
    return parser.parse_args()


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


def load_yaml_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def apply_overrides(cfg: Dict[str, Any], overrides: Optional[List[str]]):
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


def build_generation_config(cfg: PPORunConfig, tokenizer: AutoTokenizer) -> GenerationConfig:
    return GenerationConfig(
        do_sample=cfg.do_sample,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        temperature=cfg.temperature,
        repetition_penalty=cfg.repetition_penalty,
        max_new_tokens=cfg.max_response_length,
        eos_token_id=cfg.eos_token_id or tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        return_dict_in_generate=True,
        output_scores=True,
    )


def load_prompts(prompts_path: str, prompt_field: str) -> List[str]:
    """
    Load prompts from a JSONL file with a 'prompt' field.
    For large files, consider streaming or sharding in future expansions.
    """
    ds = load_dataset("json", data_files=prompts_path, split="train")
    prompts = []
    for row in ds:
        val = row.get(prompt_field, None)
        if val is None:
            # fallback to common keys
            val = row.get("instruction") or row.get("text")
        if val:
            prompts.append(val)
    return prompts


def chunked(iterable: List[Any], size: int) -> List[List[Any]]:
    return [iterable[i : i + size] for i in range(0, len(iterable), size)]


def compute_rewards_for_responses(
    reward_model: RewardModel,
    tokenizer: AutoTokenizer,
    responses: List[str],
    device: torch.device,
    max_length: int = 512,
) -> List[float]:
    """
    Score responses with the reward model. Assumes RewardModel.forward returns scalar scores.
    """
    reward_model.eval()
    rewards = []
    with torch.no_grad():
        for resp in responses:
            tokens = tokenizer(resp, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")
            input_ids = tokens["input_ids"].to(device)
            attention_mask = tokens["attention_mask"].to(device)
            score = reward_model(input_ids, attention_mask)
            # Convert to float
            if isinstance(score, torch.Tensor):
                score = score.squeeze().item()
            rewards.append(float(score))
    return rewards


def save_manifest(output_dir: str, payload: Dict[str, Any]):
    path = os.path.join(output_dir, "manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def get_ppo_config_from_dict(cfg: Dict[str, Any]) -> PPOConfig:
    """
    Map your YAML dictionary to TRL's PPOConfig fields.
    """
    ppo_args = cfg.get("ppo", {})
    return PPOConfig(
        model_name=cfg.get("model_name", "gpt2"),
        learning_rate=ppo_args.get("learning_rate", 1e-5),
        batch_size=ppo_args.get("batch_size", 8),
        mini_batch_size=ppo_args.get("mini_batch_size", 8),
        ppo_epochs=ppo_args.get("ppo_epochs", 1),
        cliprange=ppo_args.get("cliprange", 0.2),
        target_kl=ppo_args.get("target_kl", 0.1),
        init_kl_coef=ppo_args.get("init_kl_coef", 0.2),
        kl_penalty=ppo_args.get("kl_penalty", "kl"),
        seed=cfg.get("seed", 42),
        use_fp16=cfg.get("fp16", True),
        use_bf16=cfg.get("bf16", False),
        gradient_accumulation_steps=ppo_args.get("gradient_accumulation_steps", 1),
        log_with=cfg.get("report_to", None),
        project_kwargs={"run_name": cfg.get("run_name", None)} if cfg.get("run_name") else None,
    )


def main():
    args = parse_args()

    # Load config
    cfg_dict = load_yaml_config(args.config)
    cfg_dict = apply_overrides(cfg_dict, args.override)
    if args.output_dir:
        cfg_dict["output_dir"] = args.output_dir
    if args.log_dir:
        cfg_dict["log_dir"] = args.log_dir
    if args.run_name:
        cfg_dict["run_name"] = args.run_name

    # Convert dict to local config dataclass for convenience
    rc = PPORunConfig(
        model_name=cfg_dict.get("model_name", "gpt2"),
        policy_ckpt=cfg_dict.get("policy_ckpt", "./checkpoints/sft_gpt2"),
        reward_ckpt=cfg_dict.get("reward_ckpt", "./checkpoints/rm_gpt2"),
        trust_remote_code=cfg_dict.get("trust_remote_code", False),
        prompts_path=cfg_dict.get("data", {}).get("prompts_path", "data/prompts.jsonl"),
        prompt_field=cfg_dict.get("data", {}).get("prompt_field", "prompt"),
        max_prompt_length=cfg_dict.get("data", {}).get("max_prompt_length", 256),
        max_response_length=cfg_dict.get("data", {}).get("max_response_length", 256),
        batch_size=cfg_dict.get("ppo", {}).get("batch_size", 8),
        learning_rate=cfg_dict.get("ppo", {}).get("learning_rate", 1e-5),
        adam_beta1=cfg_dict.get("optim", {}).get("beta1", 0.9),
        adam_beta2=cfg_dict.get("optim", {}).get("beta2", 0.95),
        adam_eps=cfg_dict.get("optim", {}).get("eps", 1e-8),
        weight_decay=cfg_dict.get("optim", {}).get("weight_decay", 0.0),
        target_kl=cfg_dict.get("ppo", {}).get("target_kl", 0.1),
        init_kl_coef=cfg_dict.get("ppo", {}).get("init_kl_coef", 0.2),
        kl_penalty=cfg_dict.get("ppo", {}).get("kl_penalty", "kl"),
        ppo_epochs=cfg_dict.get("ppo", {}).get("ppo_epochs", 1),
        mini_batch_size=cfg_dict.get("ppo", {}).get("mini_batch_size", 8),
        cliprange=cfg_dict.get("ppo", {}).get("cliprange", 0.2),
        do_sample=cfg_dict.get("gen", {}).get("do_sample", True),
        top_p=cfg_dict.get("gen", {}).get("top_p", 0.9),
        top_k=cfg_dict.get("gen", {}).get("top_k", 0),
        temperature=cfg_dict.get("gen", {}).get("temperature", 1.0),
        repetition_penalty=cfg_dict.get("gen", {}).get("repetition_penalty", 1.0),
        eos_token_id=cfg_dict.get("gen", {}).get("eos_token_id", None),
        seed=cfg_dict.get("seed", 42),
        fp16=cfg_dict.get("fp16", True),
        bf16=cfg_dict.get("bf16", False),
        gradient_checkpointing=cfg_dict.get("training", {}).get("gradient_checkpointing", False),
        grad_clip_norm=cfg_dict.get("training", {}).get("grad_clip_norm", 1.0),
        device=cfg_dict.get("device", "auto"),
        dataloader_num_workers=cfg_dict.get("training", {}).get("dataloader_num_workers", 0),
        output_dir=cfg_dict.get("output_dir", "./checkpoints/ppo_gpt2"),
        log_dir=cfg_dict.get("log_dir", "./logs"),
        save_steps=cfg_dict.get("training", {}).get("save_steps", 200),
        save_total_limit=cfg_dict.get("training", {}).get("save_total_limit", 3),
        report_to=cfg_dict.get("report_to", []),
        run_name=cfg_dict.get("run_name", None),
        eval_sample_size=cfg_dict.get("eval", {}).get("sample_size", 128),
        eval_every_steps=cfg_dict.get("eval", {}).get("every_steps", 500),
    )

    # Init
    ensure_dirs(rc.output_dir, rc.log_dir)
    init_logging(log_dir=rc.log_dir)
    logger = get_logger(__name__)
    set_global_seed(rc.seed)
    device = pick_device(rc.device)
    logger.info(f"PPO Trainer starting with device={device}, seed={rc.seed}")

    # Tokenizer
    tokenizer = get_tokenizer(rc.model_name, use_fast=True, trust_remote_code=rc.trust_remote_code)
    if tokenizer.pad_token is None and tokenizer.eos_token:
        tokenizer.pad_token = tokenizer.eos_token

    # Load models
    logger.info("Loading policy (SFT) model...")
    policy = load_policy_model(
        rc.policy_ckpt if os.path.isdir(rc.policy_ckpt) else rc.model_name,
        trust_remote_code=rc.trust_remote_code,
    )
    policy.to(device)
    if rc.gradient_checkpointing and hasattr(policy, "gradient_checkpointing_enable"):
        policy.gradient_checkpointing_enable()

    logger.info("Loading reward model...")
    # RewardModel must support either from_pretrained or load via ckpt path; adapt to your implementation.
    reward_model = RewardModel.from_pretrained(rc.reward_ckpt) if hasattr(RewardModel, "from_pretrained") else RewardModel(policy)  # fallback
    reward_model.to(device)
    reward_model.eval()

    # PPO config
    ppo_cfg = get_ppo_config_from_dict(cfg_dict)
    # Override sensitive fields if present
    ppo_cfg.learning_rate = rc.learning_rate
    ppo_cfg.batch_size = rc.batch_size
    ppo_cfg.mini_batch_size = rc.mini_batch_size
    ppo_cfg.ppo_epochs = rc.ppo_epochs
    ppo_cfg.target_kl = rc.target_kl
    ppo_cfg.init_kl_coef = rc.init_kl_coef
    ppo_cfg.kl_penalty = rc.kl_penalty

    # PPO Trainer
    logger.info("Initializing PPOTrainer...")
    ppo_trainer = PPOTrainer(
        config=ppo_cfg,
        model=policy,
        tokenizer=tokenizer,
        dataset=None,  # we feed queries manually
        data_collator=None,
    )

    # Load prompts
    logger.info(f"Loading prompts from {rc.prompts_path}...")
    prompts = load_prompts(rc.prompts_path, rc.prompt_field)
    if len(prompts) == 0:
        logger.error("No prompts found. Populate data/prompts.jsonl with a 'prompt' field.")
        sys.exit(1)

    # Generation config
    gen_cfg = build_generation_config(rc, tokenizer)

    # Resume last checkpoint if requested
    resume_ckpt = load_checkpoint_safe(rc.output_dir) if args.resume else None
    if resume_ckpt:
        logger.info(f"Resuming from checkpoint: {resume_ckpt}")

    # Training loop
    logger.info("Starting PPO optimization...")
    global_step = 0
    start_time = time.time()

    # Iterate over prompts in batches
    for epoch in range(1):  # In TRL PPO, epochs are internal; we iterate over data once and PPO handles per-step.
        for batch_prompts in chunked(prompts, rc.batch_size):
            # Prepare inputs
            queries = batch_prompts

            # Tokenize queries
            query_toks = tokenizer(
                queries,
                truncation=True,
                max_length=rc.max_prompt_length,
                padding=True,
                return_tensors="pt",
            ).to(device)

            # Generate responses
            with torch.no_grad():
                outputs = policy.generate(
                    **query_toks,
                    generation_config=gen_cfg,
                )
            # Decode responses
            response_ids = outputs.sequences
            responses = tokenizer.batch_decode(response_ids, skip_special_tokens=True)

            # Score responses with reward model
            rewards = compute_rewards_for_responses(
                reward_model=reward_model,
                tokenizer=tokenizer,
                responses=responses,
                device=device,
                max_length=rc.max_response_length,
            )

            # PPO step requires lists
            stats = ppo_trainer.step(queries, responses, rewards)

            # Optional gradient clipping if using HF optim; PPOTrainer typically manages optimizer/grad.
            if rc.grad_clip_norm and hasattr(policy, "parameters"):
                clip_grad_norm_(policy.parameters(), rc.grad_clip_norm)

            global_step += 1

            # Logging
            if stats:
                # stats contains 'policy/...' 'ppo/...' 'kl' etc. when using TRL
                logger.info(
                    f"step={global_step} "
                    f"reward_mean={float(torch.tensor(rewards).mean().item()):.4f} "
                    f"kl={stats.get('kl', None)} "
                    f"loss/policy={stats.get('loss/policy', None)} "
                    f"ppo/returns/mean={stats.get('ppo/returns/mean', None)}"
                )

            # Save checkpoints
            if rc.save_steps and global_step % rc.save_steps == 0:
                logger.info(f"Saving checkpoint at step {global_step}...")
                save_checkpoint_safe(ppo_trainer.model, rc.output_dir, f"step_{global_step}")

            # Evaluation hooks (simple perplexity on prompts; for more, use eval_automatic.py)
            if rc.eval_every_steps and global_step % rc.eval_every_steps == 0:
                try:
                    ppl = compute_perplexity(policy, tokenizer, None, sample_size=rc.eval_sample_size)
                    logger.info(f"Eval perplexity (approx): {ppl:.2f}")
                except Exception as e:
                    logger.warning(f"Eval skipped at step {global_step}: {e}")

    duration = time.time() - start_time
    logger.info(f"PPO training completed in {duration:.2f}s, total_steps={global_step}")

    # Final save
    logger.info("Saving final policy...")
    save_checkpoint_safe(ppo_trainer.model, rc.output_dir, "final")

    # Manifest
    manifest = {
        "config_path": args.config,
        "policy_ckpt": rc.policy_ckpt,
        "reward_ckpt": rc.reward_ckpt,
        "output_dir": rc.output_dir,
        "log_dir": rc.log_dir,
        "seed": rc.seed,
        "duration_sec": duration,
        "total_steps": global_step,
        "batch_size": rc.batch_size,
        "lr": rc.learning_rate,
        "target_kl": rc.target_kl,
        "init_kl_coef": rc.init_kl_coef,
        "kl_penalty": rc.kl_penalty,
        "max_prompt_length": rc.max_prompt_length,
        "max_response_length": rc.max_response_length,
    }
    save_manifest(rc.output_dir, manifest)

    logger.info("All done. Checkpoints and logs are available.")


if __name__ == "__main__":
    main()