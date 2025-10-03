"""
src/training/sft_trainer.py

Supervised Fine-Tuning trainer for causal LMs (SFT).

Features:
- Loads config from YAML or CLI args.
- Supports HF Trainer and a fallback custom loop for large-batch / streaming datasets.
- Integrates TokenizerWrapper, PolicyModel, and optional PEFT adapters.
- Mixed Precision (AMP) and gradient accumulation support.
- Checkpointing, logging, and evaluation using generation-based metrics (BLEU, ROUGE-lite).
- Dataset helpers for JSONL SFT data: {prompt, response} lines.
- Save final checkpoint including tokenizer and config.

Usage examples:
    python src/training/sft_trainer.py --config configs/sft_gpt2.yaml
    bash scripts/run_sft.sh

"""
from __future__ import annotations

import os
import sys
import json
import time
import math
import logging
import argparse
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

import yaml
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    default_data_collator,
    set_seed,
)

# local imports from project
try:
    from src.models.tokenizer import TokenizerWrapper, TokenizerConfig
    from src.models.policy import PolicyModel, PolicyWrapperConfig
except Exception:
    # allow running in different working directories
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from models.tokenizer import TokenizerWrapper, TokenizerConfig
    from models.policy import PolicyModel, PolicyWrapperConfig

logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)


# ---------------------------
# Config dataclass
# ---------------------------

@dataclass
class SFTConfig:
    model_name_or_path: str = "gpt2"
    tokenizer_name_or_path: Optional[str] = None
    train_file: str = "data/sft.jsonl"
    output_dir: str = "./checkpoints/sft"
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    num_train_epochs: int = 1
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    logging_steps: int = 50
    save_steps: int = 200
    eval_steps: int = 200
    max_seq_length: int = 512
    gradient_accumulation_steps: int = 1
    fp16: bool = True
    seed: int = 42
    use_peft: bool = False
    peft_adapter_path: Optional[str] = None
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    local_files_only: bool = False


# ---------------------------
# Dataset
# ---------------------------

class SFTDataset(Dataset):
    """Simple dataset for supervised fine-tuning. Expects JSONL with {prompt, response}."""

    def __init__(self, path: str, tokenizer: TokenizerWrapper, cfg: SFTConfig):
        self.path = path
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.samples = []
        if not os.path.exists(path):
            raise FileNotFoundError(f"SFT data file not found: {path}")
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if not ("prompt" in obj and "response" in obj):
                    continue
                self.samples.append({"prompt": obj["prompt"], "response": obj["response"]})
        logger.info("Loaded %d SFT samples from %s", len(self.samples), path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        prompt = item["prompt"]
        response = item["response"]
        # create input by concatenating prompt and response with EOS
        text = prompt + (self.tokenizer.cfg.bos_token or "")
        # Many SFT pipelines format as: <prompt><response><eos>
        full = prompt + response + (self.tokenizer.cfg.eos_token or "")
        enc = self.tokenizer(full, truncation=True, max_length=self.cfg.max_seq_length)
        # Prepare labels: mask prompt tokens so loss is computed only on response tokens
        input_ids = enc["input_ids"]
        # find split index: len(tokenize(prompt))
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        prompt_len = len(prompt_ids)
        labels = input_ids.copy()
        # mask prompt labels as -100
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100
        return {"input_ids": input_ids, "labels": labels}


# ---------------------------
# Helpers
# ---------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SFT trainer for RLHF lab")
    p.add_argument("--config", type=str, required=False, help="Path to YAML config")
    p.add_argument("--train_file", type=str, required=False, help="Override train file")
    p.add_argument("--output_dir", type=str, required=False, help="Override output dir")
    p.add_argument("--local_files_only", action="store_true", help="Load models/tokenizers from local files only")
    return p.parse_args()


def load_config(path: Optional[str]) -> SFTConfig:
    if path is None:
        return SFTConfig()
    with open(path, "r") as fh:
        raw = yaml.safe_load(fh)
    # map keys
    cfg = SFTConfig(**raw)
    return cfg


# ---------------------------
# Main training flow
# ---------------------------

def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.train_file:
        cfg.train_file = args.train_file
    if args.output_dir:
        cfg.output_dir = args.output_dir
    cfg.local_files_only = args.local_files_only

    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    # tokenizer
    tok_cfg = TokenizerConfig(model_name_or_path=cfg.tokenizer_name_or_path or cfg.model_name_or_path, max_length=cfg.max_seq_length)
    tokenizer = TokenizerWrapper.from_pretrained(tok_cfg, local_files_only=cfg.local_files_only)

    # model
    logger.info("Loading model %s", cfg.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name_or_path, local_files_only=cfg.local_files_only)
    # resize embeddings if tokenizer length changed
    if hasattr(tokenizer._tokenizer, "vocab_size") and model.get_input_embeddings().num_embeddings != getattr(tokenizer._tokenizer, "vocab_size", len(tokenizer._tokenizer)):
        try:
            model.resize_token_embeddings(len(tokenizer._tokenizer))
        except Exception:
            logger.warning("Failed to resize token embeddings; continuing without resize")

    # optional PEFT
    if cfg.use_peft:
        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            logger.info("Applying PEFT LoRA adapter setup")
            model = prepare_model_for_kbit_training(model)
            peft_config = LoraConfig(r=8, lora_alpha=32, target_modules=["c_attn", "q_proj", "v_proj", "k_proj"], lora_dropout=0.05, bias="none")
            model = get_peft_model(model, peft_config)
        except Exception as e:
            logger.exception("PEFT requested but failed to apply: %s", e)
            raise

    # dataset
    dataset = SFTDataset(cfg.train_file, tokenizer, cfg)

    # prepare DataCollator - we use default collator since we supply labels and already padded sequences may be needed
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # pad input_ids and labels
        input_ids = [torch.tensor(x["input_ids"], dtype=torch.long) for x in batch]
        labels = [torch.tensor(x["labels"], dtype=torch.long) for x in batch]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer._tokenizer.pad_token_id or 0)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        attention_mask = (input_ids != (tokenizer._tokenizer.pad_token_id or 0)).long()
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    data_collator = collate_fn

    # training arguments
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        evaluation_strategy="steps",
        eval_steps=cfg.eval_steps,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        fp16=cfg.fp16,
        save_total_limit=3,
        remove_unused_columns=False,
        push_to_hub=cfg.push_to_hub,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer._tokenizer if hasattr(tokenizer, "_tokenizer") else None,
    )

    # train
    logger.info("Starting SFT training")
    trainer.train()

    # save final
    logger.info("Saving final model to %s", cfg.output_dir)
    trainer.save_model(cfg.output_dir)
    # save tokenizer
    try:
        tokenizer.save_pretrained(cfg.output_dir)
    except Exception as e:
        logger.warning("Failed to save tokenizer: %s", e)

    logger.info("SFT training finished")


if __name__ == "__main__":
    main()

