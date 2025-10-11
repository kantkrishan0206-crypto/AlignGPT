#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Policy Model Utilities
- Loading, saving, and utilities for causal LMs (policy)
- Generation helpers, safe config injection, device handling
"""

import os
import json
import torch
from typing import Optional, Dict, Any, List, Tuple

from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    GenerationConfig,
)

DEFAULT_MODEL_NAME = "gpt2"

class PolicyWrapper:
    """
    A thin wrapper around AutoModelForCausalLM that:
    - Tracks device
    - Offers generation helpers
    - Handles gradient checkpointing and precision
    - Encapsulates save/load utilities
    """

    def __init__(self, model: AutoModelForCausalLM, device: Optional[torch.device] = None):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def enable_gradient_checkpointing(self):
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

    def generate_texts(
        self,
        tokenizer,
        prompts: List[str],
        gen_cfg: GenerationConfig,
        max_prompt_length: int,
    ) -> Tuple[List[str], torch.Tensor]:
        """
        Batched generation. Returns decoded responses and sequences tensor.
        """
        inputs = tokenizer(
            prompts,
            truncation=True,
            max_length=max_prompt_length,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=gen_cfg,
            )
        sequences = outputs.sequences
        texts = tokenizer.batch_decode(sequences, skip_special_tokens=True)
        return texts, sequences

    def save_pretrained(self, output_dir: str, manifest_extra: Optional[Dict[str, Any]] = None):
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        manifest = {"type": "policy", "device": str(self.device)}
        if manifest_extra:
            manifest.update(manifest_extra)
        with open(os.path.join(output_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    @staticmethod
    def from_pretrained(path_or_name: str, trust_remote_code: bool = False) -> "PolicyWrapper":
        model = AutoModelForCausalLM.from_pretrained(path_or_name, trust_remote_code=trust_remote_code)
        return PolicyWrapper(model)


def load_policy_model(
    path_or_name: str = DEFAULT_MODEL_NAME,
    trust_remote_code: bool = False,
    torch_dtype: Optional[str] = None,
    device_map: Optional[str] = None,
    use_cache: Optional[bool] = None,
) -> AutoModelForCausalLM:
    """
    Load a causal LM policy.
    - Accepts HF hub ID or local checkpoint dir
    - Optional dtype and device map for advanced setups
    """
    kwargs: Dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if torch_dtype:
        # map string to torch dtype safely
        dtype_map = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }
        kwargs["torch_dtype"] = dtype_map.get(torch_dtype, None)
    if device_map:
        kwargs["device_map"] = device_map

    # Config fallback to ensure use_cache compatibility
    try:
        config = AutoConfig.from_pretrained(path_or_name, trust_remote_code=trust_remote_code)
        if use_cache is not None:
            config.use_cache = use_cache
        model = AutoModelForCausalLM.from_pretrained(path_or_name, config=config, **kwargs)
    except Exception:
        # Fallback without explicit config
        model = AutoModelForCausalLM.from_pretrained(path_or_name, **kwargs)

    return model


def build_generation_config(
    tokenizer,
    do_sample: bool = True,
    top_p: float = 0.9,
    top_k: int = 0,
    temperature: float = 1.0,
    repetition_penalty: float = 1.0,
    max_new_tokens: int = 256,
    eos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
) -> GenerationConfig:
    """
    Construct a GenerationConfig compatible with Trainer loops.
    """
    return GenerationConfig(
        do_sample=do_sample,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id or tokenizer.eos_token_id,
        pad_token_id=pad_token_id or tokenizer.pad_token_id,
        return_dict_in_generate=True,
        output_scores=True,
    )