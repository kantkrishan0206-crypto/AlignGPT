#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tokenizer Utilities
- Robust loader for HF tokenizers
- Ensures pad/eos/bos tokens exist
- Normalization utilities for prompts/responses
"""

from typing import Optional, Dict, Any
from transformers import AutoTokenizer


def get_tokenizer(
    model_name_or_path: str,
    use_fast: bool = True,
    trust_remote_code: bool = False,
    add_padding_token_if_missing: bool = True,
    add_bos_if_missing: bool = False,
    add_eos_if_missing: bool = False,
) -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(
        model_name_or_path, use_fast=use_fast, trust_remote_code=trust_remote_code
    )
    # Ensure pad/eos/bos tokens are present
    special_added = False

    if add_padding_token_if_missing and tok.pad_token is None:
        # Prefer eos as pad if available
        if tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        elif tok.bos_token is not None:
            tok.pad_token = tok.bos_token
        else:
            tok.add_special_tokens({"pad_token": "[PAD]"})
        special_added = True

    if add_bos_if_missing and tok.bos_token is None:
        tok.add_special_tokens({"bos_token": "<BOS>"})
        special_added = True

    if add_eos_if_missing and tok.eos_token is None:
        tok.add_special_tokens({"eos_token": "<EOS>"})
        special_added = True

    if special_added and hasattr(tok, "model_max_length"):
        # If special tokens added, caller may need to resize embeddings on the model
        pass

    return tok


def normalize_prompt_text(text: str) -> str:
    # Trim excessive whitespace and normalize line breaks
    return " ".join(text.strip().split())


def normalize_response_text(text: str) -> str:
    # Basic normalization to prevent odd tokens
    return text.replace("\r\n", "\n").strip()