"""
src/models/tokenizer.py

Production-ready tokenizer utilities and wrapper for RLHF pipelines.

Goals:
- Provide a single, well-documented wrapper around HuggingFace tokenizers (both fast & slow)
  and the `tokenizers` library for training custom tokenizers.
- Utilities for batching, padding, truncation, alignment between character offsets and token ids,
  on-the-fly normalization, and cached tokenization for large corpora.
- Optional helpers to train a SentencePiece / BPE tokenizer from raw text files and to push
  tokenizers to the HuggingFace hub.
- Robustness features: fallback tokenizers, safe special-token handling, deterministic behavior
  for reproducibility, and helpful CLI for common tasks.

Notes:
- This code avoids heavy runtime dependencies by guarding optional imports (sentencepiece, tokenizers).
- Designed for integration with the PolicyModel and RewardModel wrappers.

"""

from __future__ import annotations

import os
import io
import json
import logging
import hashlib
import pickle
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

try:
    from tokenizers import Tokenizer as HFTokenizer, trainers, models, pre_tokenizers, normalizers
    TOKENIZERS_AVAILABLE = True
except Exception:
    HFTokenizer = None  # type: ignore
    TOKENIZERS_AVAILABLE = False

from transformers import AutoTokenizer, PreTrainedTokenizerFast, PreTrainedTokenizer

import numpy as np

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
class TokenizerConfig:
    model_name_or_path: Optional[str] = None
    use_fast: bool = True
    unk_token: str = "<|unk|>"
    pad_token: str = "<|pad|>"
    bos_token: Optional[str] = "<|bos|>"
    eos_token: Optional[str] = "<|eos|>"
    additional_special_tokens: List[str] = field(default_factory=list)
    max_length: int = 512
    do_lower_case: bool = False
    cache_dir: Optional[str] = None


# ---------------------------
# Exceptions
# ---------------------------

class TokenizerError(Exception):
    pass


# ---------------------------
# Helpers
# ---------------------------

def _hash_file_list(filepaths: Iterable[str]) -> str:
    h = hashlib.sha256()
    for p in sorted(filepaths):
        h.update(p.encode("utf-8"))
        try:
            stat = os.stat(p)
            h.update(str(stat.st_mtime).encode("utf-8"))
            h.update(str(stat.st_size).encode("utf-8"))
        except OSError:
            continue
    return h.hexdigest()


# ---------------------------
# Tokenizer wrapper
# ---------------------------

class TokenizerWrapper:
    """A thin, user-friendly wrapper around HuggingFace tokenizers.

    Supports loading pretrained tokenizers (fast preferred), training a new tokenizer
    via the `tokenizers` library, and common utilities used in RLHF pipelines.

    Example:
        cfg = TokenizerConfig(model_name_or_path="gpt2", max_length=1024)
        tok = TokenizerWrapper.from_pretrained(cfg)
        ids = tok.encode("Hello world")

    """

    def __init__(self, tokenizer: Union[PreTrainedTokenizerFast, PreTrainedTokenizer, Any], cfg: TokenizerConfig):
        self._tokenizer = tokenizer
        self.cfg = cfg
        # unify interface: fast tokenizers have .encode_plus / __call__
        self.is_fast = hasattr(tokenizer, "is_fast") and getattr(tokenizer, "is_fast")
        self.max_length = cfg.max_length

    # ------------------
    # Factory constructors
    # ------------------

    @classmethod
    def from_pretrained(cls, cfg: Union[TokenizerConfig, str], local_files_only: bool = False) -> "TokenizerWrapper":
        if isinstance(cfg, str):
            cfg = TokenizerConfig(model_name_or_path=cfg)
        if cfg.model_name_or_path is None:
            raise TokenizerError("model_name_or_path must be provided for from_pretrained")

        try:
            tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, use_fast=cfg.use_fast, local_files_only=local_files_only)
            # ensure special tokens
            special_tokens = {}
            if getattr(tokenizer, "pad_token", None) is None and cfg.pad_token:
                special_tokens["pad_token"] = cfg.pad_token
            if getattr(tokenizer, "bos_token", None) is None and cfg.bos_token:
                special_tokens["bos_token"] = cfg.bos_token
            if getattr(tokenizer, "eos_token", None) is None and cfg.eos_token:
                special_tokens["eos_token"] = cfg.eos_token
            if cfg.additional_special_tokens:
                special_tokens["additional_special_tokens"] = cfg.additional_special_tokens
            if special_tokens:
                tokenizer.add_special_tokens(special_tokens)
            logger.info("Loaded tokenizer %s (fast=%s)", cfg.model_name_or_path, getattr(tokenizer, "is_fast", False))
            return cls(tokenizer, cfg)
        except Exception as e:
            logger.warning("AutoTokenizer failed to load %s: %s", cfg.model_name_or_path, e)
            raise

    @classmethod
    def train_new_tokenizer(
        cls,
        files: List[str],
        out_path: str,
        vocab_size: int = 32000,
        tokenizer_type: str = "bpe",
        special_tokens: Optional[List[str]] = None,
        lowercase: bool = False,
        cache_dir: Optional[str] = None,
    ) -> "TokenizerWrapper":
        """Train a new tokenizer using the `tokenizers` library.

        Args:
            files: list of text file paths with training corpus
            out_path: directory where tokenizer files will be saved
            tokenizer_type: 'bpe' or 'wordpiece' or 'unigram' (if tokenizers supports)
        Returns:
            TokenizerWrapper wrapping a PreTrainedTokenizerFast
        """
        if not TOKENIZERS_AVAILABLE:
            raise TokenizerError("`tokenizers` library not installed; cannot train new tokenizer")

        # configure model
        tokenizer_obj = None
        tokenizer_type = tokenizer_type.lower()
        if tokenizer_type == "bpe":
            model = models.BPE(unk_token="<|unk|>")
        elif tokenizer_type == "wordpiece":
            model = models.WordPiece(unk_token="<|unk|>")
        else:
            # fallback to BPE
            model = models.BPE(unk_token="<|unk|>")

        tokenizer_obj = HFTokenizer(model)
        # normalizer & pretokenizer
        tokenizer_obj.normalizer = normalizers.Sequence([normalizers.NFD(), normalizers.Lowercase()]) if lowercase else normalizers.NFD()
        tokenizer_obj.pre_tokenizer = pre_tokenizers.Whitespace()

        trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens or ["<|pad|>", "<|unk|>", "<|bos|>", "<|eos|>"], initial_alphabet=[]) if tokenizer_type == "bpe" else trainers.WordPieceTrainer(vocab_size=vocab_size, special_tokens=special_tokens)

        logger.info("Training tokenizer type=%s on %d files, vocab_size=%d", tokenizer_type, len(files), vocab_size)
        tokenizer_obj.train(files, trainer)

        # save
        os.makedirs(out_path, exist_ok=True)
        tokenizer_obj.save(os.path.join(out_path, "tokenizer.json"))

        # wrap as PreTrainedTokenizerFast
        ptf = PreTrainedTokenizerFast(tokenizer_file=os.path.join(out_path, "tokenizer.json"))
        # ensure special tokens
        if special_tokens:
            ptf.add_special_tokens({"additional_special_tokens": special_tokens})
        cfg = TokenizerConfig(model_name_or_path=out_path, use_fast=True)
        wrapper = cls(ptf, cfg)
        if cache_dir:
            wrapper.cfg.cache_dir = cache_dir
        logger.info("Saved new tokenizer to %s", out_path)
        return wrapper

    # ------------------
    # Core methods
    # ------------------

    def save_pretrained(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        try:
            if isinstance(self._tokenizer, PreTrainedTokenizerFast) or isinstance(self._tokenizer, PreTrainedTokenizer):
                self._tokenizer.save_pretrained(path)
            else:
                # tokenizers library object
                if TOKENIZERS_AVAILABLE and isinstance(self._tokenizer, HFTokenizer):
                    self._tokenizer.save(os.path.join(path, "tokenizer.json"))
                else:
                    raise TokenizerError("Unsupported tokenizer type for save_pretrained")
            logger.info("Tokenizer saved to %s", path)
        except Exception as e:
            logger.exception("Failed to save tokenizer: %s", e)
            raise

    def encode(self, text: Union[str, List[str]], add_special_tokens: bool = True, truncation: bool = True, max_length: Optional[int] = None) -> Union[List[int], List[List[int]]]:
        max_length = max_length or self.max_length
        if isinstance(text, str):
            out = self._tokenizer(text, add_special_tokens=add_special_tokens, truncation=truncation, max_length=max_length)
            return out["input_ids"]
        else:
            out = self._tokenizer(text, add_special_tokens=add_special_tokens, truncation=truncation, max_length=max_length, padding=False)
            return out["input_ids"]

    def __call__(self, texts: Union[str, List[str]], **kwargs) -> Dict[str, Any]:
        # behave like HF tokenizer
        kwargs.setdefault("padding", False)
        kwargs.setdefault("truncation", True)
        kwargs.setdefault("max_length", self.max_length)
        return self._tokenizer(texts, return_tensors=None, **kwargs)

    def batch_encode(self, texts: List[str], batch_size: int = 32, show_progress: bool = False, **kwargs) -> List[List[int]]:
        all_ids: List[List[int]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self._tokenizer(batch, truncation=kwargs.pop("truncation", True), padding=kwargs.pop("padding", False), max_length=kwargs.pop("max_length", self.max_length))
            all_ids.extend(enc["input_ids"])
        return all_ids

    def decode(self, token_ids: Union[List[int], List[List[int]]], skip_special_tokens: bool = True) -> Union[str, List[str]]:
        if isinstance(token_ids[0], list) or isinstance(token_ids[0], tuple):
            return [self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens) for ids in token_ids]
        return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def pad(self, sequences: List[List[int]], max_length: Optional[int] = None, padding_value: Optional[int] = None) -> Tuple[List[List[int]], int]:
        max_length = max_length or self.max_length
        pad_id = padding_value if padding_value is not None else self._tokenizer.pad_token_id if getattr(self._tokenizer, "pad_token_id", None) is not None else 0
        padded = [seq[:max_length] + [pad_id] * max(0, max_length - len(seq)) for seq in sequences]
        return padded, pad_id

    # ------------------
    # Alignment utilities
    # ------------------

    def char_to_token_offset(self, text: str, char_idx: int) -> Optional[int]:
        """Map a character index to a token index (best-effort). Returns None if out of range."""
        enc = self._tokenizer(text, return_offsets_mapping=True)
        offsets = enc.get("offset_mapping")
        if not offsets:
            return None
        for tidx, (s, e) in enumerate(offsets):
            if s <= char_idx < e:
                return tidx
        return None

    def token_to_char_span(self, text: str, token_idx: int) -> Optional[Tuple[int, int]]:
        enc = self._tokenizer(text, return_offsets_mapping=True)
        offsets = enc.get("offset_mapping")
        if not offsets or token_idx >= len(offsets):
            return None
        return offsets[token_idx]

    def token_spans(self, text: str) -> List[Tuple[int, int]]:
        enc = self._tokenizer(text, return_offsets_mapping=True)
        return enc.get("offset_mapping", [])

    # ------------------
    # Caching (simple disk cache)
    # ------------------

    def cache_encode_to_file(self, texts: List[str], out_file: str, batch_size: int = 64) -> None:
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        # compute a fingerprint
        fingerprint = hashlib.sha256("\n".join(texts).encode("utf-8"))
        if os.path.exists(out_file):
            logger.info("Out file %s already exists. Skipping write (overwrite if needed).", out_file)
            return
        logger.info("Encoding %d texts to %s", len(texts), out_file)
        with open(out_file, "wb") as fh:
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                enc = self._tokenizer(batch, truncation=True, padding=False, max_length=self.max_length)
                ids = enc["input_ids"]
                for seq in ids:
                    pickle.dump(seq, fh)
        logger.info("Saved encoded sequences to %s", out_file)

    # ------------------
    # Streaming tokenization
    # ------------------

    def stream_tokenize(self, fileobj: io.TextIOBase, chunk_size: int = 1024) -> Iterable[List[int]]:
        """Iteratively read text from a file-like object and yield tokenized sequences per line/paragraph.

        This is useful when processing large corpora that don't fit in memory.
        """
        for line in fileobj:
            line = line.strip()
            if not line:
                continue
            yield self._tokenizer(line, truncation=True, max_length=self.max_length)["input_ids"]

    # ------------------
    # Hub helpers
    # ------------------

    def push_to_hub(self, repo_id: str, token: Optional[str] = None, private: bool = False) -> None:
        try:
            if hasattr(self._tokenizer, "push_to_hub"):
                self._tokenizer.push_to_hub(repo_id, use_auth_token=token, private=private)
                logger.info("Pushed tokenizer to hub: %s", repo_id)
            else:
                raise TokenizerError("Underlying tokenizer doesn't support push_to_hub")
        except Exception as e:
            logger.exception("Failed to push tokenizer to hub: %s", e)
            raise


# ---------------------------
# Minimal CLI utilities
# ---------------------------

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Tokenizer utilities: train or inspect tokenizers")
    sub = p.add_subparsers(dest="cmd")

    train = sub.add_parser("train", help="Train a new tokenizer from text files")
    train.add_argument("files", nargs="+", help="Text files for training")
    train.add_argument("--out", required=True, help="Output folder")
    train.add_argument("--vocab-size", type=int, default=32000)
    train.add_argument("--type", choices=["bpe", "wordpiece"], default="bpe")
    train.add_argument("--lowercase", action="store_true")

    inspect = sub.add_parser("inspect", help="Inspect a tokenizer directory or HF id")
    inspect.add_argument("tokenizer", help="path or HF id")

    args = p.parse_args()
    if args.cmd == "train":
        TokenizerWrapper.train_new_tokenizer(args.files, args.out, vocab_size=args.vocab_size, tokenizer_type=args.type, lowercase=args.lowercase)
    elif args.cmd == "inspect":
        cfg = TokenizerConfig(model_name_or_path=args.tokenizer)
        tw = TokenizerWrapper.from_pretrained(cfg)
        print("Is fast:", getattr(tw._tokenizer, "is_fast", False))
        print("Vocab size:", getattr(tw._tokenizer, "vocab_size", None))
        print("Special tokens:", {"pad": tw._tokenizer.pad_token, "bos": getattr(tw._tokenizer, "bos_token", None), "eos": getattr(tw._tokenizer, "eos_token", None)})
    else:
        p.print_help()
