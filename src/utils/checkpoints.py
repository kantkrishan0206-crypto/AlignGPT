#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Checkpoint utilities
- Atomic save (temp file + rename)
- Retention policy (save_total_limit)
- Safe load discovery
- Model manifest recording
"""

import os
import json
import shutil
import tempfile
from typing import Optional, Dict, Any, Union, List

import torch
from transformers import PreTrainedModel

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _atomic_write(src_path: str, dst_path: str):
    """
    Atomically move a temp file to destination to avoid partial writes.
    """
    os.replace(src_path, dst_path)

def _list_checkpoints(output_dir: str) -> List[str]:
    files = []
    for name in os.listdir(output_dir):
        if name.startswith("step_") or name.startswith("epoch_") or name == "final":
            files.append(name)
    # Sort: final last, then epoch/step in natural order
    def key(n):
        if n == "final":
            return (2, 0)
        if n.startswith("epoch_"):
            try:
                return (0, int(n.split("_", 1)[1]))
            except:
                return (0, 0)
        if n.startswith("step_"):
            try:
                return (1, int(n.split("_", 1)[1]))
            except:
                return (1, 0)
        return (3, 0)
    return sorted(files, key=key)

def _apply_retention(output_dir: str, save_total_limit: Optional[int]):
    if not save_total_limit or save_total_limit <= 0:
        return
    # Keep most recent checkpoints by order
    all_ckpts = _list_checkpoints(output_dir)
    if len(all_ckpts) <= save_total_limit:
        return
    to_remove = all_ckpts[:-save_total_limit]
    for name in to_remove:
        path = os.path.join(output_dir, name)
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
        except Exception:
            pass

def save_checkpoint_safe(
    model: Union[PreTrainedModel, torch.nn.Module],
    output_dir: str,
    tag: str,
    save_total_limit: Optional[int] = None,
    extra_manifest: Optional[Dict[str, Any]] = None,
):
    """
    Save a model checkpoint atomically under output_dir/tag.
    - For HF models, use save_pretrained; else save state_dict.
    - Writes a small JSON manifest alongside.
    - Applies retention limit.
    """
    _ensure_dir(output_dir)
    ckpt_dir = os.path.join(output_dir, str(tag))
    _ensure_dir(ckpt_dir)

    # Save model
    if hasattr(model, "save_pretrained"):
        # Use temp dir then rename
        with tempfile.TemporaryDirectory() as td:
            model.save_pretrained(td)
            for fname in os.listdir(td):
                src = os.path.join(td, fname)
                dst = os.path.join(ckpt_dir, fname)
                if os.path.isdir(src):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dst)
    else:
        # Torch state_dict
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".pt")
        try:
            os.close(tmp_fd)
            torch.save(model.state_dict(), tmp_path)
            dst = os.path.join(ckpt_dir, "pytorch_model.bin")
            _atomic_write(tmp_path, dst)
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    # Manifest
    manifest = {"tag": tag}
    if extra_manifest:
        manifest.update(extra_manifest)
    with open(os.path.join(ckpt_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Retention
    _apply_retention(output_dir, save_total_limit)

def load_checkpoint_safe(output_dir: str) -> Optional[str]:
    """
    Return the latest checkpoint tag by directory listing.
    """
    if not os.path.isdir(output_dir):
        return None
    ckpts = _list_checkpoints(output_dir)
    if not ckpts:
        return None
    return ckpts[-1]

def copy_checkpoint(src_dir: str, dst_dir: str):
    """
    Copy a checkpoint directory safely.
    """
    if not os.path.isdir(src_dir):
        raise FileNotFoundError(f"Source checkpoint not found: {src_dir}")
    os.makedirs(dst_dir, exist_ok=True)
    for name in os.listdir(src_dir):
        s = os.path.join(src_dir, name)
        d = os.path.join(dst_dir, name)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)