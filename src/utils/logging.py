#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Structured logging utilities
- Console + rotating file handlers
- Contextual run metadata
- Optional integrations: WandB/MLflow (no hard dependency)
"""

import os
import sys
import json
import time
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional, Dict, Any

_LOGGER_CACHE: Dict[str, logging.Logger] = {}
_DEFAULT_LOG_DIR = "./logs"
_DEFAULT_LEVEL = logging.INFO

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record.created)),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)

class ConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        ts = time.strftime("%H:%M:%S", time.localtime(record.created))
        lvl = record.levelname
        name = record.name
        msg = record.getMessage()
        return f"[{ts}] {lvl:<7} {name}: {msg}"

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def init_logging(
    log_dir: str = _DEFAULT_LOG_DIR,
    level: int = _DEFAULT_LEVEL,
    json_file: bool = True,
    file_max_bytes: int = 10 * 1024 * 1024,
    file_backup_count: int = 3,
    run_name: Optional[str] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Initialize root logging: console + rotating file.
    Safe to call multiple times; it will reconfigure handlers.
    """
    _ensure_dir(log_dir)
    root = logging.getLogger()
    root.setLevel(level)

    # Clear existing handlers to avoid duplicate logs
    for h in list(root.handlers):
        root.removeHandler(h)

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(ConsoleFormatter())
    root.addHandler(ch)

    # File (rotating)
    logfile = os.path.join(log_dir, "run.log")
    fh = RotatingFileHandler(logfile, maxBytes=file_max_bytes, backupCount=file_backup_count, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(JsonFormatter() if json_file else ConsoleFormatter())
    root.addHandler(fh)

    # Add initial run manifest
    manifest = {"run_name": run_name, "ts": int(time.time())}
    if extra_meta:
        manifest.update(extra_meta)
    with open(os.path.join(log_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

def get_logger(name: str) -> logging.Logger:
    """
    Get a cached logger with no extra handlers (uses root handlers).
    """
    if name in _LOGGER_CACHE:
        return _LOGGER_CACHE[name]
    logger = logging.getLogger(name)
    _LOGGER_CACHE[name] = logger
    return logger

def log_dict(logger: logging.Logger, label: str, data: Dict[str, Any], level: int = logging.INFO):
    """
    Log a dict compactly; avoids massive outputs by truncating long values.
    """
    compact = {}
    for k, v in data.items():
        s = str(v)
        compact[k] = (s[:300] + "…") if len(s) > 300 else s
    logger.log(level, f"{label}: {json.dumps(compact, ensure_ascii=False)}")

# Optional integrations with WandB/MLflow (lazy import)
def init_wandb(project: str, run_name: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
    try:
        import wandb
        wandb.init(project=project, name=run_name, config=config or {})
        return wandb
    except Exception as e:
        logging.getLogger(__name__).warning(f"WandB init failed: {e}")
        return None

def log_wandb(wandb_obj, metrics: Dict[str, Any], step: Optional[int] = None):
    if wandb_obj is None:
        return
    try:
        wandb_obj.log(metrics, step=step)
    except Exception as e:
        logging.getLogger(__name__).warning(f"WandB log failed: {e}")

def finish_wandb(wandb_obj):
    if wandb_obj is None:
        return
    try:
        wandb_obj.finish()
    except Exception as e:
        logging.getLogger(__name__).warning(f"WandB finish failed: {e}")