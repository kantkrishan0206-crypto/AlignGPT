#!/bin/bash
set -euo pipefail
source .venv/Scripts/activate
python src/data/make_pref_pairs.py \
  --candidates data/candidates.jsonl \
  --output data/pref_pairs.jsonl \
  --strategy reward \
  --reward-ckpt ./checkpoints/rm_gpt2 \
  --model-name gpt2