#!/bin/bash
set -euo pipefail
source .venv/Scripts/activate
python src/data/generate_candidates.py \
  --prompts data/prompts.jsonl \
  --policy-ckpt ./checkpoints/sft_gpt2 \
  --model-name gpt2 \
  --output data/candidates.jsonl \
  --num-samples 4 \
  --max-prompt-length 256 \
  --max-new-tokens 128 \
  --top-p 0.9 --temperature 1.0