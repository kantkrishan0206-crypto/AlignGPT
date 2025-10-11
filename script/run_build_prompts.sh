#!/bin/bash
set -euo pipefail
source .venv/Scripts/activate
python src/data/build_prompts.py \
  --input data/raw_corpus.jsonl \
  --output data/prompts.jsonl \
  --prompt-field prompt \
  --min-len 5 --max-len 1024 \
  --shuffle --seed 42