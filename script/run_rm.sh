#!/bin/bash
set -euo pipefail

# Activate environment (choose one)
# conda activate rlhf-lab
source .venv/Scripts/activate

# Launch Reward Model trainer
python src/training/rm_trainer.py \
  --config configs/rm_gpt2.yaml