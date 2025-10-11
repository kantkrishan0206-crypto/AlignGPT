#!/bin/bash
set -euo pipefail

# Activate environment (choose one)
# conda activate rlhf-lab
source .venv/Scripts/activate

# Launch SFT orchestrator
python src/training/sft/train_rlhf.py \
  --config configs/sft_gpt2.yaml \
  --run_name "sft_gpt2_baseline"

# Example overrides (uncomment to use):
#   --override training.num_train_epochs=1 training.fp16=false data.max_length=256