#!/bin/bash
set -euo pipefail

# Activate environment (choose one)
# conda activate rlhf-lab
source .venv/Scripts/activate

# Launch PPO trainer
python src/training/ppo_trainer.py \
  --config configs/ppo_gpt2.yaml \
  --run_name "ppo_gpt2_baseline"