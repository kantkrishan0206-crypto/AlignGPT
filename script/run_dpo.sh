#!/bin/bash
set -euo pipefail

# Activate environment
# conda activate rlhf-lab
source .venv/Scripts/activate

# Run DPO trainer
python src/training/dpo_trainer.py --config configs/dpo_gpt2.yaml --run_name "dpo_gpt2_baseline"