# Project Overview

This repository implements a modular RLHF (Reinforcement Learning with Human Feedback) pipeline.  
It covers the full stack:

- **Data preparation**: converting open datasets into `sft.jsonl`, `prompts.jsonl`, and `pref_pairs.jsonl`.
- **Supervised Fine-Tuning (SFT)**: training a base model on instruction–response pairs.
- **Reward Modeling (RM)**: learning a scoring function from preference pairs.
- **Policy Optimization (PPO/DPO)**: aligning the model with human preferences.
- **Evaluation**: measuring alignment, helpfulness, and safety.

The design emphasizes:
- Reproducibility (clear configs, checkpoints, logs).
- Modularity (separate scripts for SFT, RM, PPO/DPO, evaluation).
- Extensibility (easy to swap models, datasets, or algorithms).