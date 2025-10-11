# PPO and DPO Training

## PPO (Proximal Policy Optimization)
- Uses the Reward Model to guide updates.
- Balances exploration vs exploitation.
- Loss combines:
  - Policy gradient (maximize reward)
  - KL penalty (stay close to SFT policy)

## DPO (Direct Preference Optimization)
- Bypasses explicit reward modeling.
- Directly optimizes policy using preference pairs.
- Simpler pipeline, fewer moving parts.

## Implementation Notes
- Both methods supported in `src/train_rlhf.py`
- Configurable via YAML/CLI flags:
  - `--algo ppo` or `--algo dpo`
  - `--kl-coef`, `--learning-rate`, etc.