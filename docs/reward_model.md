# Reward Model

The reward model learns a scalar preference score from chosen/rejected response pairs.

## Training Objective

- Input: `(prompt, chosen, rejected)`.
- Objective: encourage `score(chosen) > score(rejected)`.
- Common loss: Bradley-Terry or margin-based pairwise ranking loss.

## Implementation Notes

- Initial implementation: `src/models/reward.py`.
- Base model: transformer hidden-state provider.
- Reward head: scalar projection over pooled hidden states.
- Training loop: `src/training/rm_trainer.py`.

## Evaluation Needs

- Held-out preference-pair win rate.
- Calibration by score bucket.
- Safety slice analysis.
- Reward hacking examples.
- Model-card update after each promoted checkpoint.
