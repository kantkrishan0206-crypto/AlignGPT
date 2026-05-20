# PPO and DPO Training

## PPO

Proximal Policy Optimization uses the reward model as a learned signal while constraining drift from the supervised policy.

Core controls:

- Reward score.
- KL penalty.
- Batch size and mini-batch size.
- Response length and sampling policy.
- Checkpoint cadence and evaluation hooks.

## DPO

Direct Preference Optimization uses chosen/rejected pairs directly and avoids an explicit reward model in the optimization loop.

Core controls:

- Preference temperature beta.
- Reference model choice.
- Prompt and completion formatting.
- Pair quality and deduplication.
- Held-out preference evaluation.

## Implementation Notes

Legacy implementations remain in `src/training/ppo_trainer.py` and `src/training/dpo_trainer.py`. Future work should expose these through stable package and CLI entry points.
