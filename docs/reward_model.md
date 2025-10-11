
---

## 📂 `docs/reward_model.md`

```markdown
# Reward Model

The Reward Model (RM) learns to score responses based on human preferences.

## Training Objective
- Input: `(prompt, chosen, rejected)`
- Loss: Margin-based or pairwise ranking loss
  - Encourage `score(chosen) > score(rejected)`

## Implementation
- Base: Transformer encoder/decoder (e.g., GPT‑2, LLaMA).
- Head: Linear layer projecting hidden states to scalar reward.
- Training: AdamW optimizer, learning rate scheduling, gradient clipping.

## Outputs
- Checkpoints saved in `checkpoints/rm/`
- Used for:
  - PPO training (reward signal)
  - Evaluation of candidate responses