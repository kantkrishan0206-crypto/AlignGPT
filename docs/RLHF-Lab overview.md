# Reward Model Documentation for RLHF-Lab

This document explains the Reward Model (RM) used in the RLHF-Lab pipeline. The Reward Model is a key component for Reinforcement Learning from Human Feedback (RLHF), providing a scalar reward signal for model outputs based on human preferences or simulated data. This document covers architecture, data requirements, training methodology, evaluation, and integration with other modules.

---

## 1. Purpose of the Reward Model

The Reward Model (RM) is designed to:

1. Score candidate outputs from a language model based on their alignment with human preferences.
2. Provide differentiable feedback that can be used to optimize a policy model through PPO or DPO.
3. Quantify the quality of model outputs in a scalar format, enabling automated and scalable evaluation.

Unlike supervised fine-tuning, which directly teaches the model correct responses, the RM learns to predict a reward signal that reflects human judgment. This allows the policy model to explore multiple outputs while optimizing for higher reward scores.

---

## 2. Reward Model Architecture

A typical Reward Model in RLHF-Lab is a transformer-based language model with an additional reward head. Key components include:

### 2.1 Base Language Model

* Pretrained causal or encoder-decoder model (e.g., GPT-2, LLaMA, or Falcon).
* Provides contextual embeddings for the input prompt and candidate response.
* Can be frozen or partially fine-tuned depending on computational constraints.

### 2.2 Reward Head

* A simple feed-forward network that converts hidden states into a scalar reward.
* Usually a linear layer: `nn.Linear(hidden_size, 1)`.
* Often applies mean pooling over token embeddings before scoring.
* Optional: deeper MLP layers, layer normalization, or attention pooling to improve performance.

### Example PyTorch Skeleton

```python
import torch
import torch.nn as nn
from transformers import AutoModel

class RewardModel(nn.Module):
    def __init__(self, base_model_name='gpt2', hidden_size=768):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(base_model_name)
        self.reward_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask=None):
        hidden_states = self.base_model(input_ids, attention_mask=attention_mask).last_hidden_state
        pooled = hidden_states.mean(dim=1)  # simple mean pooling
        reward = self.reward_head(pooled)
        return reward
```

---

## 3. Input and Output Data

### 3.1 Input Data

The Reward Model takes **pre-tokenized prompts and candidate responses**. For RM training, the preferred and rejected responses are paired with the same prompt:

```json
{
  "prompt": "Explain the water cycle.",
  "chosen": "The water cycle moves water from oceans to the atmosphere and back as rain.",
  "rejected": "Water just appears randomly in nature."
}
```

### 3.2 Output

* The RM produces a scalar reward for each candidate response.
* Higher rewards correspond to better alignment with human preferences.
* For a prompt `p` and candidates `c1, c2`, the model predicts `R(p, c1)` and `R(p, c2)`. The training objective encourages `R(chosen) > R(rejected)`.

---

## 4. Loss Functions

The Reward Model is typically trained with a **pairwise ranking loss**:

### 4.1 Margin Ranking Loss

* Encourages the chosen response to have higher reward than the rejected one.

```python
loss_fn = nn.MarginRankingLoss(margin=1.0)
loss = loss_fn(R_chosen, R_rejected, torch.ones_like(R_chosen))
```

### 4.2 Cross-Entropy with Softmax

* Converts the reward into probabilities via a softmax and maximizes log-likelihood of chosen responses.

```python
import torch.nn.functional as F
logits = torch.cat([R_chosen, R_rejected], dim=1)
labels = torch.zeros(logits.size(0), dtype=torch.long)
loss = F.cross_entropy(logits, labels)
```

### 4.3 Tips

* Normalize rewards if training is unstable.
* Apply gradient clipping to avoid exploding gradients.
* Experiment with different pooling strategies for better context representation.

---

## 5. Training Procedure

### 5.1 Data Loading

* Use `pref_pairs.jsonl` as the main input.
* Batch the data and tokenize using the same tokenizer as the base LLM.
* Ensure padding and attention masks are correctly applied.

### 5.2 Optimization

* Optimizer: AdamW with weight decay (typical for transformers).
* Learning rate: small, e.g., 1e-5 to 5e-5.
* Training steps: depends on dataset size, usually a few thousand steps for prototyping.

### 5.3 Checkpointing

* Save model and optimizer state periodically.
* Store tokenizers alongside the model.
* Optionally save best model according to validation loss.

---

## 6. Evaluation

### 6.1 Metrics

* **Accuracy**: Percentage of pairs where `R(chosen) > R(rejected)`.
* **Mean Margin**: Average difference between chosen and rejected rewards.
* **Spearman Rank Correlation**: Measures correlation with human ranking.

### 6.2 Integration with Human Evaluation

* Use a small set of human-labeled prompts to validate RM quality.
* Can also generate candidate outputs from the SFT model and check reward alignment.

### 6.3 Automatic Evaluation

* Compare reward scores of outputs generated for standard prompts.
* Detect preference drift or scoring inconsistencies.

---

## 7. Integration with RLHF Training

1. **PPO Training**

   * Generate candidate outputs from policy model.
   * Compute rewards using RM.
   * Use rewards for policy gradient updates.

2. **DPO Training**

   * Directly use RM scores to optimize the policy with pairwise comparisons.
   * Encourages the policy to prefer higher-scoring outputs.

3. **Reference Model**\n   - Optionally compare rewards from a reference (baseline) model to stabilize RL updates.

---

## 8. Best Practices

* Use a diverse dataset to prevent bias.
* Regularize reward predictions (dropout, weight decay).
* Monitor overfitting; RM can memorize preferences too quickly.
* Experiment with deeper heads or attention pooling for long-context prompts.
* Normalize reward outputs to a stable range for PPO.
* Validate RM performance continuously during RLHF training.

---

## 9. Summary

The Reward Model is a crucial component for aligning LLMs with human preferences. It provides scalar feedback for candidate outputs, enabling reinforcement learning techniques like PPO and DPO. By carefully designing the architecture, selecting proper loss functions, and integrating evaluation, the RM ensures that models learn to generate outputs aligned with desired behavior. Proper data preparation, checkpointing, and monitoring are essential for market-level, robust deployment.

**Key Points:**

* Input: tokenized prompt + response pairs.
* Output: scalar reward for each candidate.
* Loss: pairwise ranking or cross-entropy.
* Training: AdamW, gradient clipping, checkpointing.
* Evaluation: pairwise accuracy, mean margin, correlation with human judgments.
* Integration: PPO/DPO for policy optimization.
