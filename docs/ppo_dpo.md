# PPO and DPO Training Documentation for RLHF-Lab

This document explains the PPO (Proximal Policy Optimization) and DPO (Direct Preference Optimization) training modules used in RLHF-Lab. These methods are used to optimize language models according to human preferences, using feedback from the Reward Model (RM). The guide covers algorithm overview, data requirements, training loops, loss functions, evaluation, and best practices.

---

## 1. Overview of PPO and DPO

### 1.1 Proximal Policy Optimization (PPO)

PPO is a policy-gradient reinforcement learning algorithm designed to stabilize updates. In the context of RLHF:

* **Policy Model**: The LLM trained to generate responses.
* **Reward Signal**: Provided by the Reward Model based on human or simulated preferences.
* **Objective**: Maximize expected reward while avoiding large updates that could destabilize the model.

PPO uses a clipped surrogate loss to prevent excessive policy updates, ensuring stable learning.

### 1.2 Direct Preference Optimization (DPO)

DPO is an alternative to PPO, designed to directly optimize the policy using pairwise preference comparisons:

* **Input**: Prompt with two candidate responses (chosen vs rejected).
* **Loss**: Encourages the model to assign higher likelihood to the preferred response.
* **Advantage**: Simpler and more computationally efficient than PPO.

DPO is particularly effective for fine-tuning LLMs on preference datasets without the complexity of full RL algorithms.

---

## 2. Input Data

Both PPO and DPO rely on the Reward Model and preference data:

### 2.1 Preference Pairs (`pref_pairs.jsonl`)

```json
{
  "prompt": "Explain the water cycle.",
  "chosen": "The water cycle moves water from oceans to the atmosphere and back as rain.",
  "rejected": "Water just appears randomly in nature."
}
```

* **prompt**: The user instruction or query.
* **chosen**: Preferred response.
* **rejected**: Less preferred response.

### 2.2 Optional SFT Checkpoints

* Load a policy model pretrained via Supervised Fine-Tuning for better initialization.
* Helps in stabilizing RLHF training and reduces catastrophic forgetting.

---

## 3. PPO Training Procedure

### 3.1 Generate Candidate Responses

* For each prompt, the policy model generates multiple candidate outputs.
* Use the `generate` method with temperature, top-k, or nucleus sampling.

### 3.2 Reward Computation

* Pass candidates through the Reward Model.
* Compute scalar rewards `R(prompt, response)`.
* Optionally normalize rewards for stable learning.

### 3.3 Compute PPO Loss

* **Surrogate Loss**: Compare new and old policy probabilities.
* **Clipping**: Limit the ratio of new/old policy to prevent large updates.
* **Value Loss**: Optional, for estimating expected reward.
* **Entropy Bonus**: Encourages exploration and diversity in responses.

```python
ratio = torch.exp(log_probs_new - log_probs_old)
loss_clip = torch.min(ratio * advantages, torch.clamp(ratio, 1-eps, 1+eps) * advantages)
loss = -loss_clip.mean() + c1 * value_loss - c2 * entropy_bonus
```

### 3.4 Optimization

* Optimizer: AdamW
* Gradient clipping to prevent explosion.
* Mini-batch updates with multiple epochs per rollout.
* Checkpoint policy model periodically.

---

## 4. DPO Training Procedure

### 4.1 Input Preparation

* Use preference pairs as input: `(prompt, chosen, rejected)`.
* Tokenize using the same tokenizer as the policy model.

### 4.2 Compute Logits and Loss

* Compute log-probabilities of `chosen` and `rejected` under the policy.
* Use pairwise cross-entropy loss to encourage higher probability for `chosen`:

```python
logits_chosen = policy(input_ids_chosen)
logits_rejected = policy(input_ids_rejected)
loss = -torch.log_softmax(logits_chosen - logits_rejected, dim=-1).mean()
```

* Optional scaling or regularization to stabilize training.

### 4.3 Optimization

* Similar to PPO: AdamW, gradient clipping, checkpointing.
* Multiple epochs over the preference dataset.
* Simpler pipeline, no reward normalization needed.

---

## 5. Integration with Reward Model

* PPO and DPO rely on the Reward Model to assign preference scores.
* PPO: Uses scalar rewards to compute advantage estimates.
* DPO: Uses RM to simulate pairwise preferences when human labels are unavailable.
* Regularly update the RM to maintain alignment with evolving policy outputs.

---

## 6. Evaluation Metrics

### 6.1 Pairwise Accuracy

* Measures fraction of preference pairs where policy assigns higher probability to `chosen`.
* Key metric for both PPO and DPO.

### 6.2 Reward Improvement

* Track average reward from RM over generated outputs.
* Evaluate whether RLHF training improves alignment over SFT baseline.

### 6.3 Human Evaluation

* Optional human-in-the-loop validation to ensure generated responses match human judgment.
* Useful for detecting reward hacking or misalignment.

### 6.4 Automatic Evaluation

* BLEU, ROUGE, or semantic similarity can be used as supplementary metrics.
* Useful for monitoring output quality during training.

---

## 7. Best Practices

* Initialize PPO/DPO from a well-trained SFT model.
* Normalize or scale rewards for PPO to prevent large updates.
* Use entropy bonuses in PPO to encourage diverse outputs.
* Avoid overfitting RM: use a separate validation set.
* Monitor training stability: PPO can diverge if updates are too large.
* Regularly checkpoint policy and RM models.
* Start with small datasets for prototyping, then scale.
* Log metrics using TensorBoard or Weights & Biases for visualization.

---

## 8. Folder and Script Integration

```
scripts/
├─ run_ppo.sh       # Launch PPO training
├─ run_dpo.sh       # Launch DPO training
```

* Training scripts read YAML configs from `configs/ppo_gpt2.yaml` or `configs/dpo_gpt2.yaml`.
* Checkpoints stored in `checkpoints/`.
* Logs stored in `logs/` with performance metrics.

`src/training/ppo_trainer.py` and `dpo_trainer.py` implement the full training loop, including:

* Data loading and batching
* Candidate generation
* Reward computation
* Loss calculation
* Optimizer step
* Checkpointing
* Evaluation hooks

---

## 9. Summary

PPO and DPO are two RLHF methods to fine-tune language models based on human or simulated preferences.

* **PPO**: Stable RL with surrogate loss, reward normalization, entropy bonus, and clipping.
* **DPO**: Direct pairwise preference optimization, simpler and faster.
* **Integration**: Both methods rely on a Reward Model for scoring outputs.
* **Evaluation**: Pairwise accuracy, reward improvement, and human validation.
* **Best Practices**: Start with SFT models, monitor metrics, use checkpointing, avoid reward hacking.

By following the structured workflow and data formats described in previous sections, PPO and DPO training can effectively align LLM outputs with human preferences, producing market-ready, high-quality RLHF models.
