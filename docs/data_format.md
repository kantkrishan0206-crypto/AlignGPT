# Data Formats for RLHF-Lab

This document explains the data formats expected by the RLHF-Lab repository. Accurate data formatting is crucial for smooth training, evaluation, and fine-tuning of models. The project uses JSON Lines (`.jsonl`) as the primary data format due to its simplicity, efficiency, and compatibility with streaming large datasets.

---

## 1. Supervised Fine-Tuning (SFT) Data Format

The SFT dataset contains prompt-response pairs for training the base LLM using supervised learning. Each line in the dataset should be a valid JSON object with the following structure:

```json
{
  "prompt": "<prompt_text>",
  "response": "<response_text>"
}
```

### Fields

* `prompt`: A string representing the user query, instruction, or context that the model should respond to. Keep the text concise but informative.
* `response`: A string containing the desired model output. This is the ground-truth response the model should learn to generate.

### Example

```json
{
  "prompt": "Explain photosynthesis in simple terms.",
  "response": "Photosynthesis is the process by which plants use sunlight to make food from carbon dioxide and water."
}
```

### Guidelines

* Ensure prompts are clear and unambiguous.
* Responses should be informative, factual, and aligned with human preferences.
* Avoid extremely long texts; if necessary, truncate or split into smaller examples.
* UTF-8 encoding is required.

---

## 2. Preference Pair Data Format

Preference pair datasets are used for Reward Model (RM) training and DPO/PPO fine-tuning. Each entry consists of a prompt and two candidate responses, labeled `chosen` and `rejected` to indicate human preference.

```json
{
  "prompt": "<prompt_text>",
  "chosen": "<preferred_response>",
  "rejected": "<less_preferred_response>"
}
```

### Fields

* `prompt`: The same as in SFT data; a user query or context.
* `chosen`: The response preferred by human annotators or simulated preference metric.
* `rejected`: The alternative response considered less optimal.

### Example

```json
{
  "prompt": "Summarize the plot of 'Romeo and Juliet'.",
  "chosen": "Romeo and Juliet are young lovers whose families are enemies. Their love leads to tragic consequences.",
  "rejected": "Romeo and Juliet had a small argument and then got married."
}
```

### Guidelines

* Ensure `chosen` responses are more accurate, helpful, or aligned with the intended instruction.
* The `rejected` response should be plausible but clearly less preferred.
* Pair generation can be automated via model sampling followed by ranking or through human annotation.
* Maintain a balanced dataset with diverse prompts.

---

## 3. Prompt Dataset Format

In addition to SFT and preference data, a prompt-only dataset can be used for candidate generation, evaluation, or pretraining tasks. Each line is a simple JSON object:

```json
{
  "prompt": "<prompt_text>"
}
```

### Example

```json
{
  "prompt": "Translate the following English sentence into French: 'Hello, how are you?'"
}
```

### Guidelines

* Prompts should cover a wide range of topics.
* Maintain high-quality text to ensure meaningful model outputs.
* Can be used to generate multiple candidate responses using `generate_candidates.py`.

---

## 4. JSONL Best Practices

* Each line must be a valid JSON object.
* Avoid trailing commas and ensure proper escaping of quotes.
* UTF-8 encoding is mandatory.
* Keep files reasonably sized for streaming; very large files can be split into multiple `.jsonl` files.
* Consistent key names are important to prevent runtime errors.
* Optional: include metadata fields such as `source`, `difficulty`, `category` to enrich training data.

---

## 5. Dataset Organization

Recommended folder structure for datasets:

```
data/
├─ sft.jsonl           # Supervised fine-tuning data
├─ pref_pairs.jsonl    # Human or simulated preference pairs
└─ prompts.jsonl       # Prompt-only dataset
```

* SFT datasets are used by `sft_trainer.py`.
* Preference pairs are used by `rm_trainer.py`, `ppo_trainer.py`, and `dpo_trainer.py`.
* Prompt datasets can be used to generate candidates or evaluate model outputs.

---

## 6. Tips for High-Quality Data

1. **Diversity**: Include prompts from multiple domains to avoid overfitting.
2. **Clarity**: Avoid ambiguous or contradictory prompts and responses.
3. **Consistency**: Ensure labeling is consistent across `chosen` and `rejected` responses.
4. **Cleaning**: Remove duplicates, formatting artifacts, or inappropriate content.
5. **Simulated Preferences**: If human labels are unavailable, you can generate preference pairs using heuristic or scoring methods.
6. **Data Augmentation**: Use paraphrasing or back-translation to expand dataset coverage.

---

## 7. Integration with Training Scripts

* `sft_trainer.py` reads `sft.jsonl` for supervised training.
* `rm_trainer.py` reads `pref_pairs.jsonl` to train the reward model.
* `ppo_trainer.py` and `dpo_trainer.py` use the reward model to score outputs generated from prompts in `prompts.jsonl` or `pref_pairs.jsonl`.
* Scripts expect JSONL input, handle batching, tokenization, and collate tensors automatically.

---

## 8. Summary

Maintaining correct data formats ensures smooth operation of the RLHF-Lab pipeline. Following these conventions allows the models to be trained end-to-end using SFT, RM, PPO, and DPO methods without runtime errors, and ensures reproducibility across experiments.

* SFT: `{prompt, response}`
* Preference Pairs: `{prompt, chosen, rejected}`
* Prompt-only: `{prompt}`
* All files in JSONL, UTF-8 encoded

Properly formatted datasets are the foundation for aligning LLMs with human preferences and achieving high-quality, market-ready RLHF models.
