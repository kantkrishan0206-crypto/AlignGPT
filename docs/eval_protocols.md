# Evaluation Protocols for RLHF-Lab (Expanded)

This document provides a comprehensive explanation of the evaluation protocols used in RLHF-Lab to assess the performance, alignment, and safety of large language models (LLMs) trained with Reinforcement Learning from Human Feedback (RLHF). Evaluation is essential to ensure that models generate responses that are accurate, contextually relevant, safe, and aligned with human preferences. This expanded guide covers objectives, datasets, automatic and human evaluation methods, metrics, pipeline workflows, analysis, reporting, and best practices.

---

## 1. Objectives of Evaluation

The evaluation protocols aim to achieve several objectives:

1. **Alignment Assessment**: Ensure that the LLM produces responses that match human preferences.
2. **Quality Control**: Maintain fluency, coherence, factual correctness, and context relevance.
3. **Comparative Analysis**: Measure improvements over baseline models such as SFT (Supervised Fine-Tuning) and other RLHF methods like PPO and DPO.
4. **Safety and Robustness**: Detect and mitigate harmful, biased, or unsafe outputs.
5. **Benchmarking**: Compare model performance across different datasets, tasks, and model versions.
6. **Model Iteration Guidance**: Provide feedback for improving Reward Models, training datasets, and fine-tuning strategies.

Evaluation ensures continuous monitoring and prevents regression during training and deployment.

---

## 2. Evaluation Datasets

### 2.1 Preference Pair Dataset (`pref_pairs.jsonl`)

* Used to train and evaluate models on their ability to prefer better outputs.
* Each entry contains a prompt, a preferred (`chosen`) response, and a less preferred (`rejected`) response.
* Example:

```json
{
  "prompt": "Explain the water cycle.",
  "chosen": "The water cycle moves water from oceans to the atmosphere and back as rain.",
  "rejected": "Water just appears randomly in nature."
}
```

* Key metric: Pairwise accuracy.

### 2.2 Prompt Dataset (`prompts.jsonl`)

* Used to assess model generalization across a variety of domains.
* Contains prompts without reference responses.
* Helps evaluate open-ended generation quality, diversity, and coherence.
* Example:

```json
{
  "prompt": "Translate 'Hello, how are you?' into French."
}
```

### 2.3 Human-Labeled Dataset

* Optional but highly recommended.
* Annotated by human evaluators for alignment, helpfulness, factual correctness, and safety.
* Provides ground truth for model evaluation and RM validation.
* Useful for benchmarking RLHF improvements over SFT baselines.

### 2.4 Synthetic or Simulated Preference Data

* Can be generated using the Reward Model or automated scoring functions.
* Used for rapid iteration and scaling evaluation when human labels are limited.
* Should be validated periodically to avoid drift or reward hacking.

---

## 3. Automatic Evaluation Methods

Automatic evaluation uses quantitative metrics that do not require human intervention. These include:

### 3.1 Pairwise Accuracy

* Measures how often the model assigns higher scores or probabilities to `chosen` responses over `rejected` responses.
* Computed using log-probabilities from the policy model or reward scores from the RM.
* Formula:

```
accuracy = (# of pairs where R(chosen) > R(rejected)) / total_pairs
```

### 3.2 Reward Improvement

* Tracks average reward of model outputs as predicted by the Reward Model.
* Monitors alignment improvements after PPO or DPO fine-tuning compared to SFT baseline.

### 3.3 Perplexity and Likelihood

* Measures the model’s ability to predict token sequences.
* Lower perplexity indicates better fluency and language modeling.
* Helps detect forgetting of previously learned SFT knowledge.

### 3.4 Semantic Metrics

* BLEU, ROUGE, METEOR: Token-level similarity with reference outputs.
* Embedding-based metrics (e.g., BERTScore, cosine similarity): Semantic similarity between generated and reference responses.
* Provide insight into content quality beyond simple token matching.

### 3.5 Safety and Robustness Metrics

* Toxicity scoring using pretrained classifiers.
* Detects harmful, biased, or unsafe language.
* Measures incoherence and factual errors.
* Quantifies the failure rate of model outputs in various domains.

---

## 4. Human Evaluation Methods

Human-in-the-loop evaluation ensures that the model truly aligns with human judgments.

### 4.1 Pairwise Comparison

* Annotators receive a prompt with two candidate responses.
* Task: select which response is better in terms of helpfulness, alignment, and factuality.
* Provides ground truth for RM and model validation.

### 4.2 Rating Scale Evaluation

* Annotators rate each response on a Likert scale (1–5) for:

  * Quality
  * Helpfulness
  * Factual correctness
  * Safety
* Ratings are aggregated to compute average performance metrics.

### 4.3 Open-ended Feedback

* Qualitative annotations describing why a response is poor or superior.
* Useful for identifying edge cases, hallucinations, or misaligned outputs.

### 4.4 Blind Testing

* Evaluators are unaware of the model version to prevent bias.
* Ensures fair comparison between SFT, PPO, and DPO-trained models.

---

## 5. Evaluation Metrics

### 5.1 Alignment Metrics

* **Pairwise Accuracy**: Measures preference alignment.
* **Win Rate**: Frequency of the model’s output preferred over a baseline.

### 5.2 Quality Metrics

* **Perplexity**: Fluency and predictability of generated text.
* **BLEU / ROUGE / METEOR**: Token-level similarity with references.
* **BERTScore / Embedding Similarity**: Semantic similarity evaluation.
* **Coherence Score**: Assesses logical flow of multi-turn dialogues.

### 5.3 Reward Metrics

* **Average Reward**: Mean RM score of generated outputs.
* **Reward Margin**: Difference between `chosen` and `rejected` responses.
* **Correlation with Human Preference**: Spearman or Pearson correlation to validate RM predictions.

### 5.4 Safety Metrics

* **Toxicity Rate**: Fraction of outputs flagged as unsafe.
* **Hallucination Rate**: Outputs contradicting known facts.
* **Incoherence Rate**: Responses that are contextually irrelevant or nonsensical.

---

## 6. Evaluation Pipeline Workflow

1. **Data Loading**: Load prompts, preference pairs, and human-labeled data.
2. **Candidate Generation**: Generate outputs from the policy model using SFT, PPO, or DPO weights.
3. **Reward Scoring**: Compute RM scores for generated outputs.
4. **Automatic Metrics Calculation**: Compute pairwise accuracy, reward improvement, perplexity, BLEU/ROUGE, and safety metrics.
5. **Human Evaluation**: Perform pairwise comparisons, ratings, and qualitative feedback.
6. **Logging and Analysis**: Record metrics in logs, dashboards, or CSV files for reproducibility.
7. **Reporting**: Summarize findings in tables, charts, and reports for model iteration.

---

## 7. Best Practices

* Combine automatic and human evaluation for a comprehensive view.
* Ensure diversity of prompts to avoid overfitting to specific domains.
* Validate RM predictions regularly with human-labeled data.
* Monitor safety and alignment metrics continuously.
* Use blind evaluation to prevent bias.
* Aggregate results over multiple runs for reliability.
* Update evaluation datasets as models evolve to capture new behaviors.
* Document all evaluation procedures for reproducibility.
* Maintain separate validation sets for training RM and evaluating RLHF models.
* Include multi-turn dialogues to assess context retention.
* Track metrics across model checkpoints to detect performance regression.

---

## 8. Integration with RLHF-Lab

* Evaluation scripts reside in `src/eval/`:

  * `metrics.py`: Automatic metric computations.
  * `eval_human.py`: Human evaluation pipeline.
  * `eval_automatic.py`: Automated batch evaluation.
* Supports evaluation of SFT, RM, PPO, and DPO models.
* Logs metrics for checkpoint selection and model comparison.
* Integration ensures evaluation can be run continuously during training or after deployment.

---

## 9. Summary

Evaluation is essential for aligning LLM outputs with human preferences while ensuring quality and safety. RLHF-Lab uses both automatic and human evaluation methods to measure:

* Alignment with human preferences (pairwise accuracy, RM correlation)
* Language fluency and semantic quality (perplexity, BLEU, ROUGE, embedding similarity)
* Safety, robustness, and factual correctness (toxicity, hallucination, incoherence rates)

Following these expanded protocols ensures thorough assessment of model performance, supports safe deployment, and guides continuous improvement of RLHF-trained models. Comprehensive evaluation helps detect failures, improve Reward Models, refine training datasets, and monitor alignment over time, making the LLM both market-ready and human-aligned.
