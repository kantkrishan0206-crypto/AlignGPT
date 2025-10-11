# Evaluation Protocols

Evaluation ensures the aligned model is:
- Helpful
- Honest
- Harmless

## 1. Automatic Metrics
- Perplexity (fluency)
- Reward Model score
- Diversity (distinct n‑grams)

## 2. Preference Evaluation
- Sample multiple responses per prompt
- Rank with Reward Model
- Compare against baseline SFT

## 3. Human Evaluation (optional)
- Side‑by‑side comparisons
- Annotator ratings for helpfulness, safety, factuality

## 4. Logging
- Results stored in `logs/eval/`
- TensorBoard/W&B dashboards for visualization