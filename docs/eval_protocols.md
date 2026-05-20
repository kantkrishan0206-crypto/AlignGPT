# Evaluation Protocols

Evaluation should measure capability, alignment, safety, and system behavior.

## Automatic Metrics

- Perplexity or loss where relevant.
- Reward-model score.
- Lexical diversity and repetition.
- Required-term coverage for deterministic smoke checks.
- Retrieval citation faithfulness.

## Preference Evaluation

- Sample multiple responses per prompt.
- Rank with reward model and human review where available.
- Compare against baseline SFT and previous promoted model.

## Human Evaluation

- Side-by-side response comparison.
- Rubrics for helpfulness, honesty, harmlessness, and factuality.
- Annotator guidelines and disagreement tracking.

## Logging

Store metrics, non-sensitive traces, config hashes, dataset fingerprints, and artifact references. Do not log raw sensitive prompts or responses without an explicit privacy review.
