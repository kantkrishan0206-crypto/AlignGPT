# Data Formats

Prototype datasets use JSON Lines so records can be streamed and validated independently.

## Supervised Fine-Tuning

File: `data/sft.jsonl`

```json
{"prompt": "Explain overfitting.", "response": "Overfitting occurs when..."}
```

## Prompt Set

File: `data/prompts.jsonl`

```json
{"prompt": "Summarize reward modeling in two sentences."}
```

## Preference Pairs

File: `data/pref_pairs.jsonl`

```json
{"prompt": "Explain RLHF.", "chosen": "A strong answer.", "rejected": "A weak answer."}
```

Production datasets should be represented by manifests under `datasets/dataset_registry/` with license, source, schema, and quality-control metadata.
