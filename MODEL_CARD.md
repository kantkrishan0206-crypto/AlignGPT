# Model Card

## Model Overview

AlignGPT is a platform for aligned language-model workflows. It does not ship a single trained model by default. The repository provides scaffolding for policy models, reward models, embeddings, rerankers, multimodal components, and evaluation manifests.

## Intended Use

- Researching alignment methods such as SFT, reward modeling, PPO, and DPO.
- Building web-facing workflows around safe AI inference.
- Running reproducible evaluation and benchmark pipelines.
- Demonstrating production architecture for AI systems.

## Out-of-Scope Use

- High-stakes decision making without domain-specific validation.
- Deployment with unreviewed datasets, unknown model provenance, or disabled safety gates.
- Processing sensitive user data without an approved privacy and retention policy.

## Data Notes

Example JSONL files in `data/` are toy fixtures. Production datasets must include source, license, consent status, schema, quality-control checks, and redaction policy.

## Limitations

- Full model training requires optional ML dependencies and suitable hardware.
- Safety policies are starter controls and require red-team validation before production use.
- Benchmark thresholds are scaffolds until calibrated with representative datasets.

## Risks

- Preference models can encode annotator bias.
- Reward optimization can produce specification gaming.
- Retrieval systems can leak sensitive context if access controls are weak.
- Prompt injection can manipulate tool use or retrieval behavior without layered defenses.

## Safety Considerations

Use policy gates, PII redaction, audit logs, access controls, benchmark regression checks, and staged deployment approvals before exposing models to users.

## Evaluation Summary

The current committed benchmark bundle is a deterministic operational proof run, not a claim about a trained production model. It covers hallucination, latency, throughput, robustness, bias, adversarial behavior, and reproducibility. Production model cards should replace this starter summary with results from the deployed inference backend and governed datasets.
