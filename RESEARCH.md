# Research

AlignGPT centers on alignment research that can be reproduced, audited, and productized.

## Scientific Scope

- Supervised fine-tuning for instruction-following baselines.
- Reward modeling from pairwise human or synthetic preferences.
- Policy optimization with PPO and DPO.
- Retrieval-augmented generation with provenance-aware evaluation.
- Safety evaluation across prompt injection, jailbreak, bias, privacy, and robustness scenarios.

## Methodology

Each experiment should define:

- Hypothesis.
- Dataset source and license.
- Model and adapter configuration.
- Training or inference budget.
- Evaluation metrics and thresholds.
- Baseline and ablation plan.
- Failure analysis protocol.
- Reproducibility manifest.

## Benchmark Strategy

Benchmarks are grouped by scientific risk:

- Capability: task accuracy, coherence, reasoning, and instruction adherence.
- Alignment: preference win rate, reward score calibration, refusal appropriateness, and helpfulness.
- Safety: jailbreak resistance, prompt-injection resistance, PII leakage, and toxicity.
- Systems: latency, throughput, cost, cache hit rate, and degradation under load.

## Reproducibility

Runs should record dataset fingerprint, model revision, config hash, random seed, dependency lock, hardware profile, and artifact locations. Lightweight smoke benchmarks can run in CI; GPU-heavy experiments should run in scheduled pipelines and publish summarized reports.
