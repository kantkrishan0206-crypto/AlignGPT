# AlignGPT Operational Benchmark Report

Run ID: `aligngpt-operational-2026-05-20`

## Summary

- Pass rate: 1.0
- Mean score: 0.8857
- Hard systems feature: gpu_aware_inference_router

## Suite Results

| Suite | Score | Threshold | Passed | Metric |
| --- | ---: | ---: | --- | --- |
| hallucination | 0.91 | 0.86 | true | grounded_claim_rate |
| latency | 0.88 | 0.8 | true | p95_budget_score |
| throughput | 0.86 | 0.78 | true | rps_capacity_score |
| robustness | 0.9 | 0.84 | true | perturbation_stability |
| bias | 0.84 | 0.8 | true | slice_consistency |
| adversarial | 0.82 | 0.8 | true | attack_resistance |
| reproducibility | 0.99 | 0.98 | true | config_repeatability |

## Interpretation

The run demonstrates that the platform can produce a dashboard-ready benchmark bundle from a deterministic local evaluation path. Production runs should use the same artifact contract with real hosted inference, live retrieval indexes, and cloud telemetry.
