# AlignGPT Metrics

The API exports Prometheus text metrics at `/metrics`.

## Core Metrics

- `aligngpt_requests_total{task}`: request volume by task.
- `aligngpt_estimated_latency_ms_count{backend}`: routed request count.
- `aligngpt_estimated_latency_ms_sum{backend}`: estimated latency sum.
- `aligngpt_estimated_latency_ms_max{backend}`: max estimated latency.
- `aligngpt_reward_score_count{backend}`: reward-scored requests.
- `aligngpt_reward_score_sum{backend}`: reward score total.
- `aligngpt_safety_findings_total{severity}`: safety finding count.

## Operational Use

Dashboards should alert when latency exceeds budget, reward score drifts downward, safety findings spike, or the router starts using fallback backends frequently.
