# GPU Router Ablation Study

## Question

How much do capacity-aware routing features improve latency and reliability over static backend selection?

## Variants

| Variant | Memory aware | Health aware | Quantization aware | Fallback chain |
| --- | --- | --- | --- | --- |
| Static | no | no | no | no |
| Capacity only | yes | no | no | no |
| Capacity + health | yes | yes | no | yes |
| Full router | yes | yes | yes | yes |

## Metrics

- p95 estimated latency.
- fallback rate.
- rejected routing decisions.
- memory budget violations.
- reward score drift after backend selection.

## Expected Outcome

The full router should reduce memory violations and maintain lower latency under degraded backend conditions while preserving evaluation quality.
