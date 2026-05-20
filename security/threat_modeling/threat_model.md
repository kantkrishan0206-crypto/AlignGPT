# Threat Model

## Assets

- User prompts and responses.
- Dataset manifests and private corpora.
- Model checkpoints and adapters.
- API keys, service tokens, and cloud credentials.
- Benchmark and safety reports.

## Primary Risks

- Prompt injection manipulates tool use or retrieval.
- Secrets leak through logs, traces, or model outputs.
- Unauthorized model deployment promotion.
- Dataset license or PII violations.
- Benchmark gaming or untracked regression.

## Mitigations

- Policy gates, redaction, access control, audit logs, CI scans, and staged promotion.
