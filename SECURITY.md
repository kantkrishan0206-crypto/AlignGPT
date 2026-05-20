# Security Policy

AlignGPT treats safety, privacy, and operational security as core platform requirements.

## Supported Versions

The `main` branch is the active supported development line until tagged releases are introduced.

## Reporting Vulnerabilities

Please do not open public issues for vulnerabilities. Report privately to the repository owner with:

- A concise description of the issue.
- Reproduction steps or proof of concept.
- Impacted files, endpoints, configs, or workflows.
- Suggested remediation if known.

The maintainers should acknowledge valid reports within 72 hours and coordinate disclosure once a fix is available.

## Secret Handling

- Never commit API keys, tokens, private datasets, checkpoints with sensitive memorized data, or cloud credentials.
- Use `.env` files locally and secret stores in deployment environments.
- CI should use GitHub Actions secrets with least privilege.
- Logs must not include prompts, responses, user identifiers, or headers unless explicitly redacted.

## Safe Defaults

- `trust_remote_code` defaults to `false` in model configs.
- Networked model or data access must be explicit.
- Safety gates should run before retrieval, tool use, and generation.
- PII redaction policies should be enabled for user-facing traces and analytics exports.

## Security Review Triggers

Request security review when changing auth, rate limiting, prompt handling, retrieval filters, tool execution, deployment manifests, CI secrets, or data retention behavior.
