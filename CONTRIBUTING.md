# Contributing

AlignGPT is organized as a research and production platform. Contributions should preserve import safety, reproducibility, and clear ownership boundaries.

## Workflow

1. Create an issue for non-trivial changes.
2. Branch from `main` using a descriptive name, for example `feature/eval-runner` or `fix/reward-schema`.
3. Keep pull requests focused on one behavior or subsystem.
4. Add or update tests for changed behavior.
5. Update documentation when interfaces, configs, workflows, or assumptions change.

## Coding Standards

- Prefer small modules with explicit dataclasses, schemas, or typed functions.
- Keep heavy ML imports lazy. Core package imports must not download models or require GPUs.
- Use config files for experiment parameters rather than hard-coded values.
- Favor deterministic tests and small fixtures.
- Keep TODOs specific: owner, missing capability, and acceptance criterion.

## Branch Strategy

- `main`: stable, reviewed platform state.
- `feature/*`: product, backend, frontend, or SDK work.
- `research/*`: experiments, ablations, and benchmark additions.
- `infra/*`: deployment, CI, observability, and security automation.

## Pull Request Expectations

Every PR should include:

- Problem statement and design summary.
- Test evidence.
- Documentation impact.
- Security and privacy considerations when data, prompts, secrets, or access control are touched.
- Rollback or migration notes for operational changes.

## Issue Rules

Use the issue templates under `.github/ISSUE_TEMPLATE`. Research issues should include method, dataset, metric, baseline, and reproducibility notes. Product issues should include user workflow, acceptance criteria, and API/UI impact.
