# Why This Architecture

AlignGPT is organized around the observation that alignment engineering is not only a training problem. A useful deployed alignment platform needs to connect research methods, production routing, safety review, benchmark evidence, and operational monitoring.

## Design Principles

1. **Research and product share contracts.** The same request, route, metric, and report schemas power tests, local demos, benchmark artifacts, API responses, and dashboard data.
2. **The model path is explainable.** The GPU-aware router emits memory, latency, quantization, fallback, and health decisions so operators can debug why a request used a specific backend.
3. **Evaluation is operational.** Benchmarks are not side notebooks. They produce artifacts that dashboards and release gates can consume.
4. **Safety happens before and after generation.** Prompt injection and PII checks run before retrieval/model execution; response checks and findings are attached to trace output.
5. **Deployment is visible early.** Docker, Kubernetes, Helm, Vercel, Render, Railway, metrics, and runbooks exist before the system reaches production traffic.

## Hard Systems Feature

The GPU-aware inference router solves the practical systems problem of selecting a model backend under changing capacity and risk constraints. It considers context window, estimated memory, batch size, health, quantization, latency budget, fallback chain, and task capability. This is the control point that lets a research model become a deployable service rather than a notebook artifact.

## Why This Becomes A SaaS Platform

The public web app communicates the product value; the internal dashboard exposes benchmark, routing, deployment, and trace state; the API executes the alignment flow; the benchmark generator produces release evidence; the observability layer exports metrics. With cloud credentials, the repo is ready to connect Vercel, Render/Railway, Postgres, Redis, and hosted inference into a live internet-facing service.
