# Operational Workflows

AlignGPT now exposes three runnable vertical flows.

## 1. Alignment Evaluation

```bash
python - <<'PY'
import sys
sys.path.insert(0, "src")
from aligngpt.alignment_pipeline import AlignmentEvaluationPipeline
from aligngpt.schemas import AlignmentRequest

pipeline = AlignmentEvaluationPipeline()
result = pipeline.run(AlignmentRequest(prompt="Explain safe reward modeling with retrieval."))
print(result.reward_score)
print(result.route.backend_name)
PY
```

This path validates the prompt, applies safety rules, retrieves context, selects a backend with the GPU-aware router, generates a deterministic response, scores reward/grounding, emits metrics, and returns trace events.

## 2. Benchmark + Reproducibility

```bash
python benchmarking/reproducibility/run_operational_benchmark.py \
  --output-dir benchmarking/results/2026-05-20
```

The run writes JSON, CSV, Markdown, and comparison artifacts. The same artifact contract can be uploaded by CI or scheduled jobs to object storage.

## 3. Deployment + Observability

```bash
python -m pip install -e ".[api]"
uvicorn backend.api_gateway.app:app --reload --port 8000
curl http://localhost:8000/ready
curl http://localhost:8000/metrics
```

The API exposes health, readiness, Prometheus metrics, event streaming, alignment inference, and evaluation endpoints.
