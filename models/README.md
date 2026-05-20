# Models

The model tree separates model families and lifecycle stages:

- `llm/`: text generation policies.
- `vision/`, `speech/`: future multimodal models.
- `reward_models/`: preference scorers and calibration plans.
- `rerankers/`, `embeddings/`: retrieval support models.
- `fine_tuning/`: adapter and training configuration.
- `quantization/`, `distillation/`: serving optimization.
- `evaluation/`: model evaluation manifests and report schemas.

Heavy implementation currently remains in the legacy `src/models` and `src/training` modules until migrated into package APIs.
