import json

from aligngpt.operational_benchmark import write_operational_benchmark


def test_operational_benchmark_writes_artifacts(tmp_path):
    paths = write_operational_benchmark(tmp_path)

    assert paths["summary"].exists()
    assert paths["metrics_csv"].exists()
    assert paths["report"].exists()
    assert paths["comparison"].exists()

    payload = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert payload["summary"]["hard_systems_feature"] == "gpu_aware_inference_router"
    assert payload["summary"]["pass_rate"] == 1.0
