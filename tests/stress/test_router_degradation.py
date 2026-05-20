from aligngpt.router import GpuAwareInferenceRouter, InferenceRequestProfile, ModelBackend


def test_router_survives_degraded_primary_backend():
    degraded = ModelBackend(
        name="degraded-gpu",
        backend_type="vllm",
        capabilities=("rag",),
        gpu_memory_gb=24,
        reserved_memory_gb=6,
        max_batch_size=8,
        context_window=8192,
        quantization="fp16",
        average_latency_ms=500,
        tokens_per_second=72,
        health="degraded",
        fallback_for=("hosted",),
    )
    hosted = ModelBackend(
        name="hosted",
        backend_type="hosted",
        capabilities=("rag",),
        gpu_memory_gb=0,
        reserved_memory_gb=0,
        max_batch_size=16,
        context_window=16384,
        quantization="bf16",
        average_latency_ms=900,
        tokens_per_second=64,
    )
    router = GpuAwareInferenceRouter((degraded, hosted))

    decision = router.select_backend(InferenceRequestProfile(task="rag", required_capabilities=("rag",)))

    assert decision.backend_name in {"degraded-gpu", "hosted"}
    assert decision.fallback_chain
