from aligngpt.router import (
    DEFAULT_BACKENDS,
    GpuAwareInferenceRouter,
    InferenceRequestProfile,
    ModelBackend,
)


def test_router_selects_gpu_backend_for_normal_rag_request():
    router = GpuAwareInferenceRouter(DEFAULT_BACKENDS)

    decision = router.select_backend(
        InferenceRequestProfile(
            task="rag",
            prompt_tokens=120,
            expected_output_tokens=220,
            required_capabilities=("rag",),
            latency_budget_ms=2500,
        )
    )

    assert decision.backend_name == "vllm-a10g-primary"
    assert decision.microbatch_size == 1
    assert "hosted-alignment-api" in decision.fallback_chain


def test_router_avoids_offline_backend_and_uses_fallback():
    offline_gpu = ModelBackend(
        name="gpu-offline",
        backend_type="vllm",
        capabilities=("rag",),
        gpu_memory_gb=24,
        reserved_memory_gb=4,
        max_batch_size=8,
        context_window=8192,
        quantization="fp16",
        average_latency_ms=200,
        tokens_per_second=100,
        health="offline",
    )
    hosted = DEFAULT_BACKENDS[1]
    router = GpuAwareInferenceRouter((offline_gpu, hosted))

    decision = router.select_backend(InferenceRequestProfile(task="rag", required_capabilities=("rag",)))

    assert decision.backend_name == "hosted-alignment-api"


def test_router_microbatch_ranges_are_stable():
    router = GpuAwareInferenceRouter(DEFAULT_BACKENDS)
    assert router.create_microbatches(total_items=10, microbatch_size=4) == ((0, 4), (4, 8), (8, 10))
