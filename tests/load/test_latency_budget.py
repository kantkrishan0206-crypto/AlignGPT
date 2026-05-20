from aligngpt.router import DEFAULT_BACKENDS, GpuAwareInferenceRouter, InferenceRequestProfile


def test_latency_budget_routes_under_expected_limit():
    router = GpuAwareInferenceRouter(DEFAULT_BACKENDS)

    decision = router.select_backend(
        InferenceRequestProfile(
            task="rag",
            prompt_tokens=256,
            expected_output_tokens=128,
            latency_budget_ms=2200,
            required_capabilities=("rag",),
        )
    )

    assert decision.estimated_latency_ms <= 2200
