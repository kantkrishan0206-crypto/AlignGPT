from aligngpt.config import PlatformConfig


def test_platform_config_defaults_are_safe():
    cfg = PlatformConfig()
    assert cfg.environment == "development"
    assert cfg.model_backend == "mock"
    assert cfg.safety_profile == "standard"
    assert cfg.max_prompt_chars > 0


def test_platform_config_preserves_unknown_file_metadata(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        '{"environment": "staging", "model_backend": "vllm", "owner": "research"}',
        encoding="utf-8",
    )

    cfg = PlatformConfig.from_file(path)

    assert cfg.environment == "staging"
    assert cfg.model_backend == "vllm"
    assert cfg.metadata["owner"] == "research"
