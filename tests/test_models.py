"""
Tests for multi-model registry and provider support.
"""

import json
import os
import pytest
import tempfile
from pathlib import Path

from mini_pi.models import (
    ModelRegistry,
    ModelInfo,
    BUILTIN_MODELS,
    create_llm,
    get_model_extra_kwargs,
)
from mini_pi.config import Config


# ── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def mock_env(monkeypatch):
    """Set up mock API keys for testing."""
    monkeypatch.setenv("MOONSHOT_API_KEY", "sk-moonshot-test")
    monkeypatch.setenv("ZHIPU_API_KEY", "sk-zhipu-test")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")


@pytest.fixture
def registry(mock_env):
    """Create a registry with mock API keys."""
    return ModelRegistry()


# ── Builtin Models ──────────────────────────────────────────────────

class TestBuiltinModels:
    def test_builtin_has_providers(self):
        assert "kimi" in BUILTIN_MODELS["providers"]
        assert "zhipu" in BUILTIN_MODELS["providers"]
        assert "deepseek" in BUILTIN_MODELS["providers"]
        assert "openai" in BUILTIN_MODELS["providers"]

    def test_kimi_models(self):
        kimi = BUILTIN_MODELS["providers"]["kimi"]
        assert "kimi-k2.6" in kimi["models"]
        assert "kimi-k2.5" in kimi["models"]
        assert kimi["base_url"] == "https://api.moonshot.cn/v1"

    def test_zhipu_models(self):
        zhipu = BUILTIN_MODELS["providers"]["zhipu"]
        assert "glm-5.1" in zhipu["models"]
        assert "glm-5" in zhipu["models"]
        assert "glm-4.7" in zhipu["models"]
        assert zhipu["base_url"] == "https://open.bigmodel.cn/api/paas/v4/"

    def test_zhipu_thinking_config(self):
        glm51 = BUILTIN_MODELS["providers"]["zhipu"]["models"]["glm-5.1"]
        assert glm51["thinking"] is True
        assert glm51["thinking_type"] == "enabled"
        assert glm51["temperature"] == 1.0


# ── ModelRegistry ───────────────────────────────────────────────────

class TestModelRegistry:
    def test_resolve_explicit_kimi(self, registry):
        info = registry.resolve("kimi/kimi-k2.6")
        assert info is not None
        assert info.provider == "kimi"
        assert info.model == "kimi-k2.6"
        assert info.api_key == "sk-moonshot-test"
        assert info.base_url == "https://api.moonshot.cn/v1"
        assert info.thinking is True

    def test_resolve_explicit_zhipu(self, registry):
        info = registry.resolve("zhipu/glm-5.1")
        assert info is not None
        assert info.provider == "zhipu"
        assert info.model == "glm-5.1"
        assert info.api_key == "sk-zhipu-test"
        assert info.base_url == "https://open.bigmodel.cn/api/paas/v4/"
        assert info.thinking is True
        assert info.thinking_type == "enabled"
        assert info.temperature == 1.0

    def test_resolve_explicit_deepseek(self, registry):
        info = registry.resolve("deepseek/deepseek-chat")
        assert info is not None
        assert info.provider == "deepseek"
        assert info.api_key == "sk-deepseek-test"

    def test_resolve_explicit_openai(self, registry):
        info = registry.resolve("openai/gpt-4o")
        assert info is not None
        assert info.provider == "openai"
        assert info.api_key == "sk-openai-test"

    def test_resolve_fuzzy_by_model_name(self, registry):
        """Without provider prefix, search all providers."""
        info = registry.resolve("glm-5.1")
        assert info is not None
        assert info.provider == "zhipu"
        assert info.model == "glm-5.1"

    def test_resolve_fuzzy_kimi(self, registry):
        info = registry.resolve("kimi-k2.6")
        assert info is not None
        assert info.provider == "kimi"

    def test_resolve_fuzzy_deepseek(self, registry):
        info = registry.resolve("deepseek-chat")
        assert info is not None
        assert info.provider == "deepseek"

    def test_resolve_unknown_model(self, registry):
        info = registry.resolve("nonexistent-model")
        assert info is None

    def test_resolve_unknown_provider(self, registry):
        info = registry.resolve("unknown/glm-5.1")
        assert info is None

    def test_list_models(self, registry):
        models = registry.list_models()
        assert len(models) > 0
        # Should have models from all providers
        providers = {m["provider"] for m in models}
        assert "kimi" in providers
        assert "zhipu" in providers
        assert "deepseek" in providers
        assert "openai" in providers

    def test_list_available(self, registry):
        available = registry.list_available()
        # All should have API keys since we set env vars
        assert len(available) > 0
        for m in available:
            assert m["api_key_set"]

    def test_list_available_filters_missing_key(self):
        """Without API keys set, models should not be available."""
        reg = ModelRegistry()  # No env vars set
        available = reg.list_available()
        # Only OPENAI_API_KEY might be set from other tests, check that
        # at least models without keys are filtered out
        for m in available:
            assert m["api_key_set"]

    def test_no_api_key_returns_none(self):
        """resolve() should still return ModelInfo even without API key,
        but api_key will be empty."""
        reg = ModelRegistry()
        info = reg.resolve("kimi/kimi-k2.6")
        # Info is returned but api_key is empty
        assert info is not None
        assert info.api_key == ""


# ── User Config Override ────────────────────────────────────────────

class TestUserConfig:
    def test_user_models_json(self, mock_env, tmp_path):
        """Test loading user models.json override."""
        models_file = tmp_path / ".mini-pi" / "models.json"
        models_file.parent.mkdir(parents=True)

        custom_config = {
            "providers": {
                "kimi": {
                    "models": {
                        "kimi-k2-custom": {
                            "max_context_tokens": 200000,
                            "thinking": True,
                        }
                    }
                },
                "custom-provider": {
                    "api_key_env": "CUSTOM_API_KEY",
                    "base_url": "https://custom.api/v1",
                    "models": {
                        "custom-model-v1": {
                            "max_context_tokens": 64000
                        }
                    }
                }
            },
            "default": "kimi/kimi-k2-custom"
        }
        models_file.write_text(json.dumps(custom_config))

        # Temporarily change cwd
        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            reg = ModelRegistry()
            # Custom model should be available
            info = reg.resolve("kimi/kimi-k2-custom")
            assert info is not None
            assert info.max_context_tokens == 200000

            # Custom provider should be available
            info2 = reg.resolve("custom-provider/custom-model-v1")
            assert info2 is not None
            assert info2.base_url == "https://custom.api/v1"

            # Default should be set
            assert reg.get_default_spec() == "kimi/kimi-k2-custom"
        finally:
            os.chdir(old_cwd)


# ── ModelInfo ───────────────────────────────────────────────────────

class TestModelInfo:
    def test_model_info_defaults(self):
        info = ModelInfo(
            provider="test",
            model="test-model",
            api_key="sk-test",
            base_url="https://api.test.com/v1",
        )
        assert info.max_context_tokens == 128000
        assert info.thinking is False
        assert info.thinking_type is None
        assert info.temperature is None


# ── Extra Kwargs ────────────────────────────────────────────────────

class TestExtraKwargs:
    def test_default_kwargs(self):
        info = ModelInfo(
            provider="openai",
            model="gpt-4o",
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
        )
        kwargs = get_model_extra_kwargs(info)
        assert kwargs["temperature"] == 0
        assert "extra_body" not in kwargs

    def test_zhipu_thinking_kwargs(self):
        info = ModelInfo(
            provider="zhipu",
            model="glm-5.1",
            api_key="sk-test",
            base_url="https://open.bigmodel.cn/api/paas/v4/",
            thinking=True,
            thinking_type="enabled",
            temperature=1.0,
        )
        kwargs = get_model_extra_kwargs(info)
        assert kwargs["temperature"] == 1.0
        assert kwargs["extra_body"]["thinking"] == {"type": "enabled"}

    def test_kimi_thinking_kwargs(self):
        info = ModelInfo(
            provider="kimi",
            model="kimi-k2.6",
            api_key="sk-test",
            base_url="https://api.moonshot.cn/v1",
            thinking=True,
        )
        kwargs = get_model_extra_kwargs(info)
        assert kwargs["extra_body"]["thinking"] == {"type": "enabled"}

    def test_deepseek_no_thinking_kwargs(self):
        info = ModelInfo(
            provider="deepseek",
            model="deepseek-chat",
            api_key="sk-test",
            base_url="https://api.deepseek.com",
            thinking=False,
        )
        kwargs = get_model_extra_kwargs(info)
        assert kwargs["temperature"] == 0
        assert "extra_body" not in kwargs


# ── Config Integration ──────────────────────────────────────────────

class TestConfigIntegration:
    def test_config_resolve_from_registry(self, mock_env):
        config = Config(model="glm-5.1")
        info = config.resolve_model()
        assert info is not None
        assert info.provider == "zhipu"
        assert info.model == "glm-5.1"
        assert info.api_key == "sk-zhipu-test"

    def test_config_resolve_with_slash(self, mock_env):
        config = Config(model="kimi/kimi-k2.6")
        info = config.resolve_model()
        assert info is not None
        assert info.provider == "kimi"

    def test_config_fallback_legacy(self, monkeypatch):
        """When model not in registry, fall back to env-based config."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-legacy")
        config = Config(model="some-unknown-model")
        info = config.resolve_model()
        assert info is not None
        assert info.provider == "custom"
        assert info.api_key == "sk-legacy"

    def test_config_get_extra_kwargs(self, mock_env):
        config = Config(model="zhipu/glm-5.1")
        config.resolve_model()
        kwargs = config.get_extra_kwargs()
        assert kwargs["temperature"] == 1.0
        assert kwargs["extra_body"]["thinking"] == {"type": "enabled"}

    def test_config_validate_with_registry(self, mock_env):
        config = Config(model="glm-5.1")
        issues = config.validate()
        assert len(issues) == 0

    def test_config_validate_no_key(self):
        """When no API key is set at all, validation should fail."""
        config = Config(model="some-random-model", api_key="")
        issues = config.validate()
        assert len(issues) > 0


# ── create_llm ──────────────────────────────────────────────────────

class TestCreateLLM:
    def test_creates_openai_llm(self, mock_env):
        info = ModelInfo(
            provider="zhipu",
            model="glm-5.1",
            api_key="sk-test",
            base_url="https://open.bigmodel.cn/api/paas/v4/",
        )
        llm = create_llm(info)
        assert isinstance(llm, OpenAILLM)
        assert llm.model == "glm-5.1"


# Need this import for the isinstance check
from mini_pi.llm.openai_llm import OpenAILLM
