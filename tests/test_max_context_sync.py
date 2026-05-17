"""
Tests for syncing max_context_tokens between ModelInfo and CompactionConfig/TokenEstimator.

Bug: CompactionConfig.max_context_tokens defaults to 128000 and never syncs
with the actual model's context window. This means:
1. TokenEstimator starts with wrong value for non-128K models
2. Legacy fallback in Config.resolve_model() always uses 128K
3. Only /model switch triggers correction, initial startup does not
"""

import pytest

from mini_pi.compactor import CompactionConfig
from mini_pi.config import Config
from mini_pi.models import ModelRegistry, ModelInfo
from mini_pi.token_estimator import TokenEstimator


@pytest.fixture
def mock_env(monkeypatch):
    """Set up mock API keys for testing."""
    monkeypatch.setenv("MOONSHOT_API_KEY", "sk-moonshot-test")
    monkeypatch.setenv("KIMI_API_KEY", "sk-kimi-test")
    monkeypatch.setenv("ZHIPU_API_KEY", "sk-zhipu-test")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")


# ── CompactionConfig + ModelInfo sync ──────────────────────────────

class TestCompactionConfigSync:
    """CompactionConfig.max_context_tokens should sync with resolved ModelInfo."""

    def test_config_resolve_model_syncs_max_context(self, mock_env):
        """When model is resolved from registry, CompactionConfig should sync."""
        config = Config(model="zhipu/glm-5.1")
        # Before resolve, default is 128000
        assert config.compaction.max_context_tokens == 128000

        info = config.resolve_model()
        assert info is not None
        assert info.max_context_tokens == 200000  # glm-5.1 has 200K

        # After resolve, CompactionConfig should be in sync
        assert config.compaction.max_context_tokens == 200000

    def test_config_resolve_model_syncs_deepseek(self, mock_env):
        """128K models should also sync correctly (128000 remains 128000)."""
        config = Config(model="deepseek/deepseek-chat")
        info = config.resolve_model()
        assert info is not None
        assert info.max_context_tokens == 128000
        assert config.compaction.max_context_tokens == 128000

    def test_config_resolve_model_syncs_kimi_coding(self, mock_env):
        """Models with non-standard context windows sync correctly."""
        config = Config(model="kimi-coding/kimi-for-coding")
        info = config.resolve_model()
        assert info is not None
        assert info.max_context_tokens == 262144  # 256K
        assert config.compaction.max_context_tokens == 262144

    def test_config_legacy_fallback_uses_compaction_config(self, monkeypatch):
        """When model not in registry, fall back to CompactionConfig value."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-legacy")
        # Set a custom CompactionConfig
        config = Config(
            model="some-unknown-model",
            compaction=CompactionConfig(max_context_tokens=32768),
        )
        info = config.resolve_model()
        assert info is not None
        assert info.provider == "custom"
        # Legacy fallback should use the (possibly user-set) CompactionConfig value
        assert info.max_context_tokens == 32768

    def test_config_resolve_without_registry_does_not_crash(self):
        """When no model can be resolved (no API key, no registry match),
        resolve_model returns None gracefully and CompactionConfig is untouched."""
        config = Config(model="nonexistent", api_key="")
        info = config.resolve_model()
        assert info is None
        # CompactionConfig keeps its default
        assert config.compaction.max_context_tokens == 128000


# ── TokenEstimator initialization ───────────────────────────────────

class TestTokenEstimatorInit:
    """TokenEstimator should be initialized with the actual model's context window."""

    def test_estimator_uses_model_info_not_fixed_default(self):
        """TokenEstimator max_context_tokens must reflect actual model, not 128K default."""
        # Simulate what Agent does: read from CompactionConfig
        cc = CompactionConfig(max_context_tokens=200000)
        estimator = TokenEstimator(
            max_context_tokens=cc.max_context_tokens,
            reserve_tokens=cc.reserve_tokens,
        )
        assert estimator.max_context_tokens == 200000

    def test_estimator_threshold_respects_synced_value(self):
        """Pi-aligned threshold should use the synced context window."""
        # A 200K model with 16K reserve → threshold = 184000
        estimator = TokenEstimator(max_context_tokens=200000, reserve_tokens=16384)
        # Short message (~6 tokens) should NOT trigger
        messages = [{"role": "user", "content": "short"}]
        assert estimator.should_compact(messages) is False

        # Large message exceeding 184K should trigger
        large_text = "x" * (200000 * 4)  # ~200K tokens in char estimate
        messages = [{"role": "user", "content": large_text}]
        assert estimator.should_compact(messages) is True

    def test_estimator_default_is_not_model_aware(self):
        """Without explicit configuration, TokenEstimator defaults to 128K.
        This test documents the current behavior — the fix is that CompactionConfig
        gets synced before TokenEstimator is created."""
        estimator = TokenEstimator()
        assert estimator.max_context_tokens == 128000


# ── Config resolves then supplies to TokenEstimator ─────────────────

class TestConfigToEstimatorFlow:
    """End-to-end: Config → resolve model → create TokenEstimator with correct context."""

    def test_full_flow_200k_model(self, mock_env):
        """Full flow: resolve a 200K model, create estimator, verify threshold."""
        config = Config(model="zhipu/glm-5.1")
        info = config.resolve_model()
        assert info is not None

        # After resolution, CompactionConfig is synced
        assert config.compaction.max_context_tokens == 200000

        # Create TokenEstimator (as Agent does)
        estimator = TokenEstimator(
            max_context_tokens=config.compaction.max_context_tokens,
            reserve_tokens=config.compaction.reserve_tokens,
        )
        assert estimator.max_context_tokens == 200000

        # Pi threshold: 200000 - 16384 = 183616
        threshold = estimator.max_context_tokens - estimator.reserve_tokens
        assert threshold == 200000 - 16384

    def test_full_flow_128k_model(self, mock_env):
        """Full flow: resolve a 128K model."""
        config = Config(model="deepseek/deepseek-chat")
        info = config.resolve_model()
        assert info is not None

        assert config.compaction.max_context_tokens == 128000

        estimator = TokenEstimator(
            max_context_tokens=config.compaction.max_context_tokens,
            reserve_tokens=config.compaction.reserve_tokens,
        )
        assert estimator.max_context_tokens == 128000

    def test_full_flow_256k_model(self, mock_env):
        """Full flow: resolve a 256K model."""
        config = Config(model="kimi-coding/kimi-for-coding")
        info = config.resolve_model()
        assert info is not None

        assert config.compaction.max_context_tokens == 262144

        estimator = TokenEstimator(
            max_context_tokens=config.compaction.max_context_tokens,
            reserve_tokens=config.compaction.reserve_tokens,
        )
        assert estimator.max_context_tokens == 262144
