"""
Tests for token estimation.

Token counting is needed to detect when context is approaching the model's limit.
We support two modes:
1. tiktoken (accurate, needs the library installed)
2. Character-based heuristic (approximate, always available)
"""

import pytest

from mini_pi.token_estimator import estimate_tokens, TokenEstimator


class TestCharacterEstimate:
    """Simple character-based token estimation."""

    def test_empty_string(self):
        assert estimate_tokens("") == 0

    def test_short_english(self):
        # English: ~4 chars per token
        tokens = estimate_tokens("Hello world")
        # Should be roughly 2-4 tokens
        assert 1 <= tokens <= 5

    def test_long_english(self):
        text = "The quick brown fox jumps over the lazy dog. " * 100
        tokens = estimate_tokens(text)
        # Should be roughly 100-200 tokens
        assert tokens > 50

    def test_chinese_text(self):
        # Chinese: ~1.5-2 chars per token
        tokens = estimate_tokens("你好世界，这是一个测试")
        assert 3 <= tokens <= 15

    def test_mixed_text(self):
        tokens = estimate_tokens("Hello 你好 world 世界")
        assert 2 <= tokens <= 10

    def test_none_input(self):
        assert estimate_tokens(None) == 0


class TestTokenEstimator:
    """TokenEstimator class with configurable strategy."""

    def test_default_strategy(self):
        estimator = TokenEstimator()
        # Should use character-based by default
        tokens = estimator.estimate("Hello world")
        assert tokens > 0

    def test_estimate_messages(self):
        """Should estimate total tokens for a list of messages."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        estimator = TokenEstimator()
        total = estimator.estimate_messages(messages)
        assert total > 0

    def test_estimate_messages_empty(self):
        estimator = TokenEstimator()
        assert estimator.estimate_messages([]) == 0

    def test_estimate_messages_with_tool_calls(self):
        """Messages with tool_calls and tool results should be counted."""
        messages = [
            {"role": "user", "content": "run this"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "bash", "arguments": '{"command":"ls"}'}},
            ]},
            {"role": "tool", "tool_call_id": "c1", "content": "file1.py\nfile2.py"},
        ]
        estimator = TokenEstimator()
        total = estimator.estimate_messages(messages)
        assert total > 0

    def test_estimate_messages_with_none_content(self):
        """Messages with None content should not crash."""
        messages = [
            {"role": "assistant", "content": None, "tool_calls": []},
        ]
        estimator = TokenEstimator()
        total = estimator.estimate_messages(messages)
        assert total >= 0


class TestEstimateWithContextLimit:
    """Test usage ratio calculation."""

    def test_usage_ratio(self):
        estimator = TokenEstimator(max_context_tokens=10000)
        messages = [
            {"role": "user", "content": "Hello world"},  # ~3 tokens
        ]
        ratio = estimator.usage_ratio(messages)
        assert 0.0 <= ratio <= 1.0

    def test_should_compact_below_threshold(self):
        """Pi-aligned: triggers when tokens > max - reserve."""
        estimator = TokenEstimator(max_context_tokens=100000, reserve_tokens=16384)
        messages = [{"role": "user", "content": "short"}]
        # 100000 - 16384 = 83616 threshold
        # ~6 tokens, way below
        assert estimator.should_compact(messages) is False

    def test_should_compact_above_threshold(self):
        """Pi-aligned: triggers when tokens > max - reserve."""
        estimator = TokenEstimator(max_context_tokens=10000, reserve_tokens=1000)
        # Create messages that exceed 10000 - 1000 = 9000 tokens
        messages = [{"role": "user", "content": "a" * 40000}]
        assert estimator.should_compact(messages) is True

    def test_should_compact_exactly_at_threshold(self):
        estimator = TokenEstimator(max_context_tokens=10000, reserve_tokens=16384)
        messages = [{"role": "user", "content": "a" * 40000}]
        # 10000 - 16384 < 0, always False since tokens can't be negative
        # Actually tokens > negative number is always True for positive tokens
        # This test verifies the behavior is sensible
        result = estimator.should_compact(messages)
        # If reserve > max, threshold is negative, any message triggers
        assert result is True

    def test_should_compact_no_reserve(self):
        estimator = TokenEstimator(max_context_tokens=100, reserve_tokens=0)
        messages = [{"role": "user", "content": "a" * 500}]
        assert estimator.should_compact(messages) is True

    def test_default_reserve_tokens(self):
        estimator = TokenEstimator()
        assert estimator.reserve_tokens == 16384
