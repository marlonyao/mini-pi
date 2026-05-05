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
        estimator = TokenEstimator(max_context_tokens=10000)
        messages = [{"role": "user", "content": "short"}]
        assert estimator.should_compact(messages, threshold=0.8) is False

    def test_should_compact_above_threshold(self):
        estimator = TokenEstimator(max_context_tokens=1000)
        # Create a very long message
        messages = [{"role": "user", "content": "a" * 5000}]
        assert estimator.should_compact(messages, threshold=0.5) is True
