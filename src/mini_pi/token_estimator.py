"""
Token estimation for mini-pi.

Provides token counting to detect when context approaches the model's limit.
The trigger condition is Pi-aligned:

    contextTokens > contextWindow - reserveTokens

This is an absolute value check, not a ratio. It ensures there's always
enough room for the LLM's response.
"""

from __future__ import annotations

from typing import Any


def estimate_tokens(text: str | None) -> int:
    """
    Estimate the number of tokens in a text string.

    Uses a simple character-based heuristic:
    - English/mixed: ~4 characters per token
    - CJK characters: ~2 characters per token

    This is intentionally simple — good enough for threshold checks.
    For exact counting, install tiktoken and use TokenEstimator with strategy="tiktoken".
    """
    if not text:
        return 0

    # Count CJK characters
    cjk_count = sum(1 for c in text if _is_cjk(c))
    non_cjk_count = len(text) - cjk_count

    # CJK: ~2 chars/token, others: ~4 chars/token
    cjk_tokens = cjk_count / 2
    other_tokens = non_cjk_count / 4

    return max(1, int(cjk_tokens + other_tokens))


def _is_cjk(char: str) -> bool:
    """Check if a character is CJK (Chinese, Japanese, Korean)."""
    cp = ord(char)
    return (
        (0x4E00 <= cp <= 0x9FFF) or    # CJK Unified Ideographs
        (0x3400 <= cp <= 0x4DBF) or    # CJK Extension A
        (0x20000 <= cp <= 0x2A6DF) or  # CJK Extension B
        (0x3000 <= cp <= 0x303F) or    # CJK Symbols and Punctuation
        (0x3040 <= cp <= 0x309F) or    # Hiragana
        (0x30A0 <= cp <= 0x30FF) or    # Katakana
        (0xAC00 <= cp <= 0xD7AF)       # Hangul Syllables
    )


def _extract_message_texts(msg: dict[str, Any]) -> list[str]:
    """Extract all text content from a message dict.

    Shared with Compactor to ensure identical token estimation.
    """
    texts: list[str] = []

    content = msg.get("content")
    if content:
        texts.append(content)

    # Tool calls contain function name + arguments
    tool_calls = msg.get("tool_calls")
    if tool_calls:
        for tc in tool_calls:
            fn = tc.get("function", {})
            if fn.get("name"):
                texts.append(fn["name"])
            if fn.get("arguments"):
                texts.append(fn["arguments"])

    return texts


class TokenEstimator:
    """
    Estimates token usage for messages against a context window.

    Pi-aligned trigger:
        should_compact() returns True when:
            estimated_tokens > max_context_tokens - reserve_tokens

    Usage:
        estimator = TokenEstimator(max_context_tokens=128000, reserve_tokens=16384)
        if estimator.should_compact(messages):
            # trigger compaction
    """

    def __init__(
        self,
        max_context_tokens: int = 128000,
        reserve_tokens: int = 16384,
        strategy: str = "char",
    ):
        self.max_context_tokens = max_context_tokens
        self.reserve_tokens = reserve_tokens
        self.strategy = strategy

    def estimate(self, text: str | None) -> int:
        """Estimate tokens for a single string."""
        return estimate_tokens(text)

    def estimate_messages(self, messages: list[dict[str, Any]]) -> int:
        """Estimate total tokens for a list of messages."""
        total = 0
        for msg in messages:
            # Each message has ~4 tokens overhead (role, formatting)
            total += 4
            for text in _extract_message_texts(msg):
                total += estimate_tokens(text)
        return total

    def usage_ratio(self, messages: list[dict[str, Any]]) -> float:
        """Get the current context usage ratio (0.0 to 1.0+)."""
        if self.max_context_tokens <= 0:
            return 0.0
        return self.estimate_messages(messages) / self.max_context_tokens

    def should_compact(self, messages: list[dict[str, Any]]) -> bool:
        """Check if context exceeds the compaction threshold.

        Pi-aligned: triggers when estimated tokens exceed the context
        window minus the reserve. This is an absolute value check,
        not a ratio.
        """
        threshold = self.max_context_tokens - self.reserve_tokens
        return self.estimate_messages(messages) > threshold
