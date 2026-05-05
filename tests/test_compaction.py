"""
Tests for compaction module.

Compaction summarizes older conversation history when context approaches
the model's limit. It should:
1. Split messages into "old" and "recent"
2. Send old messages to LLM for summarization
3. Replace old messages with a compaction summary entry
4. Preserve full history on disk (compaction is semantic, not destructive)
5. Support manual trigger with custom instructions
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from mini_pi.compactor import (
    CompactionConfig,
    CompactResult,
    Compactor,
    build_compaction_prompt,
)
from mini_pi.session import Session


# ── Fixtures ────────────────────────────────────────────────────────

def make_tool_call(call_id: str, name: str = "bash") -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": '{"command":"ls"}'},
    }


def build_long_conversation(turns: int = 10, tool_result_size: int = 200) -> list[dict]:
    """Build a conversation with many turns for compaction testing."""
    messages = []
    for i in range(turns):
        messages.append({"role": "user", "content": f"Task {i}: please do something"})
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [make_tool_call(f"call_{i}")],
        })
        messages.append({
            "role": "tool",
            "tool_call_id": f"call_{i}",
            "content": "x" * tool_result_size,
        })
        messages.append({"role": "assistant", "content": f"Done with task {i}"})
    return messages


# ── Tests ───────────────────────────────────────────────────────────

class TestCompactionConfig:
    def test_defaults(self):
        config = CompactionConfig()
        assert config.enabled is True
        assert config.keep_recent_messages == 6
        assert config.threshold == 0.8
        assert config.max_context_tokens == 128000
        assert config.model is None  # None = use session model

    def test_custom_model(self):
        config = CompactionConfig(model="gpt-4o-mini")
        assert config.model == "gpt-4o-mini"


class TestBuildCompactionPrompt:
    def test_prompt_includes_key_instructions(self):
        messages = build_long_conversation(3)
        prompt = build_compaction_prompt(messages)

        # Should instruct the model to summarize
        assert "summar" in prompt.lower()
        # Should include conversation history marker
        assert "conversation history" in prompt.lower()
        # Should mention markdown output
        assert "markdown" in prompt.lower()

    def test_prompt_includes_custom_instructions(self):
        messages = build_long_conversation(2)
        prompt = build_compaction_prompt(messages, instructions="Focus on API design")
        assert "Focus on API design" in prompt

    def test_empty_messages_prompt(self):
        prompt = build_compaction_prompt([])
        # Should still produce a valid prompt
        assert len(prompt) > 0


class TestCompactResult:
    def test_success_result(self):
        result = CompactResult(
            success=True,
            summary="## Summary\n\nCompleted tasks 0-5",
            original_count=20,
            compacted_count=2,  # summary + meta
        )
        assert result.success is True
        assert "Summary" in result.summary

    def test_failure_result(self):
        result = CompactResult(success=False, error="LLM call failed")
        assert result.success is False
        assert result.error == "LLM call failed"


class TestCompactor:
    def test_compact_splits_old_and_recent(self):
        """Compactor should split messages into old (to summarize) and recent (to keep)."""
        config = CompactionConfig(keep_recent_messages=4)
        messages = build_long_conversation(5)  # 20 messages total
        compactor = Compactor(config)

        old, recent = compactor._split_messages(messages)
        # Recent should be last 4 messages
        assert len(recent) == 4
        # Old should be the rest
        assert len(old) == len(messages) - 4

    def test_compact_split_preserves_tool_pairs(self):
        """Split should not break tool call + tool result pairs."""
        config = CompactionConfig(keep_recent_messages=4)
        messages = build_long_conversation(5)
        compactor = Compactor(config)

        old, recent = compactor._split_messages(messages)

        # Check that the recent section starts at a sensible boundary
        # (not in the middle of a tool call sequence)
        first_recent = recent[0]
        # Should not start with a tool result (would be orphaned)
        assert first_recent.get("role") != "tool"

    def test_compact_with_mock_llm(self):
        """Test full compaction flow with mocked LLM."""
        config = CompactionConfig(
            keep_recent_messages=4,
            model="gpt-4o-mini",
        )
        compactor = Compactor(config)

        messages = build_long_conversation(5)

        # Mock the LLM call
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "## Summary\n\nCompleted tasks 0-4. Modified files: main.py, utils.py"

        with patch.object(compactor, "_call_llm_for_summary", return_value="## Summary\n\nCompleted tasks 0-4"):
            result = compactor.compact(messages)

        assert result.success is True
        assert "Summary" in result.summary
        assert result.original_count == 20
        # Compacted messages = 1 (summary) + 4 (recent) = 5
        assert result.compacted_count == 5

    def test_compact_disabled(self):
        """When disabled, compact should return the original messages."""
        config = CompactionConfig(enabled=False)
        compactor = Compactor(config)

        messages = build_long_conversation(5)
        result = compactor.compact(messages)

        assert result.success is False
        assert "disabled" in result.error.lower()

    def test_compact_short_conversation(self):
        """Short conversations that don't need compaction should be returned as-is."""
        config = CompactionConfig(keep_recent_messages=10)
        compactor = Compactor(config)

        # Only 4 messages — all fit in "recent"
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        result = compactor.compact(messages)

        # Nothing to compact
        assert result.success is False
        assert "nothing to compact" in result.error.lower() or "too short" in result.error.lower()

    def test_compact_produces_valid_messages(self):
        """Compacted messages should be valid OpenAI format."""
        config = CompactionConfig(keep_recent_messages=4)
        compactor = Compactor(config)

        messages = build_long_conversation(5)

        with patch.object(compactor, "_call_llm_for_summary", return_value="Summary of old messages"):
            result = compactor.compact(messages)

        assert result.success is True
        # The result should include a system message with the summary
        compacted_messages = result.get_messages()
        assert len(compacted_messages) < len(messages)
        # First message should be the compaction summary
        assert compacted_messages[0]["role"] == "user"
        assert "Summary of old messages" in compacted_messages[0]["content"]

    def test_compact_with_custom_instructions(self):
        """Custom instructions should be passed to the LLM."""
        config = CompactionConfig(keep_recent_messages=4)
        compactor = Compactor(config)

        messages = build_long_conversation(5)

        with patch.object(compactor, "_call_llm_for_summary", return_value="Focused summary") as mock_llm:
            result = compactor.compact(messages, instructions="Focus on API decisions")

        assert result.success is True
        # Verify instructions were passed
        call_args = mock_llm.call_args
        assert "Focus on API decisions" in call_args[0][0] or "Focus on API decisions" in str(call_args)


class TestSessionCompaction:
    """Test compaction integration with Session."""

    def test_session_records_compaction(self, tmp_path):
        """Session should record compaction events in the JSONL."""
        session = Session(tmp_path / "test.jsonl")
        messages = build_long_conversation(3)
        for msg in messages:
            session.messages.append(msg)

        config = CompactionConfig(keep_recent_messages=4)
        compactor = Compactor(config)

        with patch.object(compactor, "_call_llm_for_summary", return_value="Summary"):
            result = compactor.compact(session.messages)

        # Session should be able to record the compaction
        session.record_compaction(result)
        session.save()

        # Reload and verify
        session2 = Session(tmp_path / "test.jsonl")
        # The JSONL should contain a compaction entry
        content = (tmp_path / "test.jsonl").read_text()
        assert "compaction" in content
