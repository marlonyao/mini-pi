"""
Tests for compaction module.

Compaction summarizes older conversation history when context approaches
the model's limit. Key test areas:
1. Split messages at clean turn boundaries
2. NEVER break tool call + tool result pairs
3. Recent tool outputs are preserved intact
4. Incremental summary updates work
5. Full history is preserved on disk
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from mini_pi.compactor import (
    CompactionConfig,
    CompactResult,
    Compactor,
    build_compaction_prompt,
    build_incremental_prompt,
    _format_messages_for_summary,
)
from mini_pi.session import Session


# ── Helpers ─────────────────────────────────────────────────────────

def make_tool_call(call_id: str, name: str = "bash") -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": '{"command":"ls"}'},
    }


def build_conversation(turns: int = 5, tool_result_size: int = 200) -> list[dict]:
    """Build a conversation with many turns for compaction testing.

    Each turn: user → assistant(tool_calls) → tool(result) → assistant(text)
    """
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


def build_multi_tool_turn(
    turn_id: str,
    tool_names: list[str],
    result_size: int = 200,
) -> list[dict]:
    """Build one turn with multiple tool calls in a single assistant message."""
    msgs = [{"role": "user", "content": f"Turn {turn_id}"}]
    tool_calls = []
    for j, name in enumerate(tool_names):
        tool_calls.append(make_tool_call(f"{turn_id}_{j}", name))
    msgs.append({"role": "assistant", "content": None, "tool_calls": tool_calls})
    for j, name in enumerate(tool_names):
        msgs.append({
            "role": "tool",
            "tool_call_id": f"{turn_id}_{j}",
            "content": f"{name} output: " + "y" * result_size,
        })
    msgs.append({"role": "assistant", "content": f"Done with {turn_id}"})
    return msgs


# ── Config tests ────────────────────────────────────────────────────

class TestCompactionConfig:
    def test_defaults(self):
        config = CompactionConfig()
        assert config.enabled is True
        assert config.keep_recent_messages == 8
        assert config.threshold == 0.8
        assert config.max_context_tokens == 128000

    def test_custom_model(self):
        config = CompactionConfig(model="deepseek-chat")
        assert config.model == "deepseek-chat"


# ── Prompt building tests ──────────────────────────────────────────

class TestFormatMessages:
    def test_truncates_tool_outputs(self):
        messages = [
            {"role": "tool", "content": "x" * 5000},
        ]
        text = _format_messages_for_summary(messages, max_chars=100)
        assert len(text) < 500  # Should be truncated
        assert "truncated" in text

    def test_keeps_short_outputs_intact(self):
        messages = [
            {"role": "tool", "content": "short output"},
            {"role": "user", "content": "hello"},
        ]
        text = _format_messages_for_summary(messages, max_chars=500)
        assert "short output" in text
        assert "hello" in text

    def test_includes_tool_call_info(self):
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [make_tool_call("tc1", "read")],
            }
        ]
        text = _format_messages_for_summary(messages)
        assert "Tool call: read" in text


class TestBuildCompactionPrompt:
    def test_full_prompt(self):
        messages = build_conversation(3)
        prompt = build_compaction_prompt(messages)
        assert "summar" in prompt.lower()
        assert "conversation history" in prompt.lower()

    def test_custom_instructions(self):
        messages = build_conversation(2)
        prompt = build_compaction_prompt(messages, instructions="Focus on files")
        assert "Focus on files" in prompt

    def test_empty_messages(self):
        prompt = build_compaction_prompt([])
        assert len(prompt) > 0


class TestBuildIncrementalPrompt:
    def test_includes_existing_summary(self):
        messages = build_conversation(1)
        prompt = build_incremental_prompt("Old summary here", messages)
        assert "Old summary here" in prompt
        assert "update" in prompt.lower() or "merge" in prompt.lower()

    def test_includes_new_messages(self):
        messages = [{"role": "user", "content": "new task"}]
        prompt = build_incremental_prompt("Previous summary", messages)
        assert "new task" in prompt


# ── CompactResult tests ────────────────────────────────────────────

class TestCompactResult:
    def test_success_result(self):
        result = CompactResult(
            success=True,
            summary="## Summary\n\nCompleted tasks 0-5",
            original_count=20,
            compacted_count=5,
        )
        assert result.success is True

    def test_success_get_messages(self):
        recent = [
            {"role": "user", "content": "latest"},
            {"role": "assistant", "content": "reply"},
        ]
        result = CompactResult(
            success=True,
            summary="Summary text",
            original_count=10,
            compacted_count=3,
            _recent_messages=recent,
        )
        msgs = result.get_messages()
        assert len(msgs) == 3  # 1 summary + 2 recent
        assert msgs[0]["role"] == "system"
        assert "Summary text" in msgs[0]["content"]

    def test_failure_get_messages(self):
        result = CompactResult(success=False, error="fail")
        assert result.get_messages() == []


# ── Split tests (CRITICAL: tool pair integrity) ────────────────────

class TestSplitMessages:
    def test_basic_split(self):
        config = CompactionConfig(keep_recent_messages=4)
        compactor = Compactor(config)
        messages = build_conversation(5)  # 20 messages

        old, recent = compactor._split_messages(messages)
        assert len(recent) >= 4
        assert len(old) + len(recent) == len(messages)

    def test_never_breaks_tool_call_pairs(self):
        """CRITICAL: tool result must never be separated from its tool call."""
        config = CompactionConfig(keep_recent_messages=3)
        compactor = Compactor(config)

        # Build: user → assistant(tool_call) → tool(result) → assistant(text)
        messages = build_conversation(5)

        old, recent = compactor._split_messages(messages)

        # Check: recent section must not start with a tool result
        if recent:
            assert recent[0].get("role") != "tool", \
                "Recent section starts with orphaned tool result!"

        # Check: no assistant with tool_calls in old that has its results in recent
        old_tool_call_ids = set()
        for msg in old:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    old_tool_call_ids.add(tc["id"])

        for msg in recent:
            if msg.get("role") == "tool":
                tid = msg.get("tool_call_id")
                assert tid not in old_tool_call_ids, \
                    f"Tool result {tid} is in recent but its call is in old!"

    def test_split_with_multi_tool_calls(self):
        """Multiple tool calls in one assistant message."""
        config = CompactionConfig(keep_recent_messages=4)
        compactor = Compactor(config)

        messages = []
        messages.extend(build_multi_tool_turn("t1", ["read", "bash", "grep"]))
        messages.extend(build_multi_tool_turn("t2", ["read", "edit"]))

        old, recent = compactor._split_messages(messages)

        # Verify no tool pair is broken
        self._verify_tool_pair_integrity(old, recent)

    def test_split_preserves_recent_messages_count(self):
        config = CompactionConfig(keep_recent_messages=6)
        compactor = Compactor(config)
        messages = build_conversation(5)

        old, recent = compactor._split_messages(messages)
        # Recent should be at least keep_recent_messages
        # (may be more if adjustment was needed for tool pairs)
        assert len(recent) >= 6

    def test_short_conversation_returns_empty_old(self):
        config = CompactionConfig(keep_recent_messages=20)
        compactor = Compactor(config)
        messages = build_conversation(2)  # 8 messages

        old, recent = compactor._split_messages(messages)
        assert old == []
        assert len(recent) == 8

    def _verify_tool_pair_integrity(self, old, recent):
        """Helper: verify tool calls and results stay together."""
        old_tool_call_ids = set()
        for msg in old:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    old_tool_call_ids.add(tc["id"])

        for msg in recent:
            if msg.get("role") == "tool":
                tid = msg.get("tool_call_id")
                assert tid not in old_tool_call_ids, \
                    f"Tool pair broken: call in old, result {tid} in recent"


# ── Full compaction flow tests ─────────────────────────────────────

class TestCompactor:
    def test_compact_with_mock_llm(self):
        config = CompactionConfig(keep_recent_messages=4, model="test-model")
        compactor = Compactor(config)
        messages = build_conversation(5)

        with patch.object(compactor, "_call_llm_for_summary", return_value="## Summary\nDone tasks 0-3"):
            result = compactor.compact(messages)

        assert result.success is True
        assert "Summary" in result.summary
        assert result.original_count == 20
        assert result.compacted_count == 1 + len(result._recent_messages)

    def test_compact_disabled(self):
        config = CompactionConfig(enabled=False)
        compactor = Compactor(config)
        messages = build_conversation(5)

        result = compactor.compact(messages)
        assert result.success is False
        assert "disabled" in result.error.lower()

    def test_compact_short_conversation(self):
        config = CompactionConfig(keep_recent_messages=20)
        compactor = Compactor(config)
        messages = [{"role": "user", "content": "hi"}]

        result = compactor.compact(messages)
        assert result.success is False
        assert "too short" in result.error.lower()

    def test_compact_preserves_recent_tool_outputs(self):
        """Recent tool results must be kept INTACT in the result."""
        config = CompactionConfig(keep_recent_messages=6)
        compactor = Compactor(config)

        long_output = "IMPORTANT_DATA_" * 100  # 1500 chars
        messages = build_conversation(3)
        # Make the last tool result have important content
        for msg in messages:
            if msg.get("role") == "tool":
                msg["content"] = long_output

        with patch.object(compactor, "_call_llm_for_summary", return_value="Summary"):
            result = compactor.compact(messages)

        assert result.success is True
        # Check recent messages have full content
        for msg in (result._recent_messages or []):
            if msg.get("role") == "tool":
                assert "IMPORTANT_DATA_" in msg["content"]
                assert msg["content"] == long_output  # NOT truncated!

    def test_incremental_compaction(self):
        """Second compaction should use existing summary."""
        config = CompactionConfig(keep_recent_messages=4)
        compactor = Compactor(config)
        messages = build_conversation(5)

        # First compaction
        with patch.object(compactor, "_call_llm_for_summary", return_value="First summary"):
            result1 = compactor.compact(messages)

        assert result1.success

        # Second compaction with existing summary
        with patch.object(compactor, "_call_llm_for_summary", return_value="Updated summary"):
            result2 = compactor.compact(messages, existing_summary="First summary")

        assert result2.success
        assert "Updated" in result2.summary

    def test_compact_with_custom_instructions(self):
        config = CompactionConfig(keep_recent_messages=4)
        compactor = Compactor(config)
        messages = build_conversation(5)

        with patch.object(compactor, "_call_llm_for_summary", return_value="Focused summary") as mock:
            result = compactor.compact(messages, instructions="Focus on API")

        assert result.success is True


# ── Session integration tests ──────────────────────────────────────

class TestSessionCompaction:
    def test_session_records_compaction(self, tmp_path):
        session = Session(tmp_path / "test.jsonl")
        messages = build_conversation(3)
        session.messages = list(messages)

        config = CompactionConfig(keep_recent_messages=4)
        compactor = Compactor(config)

        with patch.object(compactor, "_call_llm_for_summary", return_value="Summary text"):
            result = compactor.compact(session.messages)

        session.record_compaction(result)
        session.save()

        # Reload
        content = (tmp_path / "test.jsonl").read_text()
        assert "compaction" in content

    def test_session_tracks_summary_for_incremental(self, tmp_path):
        session = Session(tmp_path / "test.jsonl")
        messages = build_conversation(3)
        session.messages = list(messages)

        config = CompactionConfig(keep_recent_messages=4)
        compactor = Compactor(config)

        with patch.object(compactor, "_call_llm_for_summary", return_value="First summary"):
            result = compactor.compact(session.messages)

        session.record_compaction(result)

        # Session should store the summary for incremental updates
        assert hasattr(session, "_last_compaction_summary")
        assert session._last_compaction_summary == "First summary"
