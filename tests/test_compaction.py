"""
Tests for compaction module (Pi-aligned strategy).

Covers:
1. CompactionConfig new defaults
2. Token-based split (walk backwards with budget)
3. Turn boundary enforcement
4. Tool pair integrity (never break assistant+tool_result)
5. Split turn detection and handling
6. Structured summary format
7. File tracking (cumulative)
8. Message serialization
9. Compactor.full flow with mock LLM
10. Session integration with CompactionEntry
"""

import json
from unittest.mock import patch

import pytest

from mini_pi.compactor import (
    COMPACTION_SYSTEM_PROMPT,
    CompactionConfig,
    CompactResult,
    Compactor,
    append_file_tags,
    build_compaction_prompt,
    estimate_message_tokens,
    estimate_messages_tokens,
    extract_file_ops,
    serialize_conversation,
    serialize_message,
)
from mini_pi.session import Session


# ── Helpers ─────────────────────────────────────────────────────────


def make_tool_call(call_id: str, name: str = "bash", args: str = '{"command":"ls"}') -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": args},
    }


def build_conversation(turns: int = 5, tool_result_size: int = 200) -> list[dict]:
    """Build a conversation with many turns.

    Each turn: user → assistant(tool_calls) → tool(result) → assistant(text)
    """
    messages = []
    for i in range(turns):
        messages.append({"role": "user", "content": f"Task {i}: please do something"})
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [make_tool_call(f"call_{i}", "bash")],
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
    tool_calls = [make_tool_call(f"{turn_id}_{j}", name) for j, name in enumerate(tool_names)]
    msgs.append({"role": "assistant", "content": None, "tool_calls": tool_calls})
    for j, name in enumerate(tool_names):
        msgs.append({
            "role": "tool",
            "tool_call_id": f"{turn_id}_{j}",
            "content": f"{name} output: " + "y" * result_size,
        })
    msgs.append({"role": "assistant", "content": f"Done with {turn_id}"})
    return msgs


def build_conversation_with_file_ops() -> list[dict]:
    """Build a conversation with read/write tool calls for file tracking tests."""
    return [
        {"role": "user", "content": "Refactor auth module"},
        {
            "role": "assistant",
            "content": "Let me read the files first",
            "tool_calls": [
                make_tool_call("c1", "read", '{"path": "src/auth.py"}'),
                make_tool_call("c2", "read", '{"path": "src/models.py"}'),
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "c1",
            "content": "# auth.py contents...",
        },
        {
            "role": "tool",
            "tool_call_id": "c2",
            "content": "# models.py contents...",
        },
        {"role": "assistant", "content": "Now I'll modify them"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                make_tool_call("c3", "edit", '{"path": "src/auth.py", "oldText": "class Auth", "newText": "class AuthService"}'),
                make_tool_call("c4", "write", '{"path": "src/auth_test.py", "content": "..."}'),
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "c3",
            "content": "✅ Edited src/auth.py",
        },
        {
            "role": "tool",
            "tool_call_id": "c4",
            "content": "✅ Wrote src/auth_test.py",
        },
        {"role": "assistant", "content": "Refactoring complete"},
    ]


# ── Config tests ────────────────────────────────────────────────────


class TestCompactionConfig:
    def test_defaults(self):
        config = CompactionConfig()
        assert config.enabled is True
        assert config.reserve_tokens == 16384
        assert config.keep_recent_tokens == 20000
        assert config.max_context_tokens == 128000
        assert config.model is None

    def test_custom_model(self):
        config = CompactionConfig(model="deepseek-chat")
        assert config.model == "deepseek-chat"

    def test_no_threshold_field(self):
        """Pi-aligned: no ratio-based threshold."""
        config = CompactionConfig()
        assert not hasattr(config, "threshold")

    def test_no_keep_recent_messages_field(self):
        """Pi-aligned: no count-based keep_recent_messages."""
        config = CompactionConfig()
        assert not hasattr(config, "keep_recent_messages")


# ── Token estimation helpers ────────────────────────────────────────


class TestEstimateMessageTokens:
    def test_plain_text(self):
        msg = {"role": "user", "content": "Hello world"}
        tokens = estimate_message_tokens(msg)
        # 4 overhead + estimate_tokens("Hello world")
        assert tokens > 4
        assert tokens < 20  # sanity

    def test_tool_calls(self):
        msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [make_tool_call("c1", "read", '{"path": "foo.py"}')],
        }
        tokens = estimate_message_tokens(msg)
        assert tokens > 4

    def test_empty_content(self):
        msg = {"role": "assistant", "content": None}
        tokens = estimate_message_tokens(msg)
        assert tokens == 4  # just overhead


class TestEstimateMessagesTokens:
    def test_multiple_messages(self):
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        total = estimate_messages_tokens(messages)
        assert total > 8  # 4 overhead per message + content

    def test_empty_list(self):
        assert estimate_messages_tokens([]) == 0


# ── Serialization tests ────────────────────────────────────────────


class TestSerializeMessage:
    def test_user_message(self):
        msg = {"role": "user", "content": "hello"}
        text = serialize_message(msg)
        assert "[User]: hello" in text

    def test_assistant_message(self):
        msg = {"role": "assistant", "content": "I'll help you"}
        text = serialize_message(msg)
        assert "[Assistant]: I'll help you" in text

    def test_tool_result_truncation(self):
        msg = {"role": "tool", "content": "x" * 5000}
        text = serialize_message(msg, max_chars=100)
        assert len(text) < 300
        assert "truncated" in text

    def test_tool_result_kept_short(self):
        msg = {"role": "tool", "content": "short output"}
        text = serialize_message(msg, max_chars=500)
        assert "short output" in text
        assert "truncated" not in text

    def test_tool_calls_serialized(self):
        msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [make_tool_call("tc1", "read", '{"path":"foo.py"}')],
        }
        text = serialize_message(msg)
        assert "[Assistant tool calls]:" in text
        assert "read" in text

    def test_empty_content(self):
        msg = {"role": "assistant", "content": None}
        text = serialize_message(msg)
        assert text == ""


class TestSerializeConversation:
    def test_full_conversation(self):
        messages = [
            {"role": "user", "content": "read foo.py"},
            {"role": "assistant", "content": None, "tool_calls": [make_tool_call("c1", "read")]},
            {"role": "tool", "tool_call_id": "c1", "content": "file contents"},
            {"role": "assistant", "content": "I see the file"},
        ]
        text = serialize_conversation(messages)
        assert "[User]" in text
        assert "[Assistant tool calls]" in text
        assert "[Tool result]" in text
        assert "[Assistant]" in text


# ── File tracking tests ────────────────────────────────────────────


class TestExtractFileOps:
    def test_read_files(self):
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    make_tool_call("c1", "read", '{"path": "src/auth.py"}'),
                    make_tool_call("c2", "read", '{"path": "src/models.py"}'),
                ],
            },
        ]
        reads, mods = extract_file_ops(messages)
        assert "src/auth.py" in reads
        assert "src/models.py" in reads
        assert mods == []

    def test_modified_files(self):
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    make_tool_call("c1", "edit", '{"path": "src/auth.py"}'),
                    make_tool_call("c2", "write", '{"path": "src/new_file.py"}'),
                ],
            },
        ]
        reads, mods = extract_file_ops(messages)
        assert "src/auth.py" in mods
        assert "src/new_file.py" in mods

    def test_grep_and_find_paths(self):
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    make_tool_call("c1", "grep", '{"pattern": "TODO", "path": "src/"}'),
                    make_tool_call("c2", "find", '{"pattern": "*.py", "path": "tests/"}'),
                ],
            },
        ]
        reads, mods = extract_file_ops(messages)
        assert "src/" in reads
        assert "tests/" in reads

    def test_cumulative_merge(self):
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [make_tool_call("c1", "read", '{"path": "new.py"}')],
            },
        ]
        reads, mods = extract_file_ops(
            messages,
            prev_read=["old.py", "also_old.py"],
            prev_modified=["changed.py"],
        )
        assert "old.py" in reads
        assert "also_old.py" in reads
        assert "new.py" in reads
        assert "changed.py" in mods

    def test_deduplication(self):
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    make_tool_call("c1", "read", '{"path": "foo.py"}'),
                    make_tool_call("c2", "read", '{"path": "foo.py"}'),
                ],
            },
        ]
        reads, _ = extract_file_ops(messages)
        assert reads.count("foo.py") == 1

    def test_empty_messages(self):
        reads, mods = extract_file_ops([])
        assert reads == []
        assert mods == []

    def test_malformed_json_args(self):
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [make_tool_call("c1", "read", "not-json")],
            },
        ]
        reads, mods = extract_file_ops(messages)
        assert reads == []  # should not crash


class TestAppendFileTags:
    def test_appends_tags_when_missing(self):
        summary = "## Goal\nSome goal"
        result = append_file_tags(summary, ["foo.py"], ["bar.py"])
        assert "<read-files>" in result
        assert "foo.py" in result
        assert "<modified-files>" in result
        assert "bar.py" in result

    def test_preserves_existing_tags(self):
        summary = "## Goal\nGoal\n<read-files>\nfoo.py\n</read-files>\n<modified-files>\nbar.py\n</modified-files>"
        result = append_file_tags(summary, [], [])
        assert result == summary  # unchanged

    def test_empty_files(self):
        summary = "## Goal\nSome goal"
        result = append_file_tags(summary, [], [])
        assert "<read-files>" in result
        assert "<modified-files>" in result


# ── Prompt building tests ──────────────────────────────────────────


class TestBuildCompactionPrompt:
    def test_normal_compaction(self):
        messages = build_conversation(3)
        prompt = build_compaction_prompt(messages_to_summarize=messages)
        assert "Complete Conversation Turns" in prompt
        assert "structured summary" in prompt.lower()

    def test_split_turn(self):
        messages = build_conversation(1)
        prefix = messages[:2]
        prompt = build_compaction_prompt(
            messages_to_summarize=[],
            turn_prefix_messages=prefix,
            is_split_turn=True,
        )
        assert "In-Progress Turn" in prompt
        assert "do NOT mark it as complete" in prompt

    def test_custom_instructions(self):
        messages = build_conversation(2)
        prompt = build_compaction_prompt(
            messages_to_summarize=messages,
            instructions="Focus on API changes",
        )
        assert "Focus on API changes" in prompt

    def test_empty_messages_with_prefix(self):
        prefix = [{"role": "user", "content": "hello"}]
        prompt = build_compaction_prompt(
            messages_to_summarize=[],
            turn_prefix_messages=prefix,
        )
        assert "In-Progress Turn" in prompt


# ── Token-based split tests ────────────────────────────────────────


class TestFindSplitByTokens:
    def test_basic_split(self):
        """Token budget should keep recent messages within budget."""
        config = CompactionConfig(keep_recent_tokens=500)
        compactor = Compactor(config)
        messages = build_conversation(10, tool_result_size=100)

        split_idx = compactor._find_split_by_tokens(messages)
        assert 0 < split_idx < len(messages)

        # Messages from split_idx onward should be within budget
        from mini_pi.compactor import estimate_messages_tokens
        kept_tokens = estimate_messages_tokens(messages[split_idx:])
        assert kept_tokens <= 500 + 50  # allow small overhead for message-level rounding

    def test_entire_conversation_fits(self):
        """If entire conversation fits in budget, split_idx = 0."""
        config = CompactionConfig(keep_recent_tokens=100000)
        compactor = Compactor(config)
        messages = build_conversation(3)

        split_idx = compactor._find_split_by_tokens(messages)
        assert split_idx == 0  # everything fits

    def test_single_message_exceeds_budget(self):
        """A single huge message should still allow a split."""
        config = CompactionConfig(keep_recent_tokens=100)
        compactor = Compactor(config)
        messages = [
            {"role": "user", "content": "small"},
            {"role": "assistant", "content": "x" * 10000},
        ]

        split_idx = compactor._find_split_by_tokens(messages)
        # The huge assistant message can't fit, so split should be after it
        assert split_idx == 2 or split_idx == 1

    def test_empty_messages(self):
        config = CompactionConfig(keep_recent_tokens=100)
        compactor = Compactor(config)
        split_idx = compactor._find_split_by_tokens([])
        assert split_idx == 0


# ── Turn boundary enforcement tests ────────────────────────────────


class TestAdjustToTurnBoundary:
    def test_splits_at_user_message(self):
        config = CompactionConfig(keep_recent_tokens=500)
        compactor = Compactor(config)
        messages = build_conversation(5)

        raw_split = compactor._find_split_by_tokens(messages)
        adjusted = compactor._adjust_to_turn_boundary(messages, raw_split)

        # Adjusted should be at a user message
        if adjusted > 0:
            assert messages[adjusted]["role"] == "user"

    def test_never_breaks_tool_pairs(self):
        """CRITICAL: tool result must never be separated from its tool call."""
        config = CompactionConfig(keep_recent_tokens=500)
        compactor = Compactor(config)

        # Force a split that lands on a tool result
        messages = build_conversation(5)
        raw_split = compactor._find_split_by_tokens(messages)
        adjusted = compactor._adjust_to_turn_boundary(messages, raw_split)

        # Verify no orphaned tool results
        if adjusted < len(messages):
            assert messages[adjusted]["role"] != "tool", \
                "Split starts with orphaned tool result!"

    def test_preserves_recent_messages(self):
        config = CompactionConfig(keep_recent_tokens=2000)
        compactor = Compactor(config)
        messages = build_conversation(5, tool_result_size=300)

        raw_split = compactor._find_split_by_tokens(messages)
        adjusted = compactor._adjust_to_turn_boundary(messages, raw_split)

        # Kept messages should be at least as many as raw_split
        assert adjusted <= raw_split or adjusted == 0


class TestAdjustForToolPairs:
    def test_moves_past_tool_results(self):
        config = CompactionConfig()
        compactor = Compactor(config)
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None, "tool_calls": [make_tool_call("c1")]},
            {"role": "tool", "tool_call_id": "c1", "content": "output"},
            {"role": "user", "content": "next"},
        ]

        # Split at index 2 (tool result) — should move to index 3 (user)
        adjusted = compactor._adjust_for_tool_pairs(messages, 2)
        assert adjusted == 3

    def test_moves_past_consecutive_tool_results(self):
        config = CompactionConfig()
        compactor = Compactor(config)
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None,
             "tool_calls": [make_tool_call("c1"), make_tool_call("c2")]},
            {"role": "tool", "tool_call_id": "c1", "content": "out1"},
            {"role": "tool", "tool_call_id": "c2", "content": "out2"},
            {"role": "user", "content": "next"},
        ]

        # Split at index 2 (first tool result) — should move past both
        adjusted = compactor._adjust_for_tool_pairs(messages, 2)
        assert adjusted == 4

    def test_clean_boundary_unchanged(self):
        config = CompactionConfig()
        compactor = Compactor(config)
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "next"},
        ]

        # Split at index 2 (user) — should stay
        adjusted = compactor._adjust_for_tool_pairs(messages, 2)
        assert adjusted == 2

    def test_assistant_with_following_tool_results(self):
        config = CompactionConfig()
        compactor = Compactor(config)
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None, "tool_calls": [make_tool_call("c1")]},
            {"role": "tool", "tool_call_id": "c1", "content": "out"},
            {"role": "user", "content": "next"},
        ]

        # Split at index 1 (assistant with tool_calls) — should move past tool result
        adjusted = compactor._adjust_for_tool_pairs(messages, 1)
        assert adjusted == 3


# ── Split turn detection tests ──────────────────────────────────────


class TestFindTurnStart:
    def test_finds_user_before_split(self):
        config = CompactionConfig()
        compactor = Compactor(config)
        messages = [
            {"role": "user", "content": "start turn"},
            {"role": "assistant", "content": "working..."},
            {"role": "assistant", "content": "more..."},
        ]

        turn_start = compactor._find_turn_start(messages, split_idx=3)
        assert turn_start == 0

    def test_with_multiple_turns(self):
        config = CompactionConfig()
        compactor = Compactor(config)
        messages = [
            {"role": "user", "content": "turn 1"},
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "turn 2"},
            {"role": "assistant", "content": "working..."},
            {"role": "tool", "tool_call_id": "c1", "content": "output"},
        ]

        turn_start = compactor._find_turn_start(messages, split_idx=4)
        assert turn_start == 2  # "turn 2" user message


# ── Full compaction flow tests ──────────────────────────────────────


class TestCompactor:
    def test_compact_with_mock_llm(self):
        config = CompactionConfig(keep_recent_tokens=500, model="test-model")
        compactor = Compactor(config)
        messages = build_conversation(10, tool_result_size=200)

        with patch.object(compactor, "_call_llm_for_summary", return_value="## Goal\nTest summary"):
            result = compactor.compact(messages)

        assert result.success is True
        assert "Goal" in result.summary
        assert result.original_count == 40
        assert result.first_kept_index > 0
        assert result.tokens_before > 0
        assert result.compacted_count == 1 + len(result._recent_messages)

    def test_compact_disabled(self):
        config = CompactionConfig(enabled=False)
        compactor = Compactor(config)
        messages = build_conversation(5)

        result = compactor.compact(messages)
        assert result.success is False
        assert "disabled" in result.error.lower()

    def test_compact_short_conversation(self):
        """Conversation that fits entirely in keep_recent_tokens."""
        config = CompactionConfig(keep_recent_tokens=100000)
        compactor = Compactor(config)
        messages = [{"role": "user", "content": "hi"}]

        result = compactor.compact(messages)
        assert result.success is False
        assert "too short" in result.error.lower()

    def test_compact_preserves_recent_tool_outputs(self):
        """Recent tool results must be kept INTACT in the result."""
        config = CompactionConfig(keep_recent_tokens=200)
        compactor = Compactor(config)

        long_output = "IMPORTANT_DATA_" * 100
        messages = build_conversation(5, tool_result_size=500)

        with patch.object(compactor, "_call_llm_for_summary", return_value="## Goal\nSummary"):
            result = compactor.compact(messages)

        assert result.success is True
        # At least one recent tool result should have full content
        recent_tool_msgs = [m for m in (result._recent_messages or []) if m.get("role") == "tool"]
        if recent_tool_msgs:
            # Some recent tool results exist — they should be intact
            assert any(len(m["content"]) > 100 for m in recent_tool_msgs)

    def test_compact_tracks_files(self):
        """File operations should be tracked in the result."""
        config = CompactionConfig(keep_recent_tokens=50)
        compactor = Compactor(config)
        messages = build_conversation_with_file_ops()

        with patch.object(compactor, "_call_llm_for_summary", return_value="## Goal\nSummary"):
            result = compactor.compact(messages)

        assert result.success is True
        # Should have tracked file operations
        assert len(result.read_files) > 0 or len(result.modified_files) > 0

    def test_compact_cumulative_file_tracking(self):
        """File tracking should merge with previous lists."""
        config = CompactionConfig(keep_recent_tokens=50)
        compactor = Compactor(config)
        messages = build_conversation_with_file_ops()

        with patch.object(compactor, "_call_llm_for_summary", return_value="## Goal\nSummary"):
            result = compactor.compact(
                messages,
                prev_read_files=["previously_read.py"],
                prev_modified_files=["previously_changed.py"],
            )

        assert result.success is True
        assert "previously_read.py" in result.read_files
        assert "previously_changed.py" in result.modified_files

    def test_compact_split_turn_detection(self):
        """When a single turn exceeds keep_recent_tokens, is_split_turn should be True."""
        config = CompactionConfig(keep_recent_tokens=200)
        compactor = Compactor(config)

        # One huge turn with many tool calls
        messages = [{"role": "user", "content": "Do everything at once"}]
        for i in range(20):
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [make_tool_call(f"c_{i}", "read", f'{{"path": "file_{i}.py"}}')],
            })
            messages.append({
                "role": "tool",
                "tool_call_id": f"c_{i}",
                "content": "x" * 500,
            })
        messages.append({"role": "assistant", "content": "All done"})

        with patch.object(compactor, "_call_llm_for_summary", return_value="## Goal\nSummary"):
            result = compactor.compact(messages)

        assert result.success is True
        # This should be a split turn since one turn has many large tool outputs
        assert result.is_split_turn is True

    def test_compact_empty_messages(self):
        config = CompactionConfig()
        compactor = Compactor(config)
        result = compactor.compact([])
        assert result.success is False

    def test_compact_appends_file_tags(self):
        """Summary should always have <read-files> and <modified-files> tags."""
        config = CompactionConfig(keep_recent_tokens=50)
        compactor = Compactor(config)
        messages = build_conversation_with_file_ops()

        # LLM returns a summary WITHOUT file tags
        with patch.object(compactor, "_call_llm_for_summary", return_value="## Goal\nTest"):
            result = compactor.compact(messages)

        assert result.success is True
        assert "<read-files>" in result.summary
        assert "<modified-files>" in result.summary

    def test_tool_pair_integrity_after_compaction(self):
        """CRITICAL: No assistant+tool_call in old section has its results in recent."""
        config = CompactionConfig(keep_recent_tokens=300)
        compactor = Compactor(config)
        messages = build_conversation(8, tool_result_size=150)

        with patch.object(compactor, "_call_llm_for_summary", return_value="## Goal\nSummary"):
            result = compactor.compact(messages)

        assert result.success is True

        # Check tool pair integrity
        summary_messages = result._summary_messages or []
        recent_messages = result._recent_messages or []

        old_tool_call_ids = set()
        for msg in summary_messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    old_tool_call_ids.add(tc["id"])

        for msg in recent_messages:
            if msg.get("role") == "tool":
                tid = msg.get("tool_call_id")
                assert tid not in old_tool_call_ids, \
                    f"Tool pair broken: call in old, result {tid} in recent"

    def test_compact_with_multi_tool_turn(self):
        """Multiple tool calls in one assistant message."""
        config = CompactionConfig(keep_recent_tokens=200)
        compactor = Compactor(config)

        messages = []
        messages.extend(build_multi_tool_turn("t1", ["read", "bash", "grep"], result_size=500))
        messages.extend(build_multi_tool_turn("t2", ["read", "edit"], result_size=500))
        messages.extend(build_multi_tool_turn("t3", ["bash", "bash"], result_size=500))

        with patch.object(compactor, "_call_llm_for_summary", return_value="## Goal\nSummary"):
            result = compactor.compact(messages)

        assert result.success is True
        # Verify tool pair integrity
        self._verify_tool_pair_integrity(result)

    def _verify_tool_pair_integrity(self, result: CompactResult):
        summary_messages = result._summary_messages or []
        recent_messages = result._recent_messages or []

        old_tool_call_ids = set()
        for msg in summary_messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    old_tool_call_ids.add(tc["id"])

        for msg in recent_messages:
            if msg.get("role") == "tool":
                tid = msg.get("tool_call_id")
                assert tid not in old_tool_call_ids


# ── Session integration tests ──────────────────────────────────────


class TestSessionCompaction:
    def test_session_records_compaction_entry(self, tmp_path):
        session = Session(tmp_path / "test.jsonl")
        messages = build_conversation(5, tool_result_size=500)
        session.messages = list(messages)
        session.save()

        config = CompactionConfig(keep_recent_tokens=200)
        compactor = Compactor(config)

        with patch.object(compactor, "_call_llm_for_summary", return_value="## Goal\nSummary"):
            result = compactor.compact(session.messages)

        session.record_compaction(result)

        # Verify JSONL has compaction entry
        content = (tmp_path / "test.jsonl").read_text()
        assert '"type": "compaction"' in content
        assert "first_kept_index" in content
        assert "tokens_before" in content

    def test_session_tracks_compaction_state(self, tmp_path):
        session = Session(tmp_path / "test.jsonl")
        messages = build_conversation(5, tool_result_size=500)
        session.messages = list(messages)
        session.save()

        config = CompactionConfig(keep_recent_tokens=200)
        compactor = Compactor(config)

        with patch.object(compactor, "_call_llm_for_summary", return_value="## Goal\nTest summary"):
            result = compactor.compact(session.messages)

        session.record_compaction(result)

        assert session.compaction_count == 1
        assert "Test summary" in session.last_compaction_summary
        assert session.last_read_files is not None

    def test_session_get_openai_messages_injects_summary(self, tmp_path):
        session = Session(tmp_path / "test.jsonl")
        session.add_user("hello")
        session.add_assistant("hi")

        # Manually set compaction state
        session._last_compaction_summary = "## Goal\nPrevious task"
        session._compaction_count = 1

        msgs = session.get_openai_messages()
        # First message should be the summary injection
        assert msgs[0]["role"] == "system"
        assert "Previous task" in msgs[0]["content"]
        # Then the actual messages
        assert msgs[1]["role"] == "user"
        assert msgs[1]["content"] == "hello"

    def test_session_no_summary_injection_when_none(self, tmp_path):
        session = Session(tmp_path / "test.jsonl")
        session.add_user("hello")

        msgs = session.get_openai_messages()
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "hello"

    def test_session_reload_from_compaction(self, tmp_path):
        """After compaction, reload should only keep messages from first_kept_index."""
        path = tmp_path / "test.jsonl"
        session = Session(path)
        messages = build_conversation(5, tool_result_size=500)
        session.messages = list(messages)
        session.save()

        config = CompactionConfig(keep_recent_tokens=200)
        compactor = Compactor(config)

        with patch.object(compactor, "_call_llm_for_summary", return_value="## Goal\nSummary"):
            result = compactor.compact(session.messages)

        session.record_compaction(result)
        kept_count = len(session.messages)

        # Only test reload if compaction actually happened
        if kept_count < len(messages):
            session2 = Session(path)
            assert len(session2.messages) == kept_count
            assert "Summary" in session2.last_compaction_summary

    def test_session_preserves_full_history_on_disk(self, tmp_path):
        """JSONL file should still have all original messages after compaction."""
        path = tmp_path / "test.jsonl"
        session = Session(path)
        messages = build_conversation(5, tool_result_size=500)
        session.messages = list(messages)
        session.save()

        original_line_count = len(path.read_text().strip().splitlines())

        config = CompactionConfig(keep_recent_tokens=200)
        compactor = Compactor(config)

        with patch.object(compactor, "_call_llm_for_summary", return_value="## Goal\nSummary"):
            result = compactor.compact(session.messages)

        session.record_compaction(result)

        # File should have MORE lines (original + compaction entry), not fewer
        new_line_count = len(path.read_text().strip().splitlines())
        assert new_line_count >= original_line_count

    def test_session_cumulative_file_tracking(self, tmp_path):
        """Multiple compactions should accumulate file lists."""
        path = tmp_path / "test.jsonl"
        session = Session(path)

        # First conversation with file ops
        messages1 = build_conversation_with_file_ops()
        session.messages = list(messages1)
        session.save()

        config = CompactionConfig(keep_recent_tokens=50)
        compactor = Compactor(config)

        with patch.object(compactor, "_call_llm_for_summary", return_value="## Goal\nSummary 1"):
            result1 = compactor.compact(session.messages)

        session.record_compaction(result1)
        first_read_files = set(session.last_read_files)

        # Simulate more conversation after compaction
        session.add_user("Now refactor tests")
        session.add_assistant("OK",
            tool_calls=[make_tool_call("d1", "read", '{"path": "tests/test_auth.py"}')])
        session.add_tool_result("d1", "test contents")
        session.add_assistant("Done")
        session.save()

        # Second compaction — pass cumulative file lists
        with patch.object(compactor, "_call_llm_for_summary", return_value="## Goal\nSummary 2"):
            result2 = compactor.compact(
                session.messages,
                prev_read_files=session.last_read_files,
                prev_modified_files=session.last_modified_files,
            )

        if result2.success:
            # Second compaction should include files from first
            assert "tests/test_auth.py" in result2.read_files

    def test_legacy_snapshot_migration(self, tmp_path):
        """Old snapshot-format sessions should be loadable."""
        path = tmp_path / "legacy.jsonl"
        # Write a legacy snapshot format
        with open(path, "w") as f:
            f.write(json.dumps({"type": "meta", "created_at": "2024-01-01"}) + "\n")
            f.write(json.dumps({
                "type": "message",
                "data": {"role": "user", "content": "old message"},
            }) + "\n")
            f.write(json.dumps({
                "type": "message",
                "data": {"role": "assistant", "content": "old reply"},
            }) + "\n")
            f.write(json.dumps({
                "type": "snapshot",
                "messages": [
                    {"role": "system", "content": "[Compaction Summary]\nLegacy summary"},
                    {"role": "user", "content": "kept message"},
                ],
                "summary": "Legacy summary",
            }) + "\n")

        session = Session(path)
        # Should load the snapshot messages
        assert len(session.messages) == 2
        assert session.messages[0]["role"] == "system"
        assert session.messages[1]["content"] == "kept message"
        assert "Legacy" in session.last_compaction_summary


# ── CompactResult tests ────────────────────────────────────────────


class TestCompactResult:
    def test_success_result(self):
        result = CompactResult(
            success=True,
            summary="## Goal\nTest",
            original_count=20,
            compacted_count=5,
            first_kept_index=15,
            tokens_before=50000,
        )
        assert result.success is True
        assert result.first_kept_index == 15
        assert result.tokens_before == 50000

    def test_default_file_lists(self):
        result = CompactResult(success=True)
        assert result.read_files == []
        assert result.modified_files == []

    def test_split_turn_flag(self):
        result = CompactResult(success=True, is_split_turn=True)
        assert result.is_split_turn is True

    def test_failure_result(self):
        result = CompactResult(success=False, error="fail")
        assert result.success is False


# ── Summary format validation tests ────────────────────────────────


class TestSummaryFormat:
    def test_system_prompt_requires_structured_format(self):
        """COMPACTION_SYSTEM_PROMPT should instruct the LLM to use structured format."""
        assert "## Goal" in COMPACTION_SYSTEM_PROMPT
        assert "## Progress" in COMPACTION_SYSTEM_PROMPT
        assert "## Key Decisions" in COMPACTION_SYSTEM_PROMPT
        assert "## Next Steps" in COMPACTION_SYSTEM_PROMPT
        assert "<read-files>" in COMPACTION_SYSTEM_PROMPT
        assert "<modified-files>" in COMPACTION_SYSTEM_PROMPT


# ── Edge case tests ────────────────────────────────────────────────


class TestEdgeCases:
    def test_single_message(self):
        config = CompactionConfig(keep_recent_tokens=500)
        compactor = Compactor(config)
        result = compactor.compact([{"role": "user", "content": "hi"}])
        assert result.success is False  # too short

    def test_all_tool_results(self):
        """Conversation with only tool results (no user messages)."""
        config = CompactionConfig(keep_recent_tokens=100)
        compactor = Compactor(config)
        messages = [
            {"role": "assistant", "content": None, "tool_calls": [make_tool_call("c1")]},
            {"role": "tool", "tool_call_id": "c1", "content": "x" * 500},
            {"role": "assistant", "content": None, "tool_calls": [make_tool_call("c2")]},
            {"role": "tool", "tool_call_id": "c2", "content": "y" * 500},
        ]

        with patch.object(compactor, "_call_llm_for_summary", return_value="## Goal\nSummary"):
            result = compactor.compact(messages)

        # Should either succeed or fail gracefully
        if result.success:
            assert result.first_kept_index <= len(messages)

    def test_very_large_single_message(self):
        """One message that exceeds keep_recent_tokens by itself."""
        config = CompactionConfig(keep_recent_tokens=100)
        compactor = Compactor(config)
        messages = [
            {"role": "user", "content": "x" * 10000},
            {"role": "assistant", "content": "ok"},
        ]

        # The huge message can't fit in budget, so everything goes to "old"
        # But there's only 1 turn — this is a split turn
        with patch.object(compactor, "_call_llm_for_summary", return_value="## Goal\nSummary"):
            result = compactor.compact(messages)

        # Should handle gracefully (either success or clear error)
        assert result.success or result.error

    def test_alternating_user_assistant_no_tools(self):
        """Simple conversation without any tool calls."""
        config = CompactionConfig(keep_recent_tokens=100)
        compactor = Compactor(config)
        messages = []
        for i in range(20):
            messages.append({"role": "user", "content": f"Message {i}"})
            messages.append({"role": "assistant", "content": f"Reply {i}"})

        with patch.object(compactor, "_call_llm_for_summary", return_value="## Goal\nSummary"):
            result = compactor.compact(messages)

        assert result.success is True
        # No tools means every turn is a clean user→assistant pair
        # Split turn is possible if one turn spans the budget boundary
        # (this is acceptable behavior)
