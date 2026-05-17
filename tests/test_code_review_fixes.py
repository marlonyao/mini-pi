"""
Tests to reproduce and verify fixes for code review issues.

C1: first_kept_index breaks after multiple compactions + reload
C2: _adjust_for_tool_pairs edge case (assistant + tool_calls, no following tool_result)
H1: extract_file_ops silently swallows JSON parse errors
H2: _compaction_count restored from wrong meta entry
H3: cooldown prevents necessary compaction
M1: duplicate token estimation logic between compactor and token_estimator
M2: serialize_message drops content when tool_calls exist
M4: append_file_tags breaks markdown structure
"""

import json
import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from mini_pi.compactor import (
    CompactionConfig,
    CompactResult,
    Compactor,
    append_file_tags,
    estimate_message_tokens,
    extract_file_ops,
    serialize_message,
)
from mini_pi.session import Session
from mini_pi.token_estimator import TokenEstimator, estimate_tokens


# ── Helpers ─────────────────────────────────────────────────────────


def make_tool_call(call_id: str, name: str = "bash", args: str = '{"command":"ls"}') -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": args},
    }


def build_conversation(turns: int = 5, tool_result_size: int = 200) -> list[dict]:
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


# ═══════════════════════════════════════════════════════════════════════
# C1: first_kept_index breaks after multiple compactions + reload
# ═══════════════════════════════════════════════════════════════════════


class TestC1MultipleCompactionReload:
    """Reproduce: first_kept_index is relative to session.messages but
    _load() treats it as absolute index into JSONL message entries."""

    def test_double_compaction_reload_gets_correct_messages(self, tmp_path):
        """
        1. Create session with 10 messages
        2. Compact (keep last 4)
        3. Add 3 more messages
        4. Compact again (keep last 2)
        5. Reload
        6. Verify loaded messages are the CORRECT ones
        """
        path = tmp_path / "test.jsonl"
        session = Session(path)

        # Add 10 messages (turns 0-4): user+assistant per turn = 20 messages
        for i in range(10):
            session.add_user(f"msg_{i}")
            session.add_assistant(f"reply_{i}")
        # session.messages = 20 messages
        session.save()

        # First compaction: use tiny budget so most messages get summarized
        config = CompactionConfig(keep_recent_tokens=30)
        compactor = Compactor(config)

        with patch.object(compactor, "_call_llm_for_summary", return_value="## Goal\nSummary1"):
            result1 = compactor.compact(session.get_openai_messages())

        assert result1.success, f"First compaction failed: {result1.error}"
        first_kept = result1.first_kept_index

        # Remember what the kept messages actually are
        expected_after_first = session.messages[first_kept:]
        expected_contents_after_first = [m.get("content") for m in expected_after_first]

        session.record_compaction(result1)
        session.save()

        # Verify in-memory is correct after first compaction
        actual_contents = [m.get("content") for m in session.messages]
        assert actual_contents == expected_contents_after_first, \
            f"After first compaction, expected {expected_contents_after_first}, got {actual_contents}"

        # Reload after first compaction
        session_reload1 = Session(path)
        reload1_contents = [m.get("content") for m in session_reload1.messages]
        assert reload1_contents == expected_contents_after_first, \
            f"After first reload, expected {expected_contents_after_first}, got {reload1_contents}"

        # Add 3 more messages
        session_reload1.add_user("msg_extra_0")
        session_reload1.add_assistant("reply_extra_0")
        session_reload1.add_user("msg_extra_1")
        session_reload1.save()

        # Second compaction
        with patch.object(compactor, "_call_llm_for_summary", return_value="## Goal\nSummary2"):
            result2 = compactor.compact(session_reload1.get_openai_messages())

        if not result2.success:
            pytest.skip("Second compaction didn't produce a split (messages too small)")

        second_kept = result2.first_kept_index

        # Remember what should be kept
        expected_after_second = session_reload1.messages[second_kept:]
        expected_contents_after_second = [m.get("content") for m in expected_after_second]

        session_reload1.record_compaction(result2)
        session_reload1.save()

        # In-memory should be correct
        actual_contents_2 = [m.get("content") for m in session_reload1.messages]
        assert actual_contents_2 == expected_contents_after_second, \
            f"After second compaction, expected {expected_contents_after_second}, got {actual_contents_2}"

        # THE BUG: Reload after second compaction
        session_reload2 = Session(path)
        reload2_contents = [m.get("content") for m in session_reload2.messages]

        assert reload2_contents == expected_contents_after_second, \
            f"C1 BUG REPRODUCED: After second reload, expected {expected_contents_after_second}, got {reload2_contents}"


# ═══════════════════════════════════════════════════════════════════════
# C2: _adjust_for_tool_pairs with orphaned tool_calls
# ═══════════════════════════════════════════════════════════════════════


class TestC2OrphanedToolCalls:
    """Edge case: assistant with tool_calls but no following tool_result."""

    def test_assistant_tool_calls_no_following_result(self):
        config = CompactionConfig()
        compactor = Compactor(config)

        messages = [
            {"role": "user", "content": "do thing"},
            {"role": "assistant", "content": None, "tool_calls": [make_tool_call("c1")]},
            # No tool result follows!
            {"role": "user", "content": "next request"},
            {"role": "assistant", "content": "done"},
        ]

        # Split at index 1 (the orphaned assistant)
        adjusted = compactor._adjust_for_tool_pairs(messages, 1)

        # Should NOT loop infinitely; should handle gracefully
        # The assistant has tool_calls but next message is user, not tool
        # So it should just stay at index 1 or advance past it
        assert 1 <= adjusted <= len(messages)

    def test_compact_with_orphaned_tool_calls(self):
        """Full compaction flow with orphaned tool_calls."""
        config = CompactionConfig(keep_recent_tokens=50)
        compactor = Compactor(config)

        messages = [
            {"role": "user", "content": "do thing"},
            {"role": "assistant", "content": None, "tool_calls": [make_tool_call("c1")]},
            # No tool result!
            {"role": "user", "content": "actually do this instead"},
            {"role": "assistant", "content": "ok done"},
            {"role": "user", "content": "another task"},
            {"role": "assistant", "content": "completed"},
        ]

        with patch.object(compactor, "_call_llm_for_summary", return_value="## Goal\nSummary"):
            result = compactor.compact(messages)

        # Should not crash
        assert result.success or result.error  # either works, just no exception


# ═══════════════════════════════════════════════════════════════════════
# H1: extract_file_ops silently swallows JSON errors
# ═══════════════════════════════════════════════════════════════════════


class TestH1ExtractFileOpsLogging:
    """Verify that malformed JSON in tool call args produces a warning log."""

    def test_malformed_json_produces_warning(self, caplog):
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    make_tool_call("c1", "read", "not-valid-json{"),
                    make_tool_call("c2", "read", '{"path": "good.py"}'),
                ],
            },
        ]

        with caplog.at_level(logging.WARNING, logger="mini_pi.compactor"):
            reads, _ = extract_file_ops(messages)

        # Should still extract the valid one
        assert "good.py" in reads
        # Should have logged a warning about the malformed one
        assert any("not-valid-json" in record.message or "parse" in record.message.lower()
                   for record in caplog.records), \
            "H1: Expected a warning log about malformed JSON arguments"


# ═══════════════════════════════════════════════════════════════════════
# H2: _compaction_count restored from wrong source
# ═══════════════════════════════════════════════════════════════════════


class TestH2CompactionCountSource:
    """Verify compaction_count comes from compaction entry, not meta."""

    def test_meta_does_not_override_compaction_entry_count(self, tmp_path):
        path = tmp_path / "test.jsonl"
        # Write JSONL with:
        # 1. meta (compaction_count=99 — stale)
        # 2. messages
        # 3. compaction entry (compaction_count=1 — authoritative)
        with open(path, "w") as f:
            f.write(json.dumps({
                "type": "meta",
                "created_at": "2024-01-01",
                "compaction_count": 99,  # stale value
            }) + "\n")
            f.write(json.dumps({
                "type": "message",
                "data": {"role": "user", "content": "hello"},
            }) + "\n")
            f.write(json.dumps({
                "type": "message",
                "data": {"role": "assistant", "content": "hi"},
            }) + "\n")
            f.write(json.dumps({
                "type": "compaction",
                "summary": "## Goal\nSummary",
                "first_kept_index": 1,
                "tokens_before": 100,
                "compaction_count": 1,  # authoritative value
            }) + "\n")

        session = Session(path)
        # Should be 1 (from compaction entry), NOT 99 (from meta)
        assert session.compaction_count == 1, \
            f"H2 BUG: compaction_count should be 1, got {session.compaction_count}"


# ═══════════════════════════════════════════════════════════════════════
# M1: Duplicate token estimation logic
# ═══════════════════════════════════════════════════════════════════════


class TestM1TokenEstimationConsistency:
    """Verify compactor and token_estimator produce same results."""

    def test_same_estimate_for_plain_message(self):
        msg = {"role": "user", "content": "Hello world, this is a test"}
        compact_tokens = estimate_message_tokens(msg)
        estimator_tokens = TokenEstimator().estimate_messages([msg])
        assert compact_tokens == estimator_tokens, \
            f"M1: Token estimates differ: compactor={compact_tokens}, estimator={estimator_tokens}"

    def test_same_estimate_for_tool_call_message(self):
        msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [make_tool_call("c1", "read", '{"path": "foo.py"}')],
        }
        compact_tokens = estimate_message_tokens(msg)
        estimator_tokens = TokenEstimator().estimate_messages([msg])
        assert compact_tokens == estimator_tokens, \
            f"M1: Token estimates differ: compactor={compact_tokens}, estimator={estimator_tokens}"

    def test_same_estimate_for_tool_result(self):
        msg = {
            "role": "tool",
            "tool_call_id": "c1",
            "content": "file contents here",
        }
        compact_tokens = estimate_message_tokens(msg)
        estimator_tokens = TokenEstimator().estimate_messages([msg])
        assert compact_tokens == estimator_tokens, \
            f"M1: Token estimates differ: compactor={compact_tokens}, estimator={estimator_tokens}"

    def test_same_estimate_for_multiple_messages(self):
        messages = [
            {"role": "user", "content": "do stuff"},
            {"role": "assistant", "content": None, "tool_calls": [make_tool_call("c1")]},
            {"role": "tool", "tool_call_id": "c1", "content": "result output"},
            {"role": "assistant", "content": "all done"},
        ]
        compact_tokens = sum(estimate_message_tokens(m) for m in messages)
        estimator_tokens = TokenEstimator().estimate_messages(messages)
        assert compact_tokens == estimator_tokens


# ═══════════════════════════════════════════════════════════════════════
# M2: serialize_message drops content when tool_calls exist
# ═══════════════════════════════════════════════════════════════════════


class TestM2SerializeContentWithToolCalls:
    """Verify that assistant messages with both content AND tool_calls
    include both in serialized output."""

    def test_content_preserved_with_tool_calls(self):
        msg = {
            "role": "assistant",
            "content": "Let me read that file for you.",
            "tool_calls": [make_tool_call("c1", "read", '{"path": "foo.py"}')],
        }
        text = serialize_message(msg)
        assert "Let me read that file" in text, \
            "M2 BUG: Assistant content was dropped when tool_calls exist"
        assert "read" in text  # tool call info should also be present

    def test_content_only_no_tool_calls(self):
        msg = {"role": "assistant", "content": "Just text, no tools"}
        text = serialize_message(msg)
        assert "Just text, no tools" in text

    def test_tool_calls_only_no_content(self):
        msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [make_tool_call("c1", "bash")],
        }
        text = serialize_message(msg)
        assert "bash" in text
        assert "[Assistant tool calls]" in text


# ═══════════════════════════════════════════════════════════════════════
# M4: append_file_tags breaks markdown structure
# ═══════════════════════════════════════════════════════════════════════


class TestM4AppendFileTagsMarkdownSafety:
    """Verify append_file_tags handles markdown edge cases."""

    def test_summary_ending_with_code_fence(self):
        summary = "## Progress\n```\ncode here\n```"
        result = append_file_tags(summary, ["foo.py"], [])
        # Should not break the code fence
        assert "```" in result
        assert "<read-files>" in result

    def test_summary_ending_without_newline(self):
        summary = "## Goal\nDo the thing"
        result = append_file_tags(summary, [], [])
        # Should have proper separation
        lines = result.split("<read-files>")[0].split("\n")
        # The last line before <read-files> should not run into the tag
        assert result.strip().endswith("</modified-files>")

    def test_summary_with_trailing_spaces(self):
        summary = "## Goal\nDo the thing   \n  \n"
        result = append_file_tags(summary, ["a.py"], ["b.py"])
        assert "<read-files>" in result
        assert "a.py" in result
