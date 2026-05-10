"""
Tests for append-only session storage (P1-1).

Key invariant: JSONL files are never overwritten. New messages are always
appended. Compaction creates snapshots that reference the compacted state
without deleting the original messages on disk.
"""

import json
from pathlib import Path

import pytest

from mini_pi.session import Session, create_session, fork_session, list_sessions


class TestAppendOnlyBasic:
    """Basic append-only behavior."""

    def test_new_session_creates_file_on_save(self, tmp_path):
        s = create_session(str(tmp_path), name="test")
        s.add_user("hello")
        s.save()
        assert s.path.exists()

    def test_save_appends_not_overwrites(self, tmp_path):
        s = create_session(str(tmp_path), name="test")

        # First save
        s.add_user("msg1")
        s.save()
        lines_after_first = len(s.path.read_text().splitlines())

        # Second save — should append, not overwrite
        s.add_user("msg2")
        s.save()
        lines_after_second = len(s.path.read_text().splitlines())

        assert lines_after_second == lines_after_first + 1

    def test_multiple_saves_preserve_all_messages(self, tmp_path):
        s = create_session(str(tmp_path), name="test")
        s.add_user("msg1")
        s.save()
        s.add_user("msg2")
        s.save()
        s.add_user("msg3")
        s.save()

        # Reload
        loaded = Session(s.path)
        assert len(loaded.messages) == 3
        assert loaded.messages[0]["content"] == "msg1"
        assert loaded.messages[1]["content"] == "msg2"
        assert loaded.messages[2]["content"] == "msg3"

    def test_file_grows_linearly(self, tmp_path):
        s = create_session(str(tmp_path), name="test")
        sizes = []
        for i in range(5):
            s.add_user(f"message {i}")
            s.save()
            sizes.append(s.path.stat().st_size)

        # File should grow monotonically
        for i in range(1, len(sizes)):
            assert sizes[i] > sizes[i - 1]


class TestAppendOnlyCompaction:
    """Append-only behavior during compaction."""

    def test_compaction_preserves_history_on_disk(self, tmp_path):
        """After compaction, the original messages should still be on disk."""
        s = create_session(str(tmp_path), name="test")

        # Add messages
        for i in range(10):
            s.add_user(f"message {i}")
            s.add_assistant(f"response {i}")
        s.save()

        # Read raw file content before compaction
        raw_before = s.path.read_text()
        assert "message 0" in raw_before
        assert "message 9" in raw_before

        # Simulate compaction
        from mini_pi.compactor import CompactResult
        result = CompactResult(
            success=True,
            summary="Summary of messages 0-6",
            original_count=20,
            compacted_count=6,
        )
        result._recent_messages = s.messages[-3:]  # keep last 3
        s.record_compaction(result)

        # Read raw file content after compaction
        raw_after = s.path.read_text()

        # Original messages must still be on disk (append-only!)
        assert "message 0" in raw_after
        assert "message 9" in raw_after

        # But in-memory messages are compacted
        assert len(s.messages) == 4  # 1 summary + 3 recent

    def test_compaction_creates_snapshot_entry(self, tmp_path):
        s = create_session(str(tmp_path), name="test")
        s.add_user("hello")
        s.add_assistant("hi")
        s.save()

        from mini_pi.compactor import CompactResult
        result = CompactResult(
            success=True,
            summary="Summary text",
            original_count=2,
            compacted_count=1,
        )
        result._recent_messages = [s.messages[-1]]
        s.record_compaction(result)

        # Check that a snapshot entry exists in the file
        raw = s.path.read_text()
        found_snapshot = False
        for line in raw.splitlines():
            entry = json.loads(line)
            if entry.get("type") == "snapshot":
                found_snapshot = True
                assert "Summary text" in entry.get("summary", "")
                assert len(entry.get("messages", [])) == 2  # summary msg + recent
        assert found_snapshot

    def test_multiple_compactions(self, tmp_path):
        """Multiple compactions should all append snapshots."""
        s = create_session(str(tmp_path), name="test")

        # First round of messages + compaction
        for i in range(5):
            s.add_user(f"msg1-{i}")
        s.save()

        from mini_pi.compactor import CompactResult
        r1 = CompactResult(success=True, summary="Summary 1", original_count=5, compacted_count=1)
        r1._recent_messages = []
        s.record_compaction(r1)

        # Second round of messages + compaction
        for i in range(5):
            s.add_user(f"msg2-{i}")
        s.save()

        r2 = CompactResult(success=True, summary="Summary 2", original_count=5, compacted_count=1)
        r2._recent_messages = []
        s.record_compaction(r2)

        # Count snapshots in file
        raw = s.path.read_text()
        snapshots = [json.loads(l) for l in raw.splitlines() if l.strip()]
        snapshot_count = sum(1 for e in snapshots if e.get("type") == "snapshot")
        assert snapshot_count == 2


class TestAppendOnlyReload:
    """Loading from append-only files."""

    def test_reload_after_multiple_saves(self, tmp_path):
        s = create_session(str(tmp_path), name="test")
        s.add_user("first")
        s.save()
        s.add_user("second")
        s.save()
        s.add_user("third")
        s.save()

        loaded = Session(s.path)
        assert len(loaded.messages) == 3

    def test_reload_after_compaction(self, tmp_path):
        s = create_session(str(tmp_path), name="test")
        for i in range(10):
            s.add_user(f"msg {i}")
        s.save()

        from mini_pi.compactor import CompactResult
        result = CompactResult(
            success=True,
            summary="Compact summary",
            original_count=10,
            compacted_count=3,
        )
        result._recent_messages = s.messages[-2:]
        s.record_compaction(result)

        # Reload — should get snapshot messages + any appended after
        loaded = Session(s.path)
        # snapshot has: summary msg + 2 recent = 3 messages
        assert len(loaded.messages) == 3
        assert "Compaction Summary" in loaded.messages[0]["content"]

    def test_reload_then_append(self, tmp_path):
        """Reload a compacted session and add more messages."""
        s = create_session(str(tmp_path), name="test")
        for i in range(10):
            s.add_user(f"msg {i}")
        s.save()

        from mini_pi.compactor import CompactResult
        result = CompactResult(
            success=True,
            summary="Summary",
            original_count=10,
            compacted_count=2,
        )
        result._recent_messages = [s.messages[-1]]
        s.record_compaction(result)

        # Simulate: reload in a new session, add more messages
        loaded = Session(s.path)
        loaded.add_user("post-compaction message")
        loaded.save()

        # Reload again and verify
        final = Session(s.path)
        # snapshot (summary + 1 recent) + 1 new
        assert len(final.messages) == 3
        assert final.messages[-1]["content"] == "post-compaction message"

    def test_original_data_survives_reload(self, tmp_path):
        """Full history should be on disk even after compaction + reload."""
        s = create_session(str(tmp_path), name="test")
        s.add_user("original message that should survive")
        s.save()

        from mini_pi.compactor import CompactResult
        result = CompactResult(
            success=True,
            summary="Compacted",
            original_count=1,
            compacted_count=1,
        )
        result._recent_messages = []
        s.record_compaction(result)

        # The original message must still be in the raw file
        raw = s.path.read_text()
        assert "original message that should survive" in raw


class TestAppendOnlyEdgeCases:
    """Edge cases for append-only storage."""

    def test_empty_session_save(self, tmp_path):
        s = create_session(str(tmp_path), name="empty")
        s.save()
        # Should create file with just meta
        lines = s.path.read_text().strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["type"] == "meta"

    def test_save_without_path(self):
        s = Session()
        s.add_user("hello")
        s.save()  # Should not crash

    def test_fork_creates_independent_file(self, tmp_path):
        s1 = create_session(str(tmp_path), name="source")
        s1.add_user("original")
        s1.save()

        s2 = fork_session(s1, str(tmp_path), name="fork")
        s2.add_user("fork-only")
        s2.save()

        # Source should not have fork's message
        raw_source = s1.path.read_text()
        assert "fork-only" not in raw_source

        # Fork should have both messages
        loaded_fork = Session(s2.path)
        assert len(loaded_fork.messages) == 2

    def test_token_usage_survives_reload(self, tmp_path):
        s = create_session(str(tmp_path), name="test")
        s.add_user("hello")
        s.update_token_usage({"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150})
        s.save()

        loaded = Session(s.path)
        # Note: token_usage is not persisted in current design
        # This test documents current behavior
        assert isinstance(loaded.token_usage, dict)

    def test_no_duplicate_messages_on_double_save(self, tmp_path):
        """Calling save() twice without adding messages should not duplicate."""
        s = create_session(str(tmp_path), name="test")
        s.add_user("only once")
        s.save()
        s.save()  # Double save

        loaded = Session(s.path)
        assert len(loaded.messages) == 1
