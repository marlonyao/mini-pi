"""
Tests for P0 features: System Prompt Enhancement and Session Management.
"""

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from mini_pi.session import Session, create_session, fork_session, list_sessions
from mini_pi.system_prompt import (
    CONTEXT_FILENAMES,
    MAX_CONTEXT_FILE_SIZE,
    build_system_prompt,
    discover_context_files,
)


# ── System Prompt Tests ─────────────────────────────────────────────


class TestDiscoverContextFiles:
    """Test auto-discovery of project context files."""

    def test_discovers_agents_md(self, tmp_path):
        (tmp_path / "AGENTS.md").write_text("# My Agent\nDo stuff")
        results = discover_context_files(str(tmp_path))
        assert len(results) == 1
        assert results[0][0] == "AGENTS.md"
        assert "My Agent" in results[0][1]

    def test_discovers_readme(self, tmp_path):
        (tmp_path / "README.md").write_text("# Project\nHello world")
        results = discover_context_files(str(tmp_path))
        assert len(results) == 1
        assert results[0][0] == "README.md"

    def test_discovers_multiple_files(self, tmp_path):
        (tmp_path / "AGENTS.md").write_text("agent rules")
        (tmp_path / "README.md").write_text("project readme")
        results = discover_context_files(str(tmp_path))
        names = [r[0] for r in results]
        assert "AGENTS.md" in names
        assert "README.md" in names
        # AGENTS.md should come first (higher priority)
        assert names.index("AGENTS.md") < names.index("README.md")

    def test_agents_md_priority_over_readme(self, tmp_path):
        (tmp_path / "AGENTS.md").write_text("agent")
        (tmp_path / "README.md").write_text("readme")
        results = discover_context_files(str(tmp_path))
        assert results[0][0] == "AGENTS.md"

    def test_skips_empty_files(self, tmp_path):
        (tmp_path / "AGENTS.md").write_text("")
        results = discover_context_files(str(tmp_path))
        assert len(results) == 0

    def test_skips_oversized_files(self, tmp_path):
        (tmp_path / "AGENTS.md").write_text("x" * (MAX_CONTEXT_FILE_SIZE + 1))
        results = discover_context_files(str(tmp_path))
        assert len(results) == 0

    def test_skips_nonexistent_dir(self):
        results = discover_context_files("/nonexistent/path/12345")
        assert len(results) == 0

    def test_skips_binary_files(self, tmp_path):
        (tmp_path / "AGENTS.md").write_bytes(b"\x00\x01\x02\xff\xfe")
        results = discover_context_files(str(tmp_path))
        # Binary content may or may not decode, just ensure no crash
        assert isinstance(results, list)

    def test_discovers_system_md(self, tmp_path):
        (tmp_path / "SYSTEM.md").write_text("System rules")
        results = discover_context_files(str(tmp_path))
        assert any(r[0] == "SYSTEM.md" for r in results)

    def test_discovers_claude_md(self, tmp_path):
        (tmp_path / "CLAUDE.md").write_text("Claude rules")
        results = discover_context_files(str(tmp_path))
        assert any(r[0] == "CLAUDE.md" for r in results)

    def test_discovers_cursorrules(self, tmp_path):
        (tmp_path / ".cursorrules").write_text("Cursor rules")
        results = discover_context_files(str(tmp_path))
        assert any(r[0] == ".cursorrules" for r in results)


class TestBuildSystemPrompt:
    """Test system prompt construction."""

    def test_includes_date(self):
        prompt = build_system_prompt(cwd="/tmp")
        today = datetime.now().strftime("%Y-%m-%d")
        assert today in prompt

    def test_includes_weekday(self):
        prompt = build_system_prompt(cwd="/tmp")
        weekday = datetime.now().strftime("%A")
        assert weekday in prompt

    def test_includes_cwd(self):
        prompt = build_system_prompt(cwd="/home/user/project")
        assert "/home/user/project" in prompt

    def test_includes_tools_section(self):
        prompt = build_system_prompt(cwd="/tmp")
        assert "Available Tools" in prompt
        assert "bash" in prompt
        assert "read" in prompt
        assert "edit" in prompt
        assert "write" in prompt

    def test_includes_guidelines(self):
        prompt = build_system_prompt(cwd="/tmp")
        assert "Guidelines" in prompt

    def test_includes_context_files(self):
        prompt = build_system_prompt(
            cwd="/tmp",
            context_files={"AGENTS.md": "# Rules\nBe helpful"},
            discover_files=False,
        )
        assert "Project Context" in prompt
        assert "AGENTS.md" in prompt
        assert "Be helpful" in prompt

    def test_auto_discovers_files(self, tmp_path):
        (tmp_path / "AGENTS.md").write_text("# Auto discovered")
        prompt = build_system_prompt(cwd=str(tmp_path))
        assert "Auto discovered" in prompt

    def test_explicit_overrides_discovered(self, tmp_path):
        (tmp_path / "AGENTS.md").write_text("Original")
        prompt = build_system_prompt(
            cwd=str(tmp_path),
            context_files={"AGENTS.md": "Override"},
        )
        assert "Override" in prompt

    def test_append_text(self):
        prompt = build_system_prompt(cwd="/tmp", append="Custom appendix")
        assert "Custom appendix" in prompt

    def test_custom_tool_snippets(self):
        snippets = {"bash": "Run commands", "read": "Read files"}
        prompt = build_system_prompt(cwd="/tmp", tool_snippets=snippets)
        assert "Run commands" in prompt
        assert "Read files" in prompt

    def test_no_discover_flag(self, tmp_path):
        (tmp_path / "AGENTS.md").write_text("Should not appear")
        prompt = build_system_prompt(cwd=str(tmp_path), discover_files=False)
        assert "Should not appear" not in prompt


# ── Session Management Tests ────────────────────────────────────────


class TestSessionManagement:
    """Test session CRUD operations."""

    def test_create_session(self, tmp_path):
        session = create_session(str(tmp_path))
        assert session.path is not None
        assert session.path.parent == tmp_path
        assert session.path.suffix == ".jsonl"

    def test_create_session_with_name(self, tmp_path):
        session = create_session(str(tmp_path), name="my-session")
        assert session.path.stem == "my-session"

    def test_list_sessions_empty(self, tmp_path):
        sessions = list_sessions(str(tmp_path))
        assert sessions == []

    def test_list_sessions_finds_jsonl(self, tmp_path):
        (tmp_path / "test.jsonl").write_text('{"type":"meta","created_at":"2024-01-01"}\n')
        sessions = list_sessions(str(tmp_path))
        assert len(sessions) == 1
        assert sessions[0]["name"] == "test"

    def test_list_sessions_sorted_newest_first(self, tmp_path):
        # Names with timestamps should sort newest first
        (tmp_path / "2024-01-01_10-00-00.jsonl").write_text('{"type":"meta"}\n')
        (tmp_path / "2024-06-15_14-30-00.jsonl").write_text('{"type":"meta"}\n')
        sessions = list_sessions(str(tmp_path))
        assert sessions[0]["name"] == "2024-06-15_14-30-00"  # newest first

    def test_list_sessions_nonexistent_dir(self):
        sessions = list_sessions("/nonexistent/path")
        assert sessions == []


class TestSessionFork:
    """Test session forking."""

    def test_fork_copies_messages(self, tmp_path):
        source = create_session(str(tmp_path), name="source")
        source.add_user("hello")
        source.add_assistant("hi there")
        source.add_user("how are you")
        source.save()

        forked = fork_session(source, str(tmp_path), name="fork-test")
        assert len(forked.messages) == 3
        assert forked.messages[0]["content"] == "hello"
        assert forked.messages[1]["content"] == "hi there"
        assert forked.path != source.path

    def test_fork_preserves_independence(self, tmp_path):
        source = create_session(str(tmp_path), name="source")
        source.add_user("original message")
        source.save()

        forked = fork_session(source, str(tmp_path), name="forked")
        forked.add_user("fork only message")
        forked.save()

        # Source should not have the fork's new message
        assert len(source.messages) == 1
        assert len(forked.messages) == 2

    def test_fork_auto_names(self, tmp_path):
        source = create_session(str(tmp_path), name="source")
        source.add_user("test")
        source.save()

        forked = fork_session(source, str(tmp_path))
        assert forked.path.stem.startswith("202")  # auto-timestamp

    def test_fork_copies_token_usage(self, tmp_path):
        source = create_session(str(tmp_path), name="source")
        source.add_user("hello")
        source.update_token_usage({"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150})
        source.save()

        forked = fork_session(source, str(tmp_path), name="forked")
        assert forked.token_usage["total"] == 150

    def test_fork_persists_to_disk(self, tmp_path):
        source = create_session(str(tmp_path), name="source")
        source.add_user("persist test")
        source.save()

        forked = fork_session(source, str(tmp_path), name="fork-persist")
        assert forked.path.exists()

        # Load it back
        loaded = Session(forked.path)
        assert len(loaded.messages) == 1
        assert loaded.messages[0]["content"] == "persist test"

    def test_fork_with_tool_calls(self, tmp_path):
        source = create_session(str(tmp_path), name="source")
        source.add_user("list files")
        source.add_assistant(
            content=None,
            tool_calls=[{"id": "tc1", "type": "function", "function": {"name": "bash", "arguments": '{"command":"ls"}'}}]
        )
        source.add_tool_result("tc1", "file1.py\nfile2.py")
        source.save()

        forked = fork_session(source, str(tmp_path), name="forked")
        assert len(forked.messages) == 3
        assert forked.messages[1]["tool_calls"] is not None

    def test_fork_empty_session(self, tmp_path):
        source = create_session(str(tmp_path), name="source")
        source.save()

        forked = fork_session(source, str(tmp_path), name="forked")
        assert len(forked.messages) == 0
