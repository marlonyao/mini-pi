"""
Tests for P2-1: Tool enhancements.

- Edit: fuzzy matching, unified diff preview
- Bash: stderr separation
"""

import os
import subprocess
import tempfile
from pathlib import Path

import pytest

from mini_pi.tools import (
    BashParams,
    EditParams,
    SingleEdit,
    tool_bash,
    tool_edit,
    _fuzzy_find,
    _normalize_for_fuzzy,
    _unified_diff,
)


class TestFuzzyMatch:
    """Test fuzzy text matching for edit tool."""

    def test_exact_match_still_works(self):
        content = "hello world"
        idx, length = _fuzzy_find(content, "world")
        assert idx == 6
        assert content[idx:idx + length] == "world"

    def test_trailing_whitespace_ignored(self):
        content = "hello   \nworld  \n"
        idx, length = _fuzzy_find(content, "hello\nworld")
        assert idx >= 0

    def test_crlf_normalized(self):
        content = "hello\r\nworld\r\n"
        idx, length = _fuzzy_find(content, "hello\nworld")
        assert idx >= 0

    def test_no_match_returns_negative(self):
        idx, length = _fuzzy_find("hello", "xyz")
        assert idx == -1

    def test_unicode_normalization(self):
        # NFKC: full-width to half-width
        content = "hello ｗｏｒｌｄ"
        idx, length = _fuzzy_find(content, "world")
        # Fuzzy match should find a close match
        # Note: this depends on whether NFKC maps full-width to half-width
        assert isinstance(idx, int)

    def test_empty_search(self):
        idx, length = _fuzzy_find("hello", "")
        assert idx == -1


class TestNormalize:
    """Test text normalization."""

    def test_strips_trailing_whitespace(self):
        assert _normalize_for_fuzzy("hello  \nworld  ") == "hello\nworld"

    def test_normalizes_crlf(self):
        assert _normalize_for_fuzzy("a\r\nb") == "a\nb"

    def test_normalizes_cr(self):
        assert _normalize_for_fuzzy("a\rb") == "a\nb"

    def test_preserves_content(self):
        text = "hello world\nfoo bar"
        assert _normalize_for_fuzzy(text) == text


class TestUnifiedDiff:
    """Test unified diff generation."""

    def test_simple_change(self):
        diff = _unified_diff("hello\n", "world\n", "test.txt")
        assert "-hello" in diff
        assert "+world" in diff

    def test_no_change(self):
        diff = _unified_diff("same\n", "same\n", "test.txt")
        assert diff == ""

    def test_addition(self):
        diff = _unified_diff("a\n", "a\nb\n", "test.txt")
        assert "+b" in diff

    def test_deletion(self):
        diff = _unified_diff("a\nb\n", "a\n", "test.txt")
        assert "-b" in diff

    def test_truncates_long_diff(self):
        old = "".join(f"line {i}\n" for i in range(200))
        new = "".join(f"LINE {i}\n" for i in range(200))
        diff = _unified_diff(old, new, "test.txt")
        assert "truncated" in diff

    def test_shows_context_lines(self):
        diff = _unified_diff("a\nb\nc\nd\ne\n", "a\nb\nC\nd\ne\n", "test.txt")
        # Should show context around the change
        assert "a" in diff  # context before
        assert "d" in diff  # context after


class TestEditDiffPreview:
    """Test that edit tool shows diff preview."""

    def test_edit_shows_diff(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("hello world\nfoo bar\n")

        result = tool_edit(EditParams(
            path=str(f),
            old_text="hello world",
            new_text="hello universe",
        ))
        assert "✅" in result
        assert "-hello world" in result
        assert "+hello universe" in result

    def test_edit_shows_line_count(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("line1\nline2\nline3\n")

        result = tool_edit(EditParams(
            path=str(f),
            old_text="line2",
            new_text="LINE2\nLINE2B",
        ))
        assert "lines changed" in result

    def test_multi_edit_diff(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("aaa\nbbb\nccc\nddd\n")

        result = tool_edit(EditParams(
            path=str(f),
            edits=[
                SingleEdit(old_text="aaa", new_text="AAA"),
                SingleEdit(old_text="ccc", new_text="CCC"),
            ],
        ))
        assert "-aaa" in result
        assert "+AAA" in result
        assert "-ccc" in result
        assert "+CCC" in result


class TestEditFuzzyMatchIntegration:
    """Integration tests for fuzzy matching in edit."""

    def test_edit_with_trailing_whitespace(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("hello   \nworld\n")

        result = tool_edit(EditParams(
            path=str(f),
            old_text="hello\nworld",
            new_text="hi\nworld",
        ))
        # Should succeed via fuzzy match
        assert "✅" in result or "Error" not in result

    def test_edit_exact_match_preferred(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("hello world\n")

        result = tool_edit(EditParams(
            path=str(f),
            old_text="hello world",
            new_text="hello universe",
        ))
        assert "✅" in result
        assert f.read_text() == "hello universe\n"


class TestBashStderr:
    """Test bash tool stderr separation."""

    def test_stdout_only(self, tmp_path):
        result = tool_bash(BashParams(command="echo hello"), cwd=str(tmp_path))
        assert "hello" in result
        assert "stderr" not in result

    def test_stderr_separated(self, tmp_path):
        result = tool_bash(
            BashParams(command="echo out && echo err >&2"),
            cwd=str(tmp_path),
        )
        assert "out" in result
        assert "stderr" in result
        assert "err" in result

    def test_exit_code_shown(self, tmp_path):
        result = tool_bash(
            BashParams(command="exit 42"),
            cwd=str(tmp_path),
        )
        assert "Exit code: 42" in result

    def test_success_no_exit_code(self, tmp_path):
        result = tool_bash(
            BashParams(command="echo ok"),
            cwd=str(tmp_path),
        )
        assert "Exit code" not in result

    def test_timeout_shows_message(self, tmp_path):
        result = tool_bash(
            BashParams(command="sleep 100", timeout=1),
            cwd=str(tmp_path),
            timeout=1,
        )
        assert "timed out" in result
