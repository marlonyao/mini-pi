"""
Tests for core tools: bash, read, write, edit, grep, find, ls.
"""

import os
import pytest
from pathlib import Path

from mini_pi.tools import (
    execute_tool,
    get_openai_tools,
    tool_bash,
    tool_read,
    tool_write,
    tool_edit,
    tool_grep,
    tool_find,
    tool_ls,
    BashParams,
    ReadParams,
    WriteParams,
    EditParams,
    SingleEdit,
    GrepParams,
    FindParams,
    LsParams,
    TOOL_DEFINITIONS,
)


# ── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def workdir(tmp_path):
    """Create a temp directory with some files for testing."""
    (tmp_path / "hello.py").write_text("print('hello')\nname = 'world'\nprint(name)\n")
    (tmp_path / "data.txt").write_text("line1\nline2\nline3\nline4\nline5\n")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "nested.py").write_text("# nested\n")
    return tmp_path


# ── Registry tests ─────────────────────────────────────────────────

class TestToolRegistry:
    def test_all_7_tools_registered(self):
        assert set(TOOL_DEFINITIONS.keys()) == {"bash", "read", "write", "edit", "grep", "find", "ls"}

    def test_get_openai_tools_returns_7(self):
        tools = get_openai_tools()
        assert len(tools) == 7
        names = {t["function"]["name"] for t in tools}
        assert names == {"bash", "read", "write", "edit", "grep", "find", "ls"}

    def test_execute_unknown_tool(self):
        result = execute_tool("unknown", {}, cwd=".")
        assert "Error" in result
        assert "Unknown" in result


# ── Bash tests ─────────────────────────────────────────────────────

class TestBash:
    def test_simple_command(self, workdir):
        params = BashParams(command="echo hello")
        result = tool_bash(params, cwd=str(workdir))
        assert "hello" in result

    def test_command_with_output(self, workdir):
        params = BashParams(command="ls hello.py")
        result = tool_bash(params, cwd=str(workdir))
        assert "hello.py" in result

    def test_command_error_exit_code(self, workdir):
        params = BashParams(command="exit 1")
        result = tool_bash(params, cwd=str(workdir))
        assert "Exit code: 1" in result

    def test_command_timeout(self, workdir):
        params = BashParams(command="sleep 10", timeout=1)
        result = tool_bash(params, timeout=1, cwd=str(workdir))
        assert "timed out" in result.lower()

    def test_command_stderr(self, workdir):
        params = BashParams(command="echo error >&2")
        result = tool_bash(params, cwd=str(workdir))
        assert "error" in result


# ── Read tests ─────────────────────────────────────────────────────

class TestRead:
    def test_read_file(self, workdir):
        params = ReadParams(path="hello.py")
        result = tool_read(params, cwd=str(workdir))
        assert "print('hello')" in result
        assert "3 lines" in result

    def test_read_with_offset(self, workdir):
        params = ReadParams(path="data.txt", offset=2, limit=2)
        result = tool_read(params, cwd=str(workdir))
        assert "line2" in result
        assert "line3" in result
        assert "line1" not in result

    def test_read_nonexistent_file(self, workdir):
        params = ReadParams(path="nonexistent.txt")
        result = tool_read(params, cwd=str(workdir))
        assert "not found" in result.lower() or "Error" in result

    def test_read_directory_lists_files(self, workdir):
        params = ReadParams(path=".")
        result = tool_read(params, cwd=str(workdir))
        assert "hello.py" in result

    def test_read_continuation_hint(self, workdir):
        """When limit stops early, should show continuation hint."""
        params = ReadParams(path="data.txt", limit=2)
        result = tool_read(params, cwd=str(workdir))
        assert "offset=" in result

    def test_read_absolute_path(self, workdir):
        params = ReadParams(path=str(workdir / "hello.py"))
        result = tool_read(params, cwd="/tmp")
        assert "print('hello')" in result


# ── Write tests ────────────────────────────────────────────────────

class TestWrite:
    def test_write_new_file(self, workdir):
        params = WriteParams(path="new.txt", content="hello world")
        result = tool_write(params, cwd=str(workdir))
        assert "✅" in result
        assert (workdir / "new.txt").read_text() == "hello world"

    def test_write_creates_parent_dirs(self, workdir):
        params = WriteParams(path="deep/nested/dir/file.txt", content="deep")
        result = tool_write(params, cwd=str(workdir))
        assert "✅" in result
        assert (workdir / "deep/nested/dir/file.txt").read_text() == "deep"

    def test_write_overwrites(self, workdir):
        params = WriteParams(path="hello.py", content="replaced")
        tool_write(params, cwd=str(workdir))
        assert (workdir / "hello.py").read_text() == "replaced"


# ── Edit tests ─────────────────────────────────────────────────────

class TestEdit:
    def test_single_edit_old_new_text(self, workdir):
        params = EditParams(
            path="hello.py",
            old_text="print('hello')",
            new_text="print('hi')",
        )
        result = tool_edit(params, cwd=str(workdir))
        assert "✅" in result
        content = (workdir / "hello.py").read_text()
        assert "print('hi')" in content
        assert "print('hello')" not in content

    def test_edit_with_edits_array(self, workdir):
        """Multiple edits in one call — the key Pi feature."""
        params = EditParams(
            path="hello.py",
            edits=[
                SingleEdit(old_text="print('hello')", new_text="print('greetings')"),
                SingleEdit(old_text="'world'", new_text="'universe'"),
            ],
        )
        result = tool_edit(params, cwd=str(workdir))
        assert "✅" in result
        assert "2 block(s)" in result
        content = (workdir / "hello.py").read_text()
        assert "print('greetings')" in content
        assert "'universe'" in content
        # Original should be gone
        assert "print('hello')" not in content

    def test_edit_not_found(self, workdir):
        params = EditParams(
            path="hello.py",
            old_text="nonexistent text",
            new_text="replacement",
        )
        result = tool_edit(params, cwd=str(workdir))
        assert "Error" in result
        assert "not find" in result.lower()

    def test_edit_multiple_occurrences(self, workdir):
        """If old_text appears multiple times, should error."""
        (workdir / "dup.py").write_text("x = 1\nx = 1\n")
        params = EditParams(
            path="dup.py",
            old_text="x = 1",
            new_text="x = 2",
        )
        result = tool_edit(params, cwd=str(workdir))
        assert "Error" in result
        assert "2 occurrences" in result

    def test_edit_overlapping_edits(self, workdir):
        """Overlapping edits should be rejected."""
        (workdir / "overlap.py").write_text("abcdefghij\n")
        params = EditParams(
            path="overlap.py",
            edits=[
                SingleEdit(old_text="abcde", new_text="12345"),
                SingleEdit(old_text="cdefg", new_text="67890"),
            ],
        )
        result = tool_edit(params, cwd=str(workdir))
        assert "Error" in result or "overlap" in result.lower()

    def test_edit_file_not_found(self, workdir):
        params = EditParams(path="nope.py", old_text="x", new_text="y")
        result = tool_edit(params, cwd=str(workdir))
        assert "not found" in result.lower()

    def test_edit_no_changes(self, workdir):
        params = EditParams(
            path="hello.py",
            old_text="print('hello')",
            new_text="print('hello')",
        )
        result = tool_edit(params, cwd=str(workdir))
        assert "No changes" in result

    def test_edit_empty_old_text(self, workdir):
        params = EditParams(
            path="hello.py",
            old_text="",
            new_text="something",
        )
        result = tool_edit(params, cwd=str(workdir))
        assert "Error" in result

    def test_edit_no_edits_provided(self, workdir):
        params = EditParams(path="hello.py")
        result = tool_edit(params, cwd=str(workdir))
        assert "Error" in result


# ── Grep tests ─────────────────────────────────────────────────────

class TestGrep:
    def test_grep_finds_pattern(self, workdir):
        params = GrepParams(pattern="print", path=".")
        result = tool_grep(params, cwd=str(workdir))
        assert "print" in result
        assert "hello.py" in result

    def test_grep_no_match(self, workdir):
        params = GrepParams(pattern="zzzzz_nonexistent", path=".")
        result = tool_grep(params, cwd=str(workdir))
        assert "No matches" in result

    def test_grep_literal(self, workdir):
        (workdir / "regex.py").write_text("hello (world)\nhello world\n")
        params = GrepParams(pattern="(world)", literal=True, path=".")
        result = tool_grep(params, cwd=str(workdir))
        assert "(world)" in result

    def test_grep_ignore_case(self, workdir):
        (workdir / "case.py").write_text("HELLO\n")
        params = GrepParams(pattern="hello", ignore_case=True, path=".")
        result = tool_grep(params, cwd=str(workdir))
        assert "HELLO" in result


# ── Find tests ─────────────────────────────────────────────────────

class TestFind:
    def test_find_by_extension(self, workdir):
        params = FindParams(pattern="*.py")
        result = tool_find(params, cwd=str(workdir))
        assert "hello.py" in result

    def test_find_nested(self, workdir):
        params = FindParams(pattern="*.py")
        result = tool_find(params, cwd=str(workdir))
        assert "nested.py" in result or "sub" in result

    def test_find_no_match(self, workdir):
        params = FindParams(pattern="*.zzzzz")
        result = tool_find(params, cwd=str(workdir))
        assert "No files" in result

    def test_find_nonexistent_path(self, workdir):
        params = FindParams(pattern="*", path="nonexistent")
        result = tool_find(params, cwd=str(workdir))
        assert "Error" in result


# ── Ls tests ───────────────────────────────────────────────────────

class TestLs:
    def test_ls_directory(self, workdir):
        params = LsParams(path=".")
        result = tool_ls(params, cwd=str(workdir))
        assert "hello.py" in result
        assert "sub/" in result

    def test_ls_empty_directory(self, workdir):
        empty = workdir / "empty"
        empty.mkdir()
        params = LsParams(path="empty")
        result = tool_ls(params, cwd=str(workdir))
        assert "empty" in result.lower()

    def test_ls_nonexistent(self, workdir):
        params = LsParams(path="nope")
        result = tool_ls(params, cwd=str(workdir))
        assert "Error" in result

    def test_ls_file_not_dir(self, workdir):
        params = LsParams(path="hello.py")
        result = tool_ls(params, cwd=str(workdir))
        assert "Not a directory" in result


# ── Execute tool integration ───────────────────────────────────────

class TestExecuteTool:
    def test_execute_bash(self, workdir):
        result = execute_tool("bash", {"command": "echo test"}, cwd=str(workdir))
        assert "test" in result

    def test_execute_read(self, workdir):
        result = execute_tool("read", {"path": "hello.py"}, cwd=str(workdir))
        assert "print" in result

    def test_execute_edit_with_edits_array(self, workdir):
        result = execute_tool("edit", {
            "path": "hello.py",
            "edits": [
                {"old_text": "print('hello')", "new_text": "print('bonjour')"},
                {"old_text": "'world'", "new_text": "'monde'"},
            ],
        }, cwd=str(workdir))
        assert "✅" in result
        content = (workdir / "hello.py").read_text()
        assert "bonjour" in content
        assert "monde" in content
