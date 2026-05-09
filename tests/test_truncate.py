"""
Tests for truncation utilities.
"""

import pytest

from mini_pi.truncate import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
    format_size,
    truncate_head,
    truncate_tail,
    truncate_line,
)


class TestFormatSize:
    def test_bytes(self):
        assert format_size(500) == "500B"

    def test_kilobytes(self):
        assert format_size(2048) == "2.0KB"

    def test_megabytes(self):
        assert format_size(1024 * 1024) == "1.0MB"


class TestTruncateHead:
    def test_no_truncation_needed(self):
        result = truncate_head("hello\nworld")
        assert result["truncated"] is False
        assert result["content"] == "hello\nworld"
        assert result["total_lines"] == 2

    def test_truncation_by_lines(self):
        content = "\n".join(f"line {i}" for i in range(3000))
        result = truncate_head(content, max_lines=100, max_bytes=10 * 1024 * 1024)
        assert result["truncated"] is True
        assert result["truncated_by"] == "lines"
        assert result["output_lines"] == 100
        assert result["total_lines"] == 3000

    def test_truncation_by_bytes(self):
        content = "x" * 1000 + "\n" + "y" * 1000
        result = truncate_head(content, max_lines=100, max_bytes=500)
        assert result["truncated"] is True
        assert result["truncated_by"] == "bytes"

    def test_first_line_exceeds_limit(self):
        content = "x" * 10000
        result = truncate_head(content, max_bytes=500)
        assert result["truncated"] is True
        assert result["first_line_exceeds_limit"] is True
        assert result["content"] == ""

    def test_empty_content(self):
        result = truncate_head("")
        assert result["truncated"] is False
        assert result["total_lines"] == 1  # empty string splits to ['']

    def test_single_line_no_truncation(self):
        result = truncate_head("hello")
        assert result["truncated"] is False
        assert result["content"] == "hello"


class TestTruncateTail:
    def test_no_truncation_needed(self):
        result = truncate_tail("hello\nworld")
        assert result["truncated"] is False
        assert result["content"] == "hello\nworld"

    def test_keeps_end(self):
        lines = [f"line {i}" for i in range(100)]
        content = "\n".join(lines)
        result = truncate_tail(content, max_lines=10, max_bytes=10 * 1024 * 1024)
        assert result["truncated"] is True
        assert "line 99" in result["content"]
        assert "line 0" not in result["content"]

    def test_truncation_by_bytes(self):
        content = "x" * 1000 + "\n" + "y" * 1000
        result = truncate_tail(content, max_lines=100, max_bytes=500)
        assert result["truncated"] is True
        # Should keep the end
        assert "y" in result["content"]


class TestTruncateLine:
    def test_short_line(self):
        text, truncated = truncate_line("hello", 100)
        assert text == "hello"
        assert truncated is False

    def test_long_line(self):
        text, truncated = truncate_line("x" * 1000, 500)
        assert truncated is True
        assert text.endswith("... [truncated]")
        assert len(text) < 600
