"""
Shared truncation utilities for tool outputs.

Two strategies:
- truncate_head: Keep the beginning (for read, ls, find)
- truncate_tail: Keep the end (for bash — errors/results at the bottom)

Limits: 2000 lines OR 50KB, whichever is hit first.
Never returns partial lines.
"""

from __future__ import annotations


DEFAULT_MAX_LINES = 2000
DEFAULT_MAX_BYTES = 50 * 1024  # 50KB


def format_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes}B"
    elif num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f}KB"
    else:
        return f"{num_bytes / (1024 * 1024):.1f}MB"


def truncate_head(
    content: str,
    *,
    max_lines: int = DEFAULT_MAX_LINES,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> dict:
    """
    Truncate content from the head — keep the beginning.

    Returns dict:
        content: str — the truncated content
        truncated: bool — whether truncation occurred
        truncated_by: str | None — "lines" or "bytes"
        total_lines: int
        output_lines: int
        first_line_exceeds_limit: bool
    """
    lines = content.split("\n")
    total_lines = len(lines)
    total_bytes = len(content.encode("utf-8"))

    if total_lines <= max_lines and total_bytes <= max_bytes:
        return {
            "content": content,
            "truncated": False,
            "truncated_by": None,
            "total_lines": total_lines,
            "output_lines": total_lines,
            "first_line_exceeds_limit": False,
        }

    # Check if first line alone exceeds byte limit
    first_line_bytes = len(lines[0].encode("utf-8"))
    if first_line_bytes > max_bytes:
        return {
            "content": "",
            "truncated": True,
            "truncated_by": "bytes",
            "total_lines": total_lines,
            "output_lines": 0,
            "first_line_exceeds_limit": True,
        }

    # Collect complete lines that fit
    output_lines = []
    output_bytes = 0
    truncated_by = "lines"

    for i, line in enumerate(lines):
        line_bytes = len(line.encode("utf-8")) + (1 if i > 0 else 0)
        if output_bytes + line_bytes > max_bytes:
            truncated_by = "bytes"
            break
        if len(output_lines) >= max_lines:
            truncated_by = "lines"
            break
        output_lines.append(line)
        output_bytes += line_bytes

    return {
        "content": "\n".join(output_lines),
        "truncated": True,
        "truncated_by": truncated_by,
        "total_lines": total_lines,
        "output_lines": len(output_lines),
        "first_line_exceeds_limit": False,
    }


def truncate_tail(
    content: str,
    *,
    max_lines: int = DEFAULT_MAX_LINES,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> dict:
    """
    Truncate content from the tail — keep the end.

    Suitable for bash output where errors/results are at the bottom.
    """
    lines = content.split("\n")
    total_lines = len(lines)
    total_bytes = len(content.encode("utf-8"))

    if total_lines <= max_lines and total_bytes <= max_bytes:
        return {
            "content": content,
            "truncated": False,
            "truncated_by": None,
            "total_lines": total_lines,
            "output_lines": total_lines,
        }

    # Work backwards from the end
    output_lines = []
    output_bytes = 0

    for line in reversed(lines):
        line_bytes = len(line.encode("utf-8")) + (1 if output_lines else 0)
        if output_bytes + line_bytes > max_bytes and output_lines:
            break
        if len(output_lines) >= max_lines:
            break
        output_lines.insert(0, line)
        output_bytes += line_bytes

    # If we couldn't fit even one line, take the tail of the last line
    if not output_lines and lines:
        last = lines[-1].encode("utf-8")
        if len(last) > max_bytes:
            output_lines.append(last[-max_bytes:].decode("utf-8", errors="replace"))

    return {
        "content": "\n".join(output_lines),
        "truncated": True,
        "truncated_by": "bytes" if len(output_lines) < min(total_lines, max_lines) else "lines",
        "total_lines": total_lines,
        "output_lines": len(output_lines),
    }


def truncate_line(line: str, max_chars: int = 500) -> tuple[str, bool]:
    """Truncate a single line to max_chars, adding [truncated] suffix."""
    if len(line) <= max_chars:
        return line, False
    return line[:max_chars] + "... [truncated]", True
