"""
Core tools for mini-pi: bash, read, write, edit, grep, find, ls.

7 tools — matching Pi's tool set with simplified implementations.
Key improvements over v1:
- edit supports edits[] for multiple replacements in one call
- Unified truncation (truncate_head / truncate_tail, 2000 lines / 50KB)
- bash uses Popen for streaming output
- read gives continuation hints (offset=NN to continue)
- grep supports context, ignoreCase, literal, glob
- find and ls are new additions
"""

from __future__ import annotations

import fnmatch
import os
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any

from openai import pydantic_function_tool
from pydantic import BaseModel, Field

from .truncate import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
    format_size,
    truncate_head,
    truncate_tail,
    truncate_line,
)


# ── Tool parameter schemas ──────────────────────────────────────────

class BashParams(BaseModel):
    """Run a bash command."""
    command: str = Field(description="The bash command to execute")
    timeout: int | None = Field(default=None, description="Timeout in seconds (optional)")


class ReadParams(BaseModel):
    """Read file contents."""
    path: str = Field(description="Path to the file to read")
    offset: int | None = Field(default=None, description="Line number to start reading from (1-indexed)")
    limit: int | None = Field(default=None, description="Maximum number of lines to read")


class WriteParams(BaseModel):
    """Write content to a file, creating parent directories as needed."""
    path: str = Field(description="Path to the file to write")
    content: str = Field(description="Content to write")


class SingleEdit(BaseModel):
    """A single edit operation."""
    old_text: str = Field(description="Exact text to find. Must be unique in the file.")
    new_text: str = Field(description="Replacement text")


class EditParams(BaseModel):
    """Edit a file with one or more replacements."""
    path: str = Field(description="Path to the file to edit")
    # Support both old_text/new_text (single) and edits[] (multiple)
    old_text: str | None = Field(default=None, description="Deprecated: use edits[] instead")
    new_text: str | None = Field(default=None, description="Deprecated: use edits[] instead")
    edits: list[SingleEdit] | None = Field(default=None, description="One or more replacements")


class GrepParams(BaseModel):
    """Search for a pattern in files."""
    pattern: str = Field(description="The regex pattern to search for")
    path: str | None = Field(default=None, description="Directory or file to search in (default: .)")
    glob: str | None = Field(default=None, description="Filter by glob pattern, e.g. '*.py'")
    ignore_case: bool | None = Field(default=None, description="Case-insensitive search")
    literal: bool | None = Field(default=None, description="Treat pattern as literal string")
    context: int | None = Field(default=None, description="Lines of context before/after match")


class FindParams(BaseModel):
    """Find files by glob pattern."""
    pattern: str = Field(description="Glob pattern, e.g. '*.py', '**/*.json', 'src/**/*.spec.ts'")
    path: str | None = Field(default=None, description="Directory to search in (default: .)")


class LsParams(BaseModel):
    """List directory contents."""
    path: str | None = Field(default=None, description="Directory to list (default: .)")


# ── Tool implementations ────────────────────────────────────────────

def tool_bash(params: BashParams, *, timeout: int = 30, cwd: str = "") -> str:
    """Execute a bash command. Streaming output with tail truncation."""
    try:
        path_obj = Path(cwd) if cwd else Path.cwd()
        if not path_obj.exists():
            return f"Error: Working directory does not exist: {path_obj}"

        timeout_sec = params.timeout or timeout

        proc = subprocess.Popen(
            params.command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(path_obj),
        )

        # Read output in a thread so we can enforce timeout
        output_lines: list[str] = []

        def _read():
            assert proc.stdout is not None
            for line in proc.stdout:
                output_lines.append(line.decode("utf-8", errors="replace"))

        reader = threading.Thread(target=_read, daemon=True)
        reader.start()

        try:
            proc.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            reader.join(timeout=2)
            full = "".join(output_lines)
            result = truncate_tail(full)
            output = result["content"]
            if result["truncated"]:
                output += f"\n\n[Truncated: showing last {result['output_lines']} of {result['total_lines']} lines]"
            output += f"\n\nCommand timed out after {timeout_sec}s"
            return output

        reader.join(timeout=2)
        full = "".join(output_lines)
        result = truncate_tail(full)

        output = result["content"]
        if result["truncated"]:
            output += f"\n\n[Showing last {result['output_lines']} of {result['total_lines']} lines ({format_size(DEFAULT_MAX_BYTES)} limit)]"

        if proc.returncode != 0:
            output += f"\n\nExit code: {proc.returncode}"

        return output.strip() or "(no output)"

    except Exception as e:
        return f"Error: {e}"


def tool_read(params: ReadParams, *, cwd: str = "") -> str:
    """Read file contents with head truncation and continuation hints."""
    path = _resolve(params.path, cwd)
    try:
        if not path.exists():
            return f"Error: File not found: {path}"

        if path.is_dir():
            return _list_dir(path)

        content = path.read_text(encoding="utf-8", errors="replace")
        all_lines = content.splitlines()
        total_lines = len(all_lines)

        # Apply offset (1-indexed)
        start = (params.offset or 1) - 1
        if start >= total_lines:
            return f"Error: Offset {params.offset} is beyond end of file ({total_lines} lines)"

        if params.limit is not None:
            selected_lines = all_lines[start:start + params.limit]
        else:
            selected_lines = all_lines[start:]

        selected_text = "\n".join(selected_lines)

        # Apply truncation
        truncation = truncate_head(selected_text)

        # Build output with line numbers
        output_lines = truncation["content"].splitlines()
        start_display = start + 1
        numbered = []
        for i, line in enumerate(output_lines):
            line_num = start_display + i
            numbered.append(f"{line_num:>6} | {line}")

        header = f"File: {path} ({total_lines} lines)"
        output_parts = [header, *numbered]

        if truncation["truncated"]:
            end_display = start_display + truncation["output_lines"] - 1
            if truncation["first_line_exceeds_limit"]:
                output_parts.append(
                    f"\n[Line {start_display} exceeds {format_size(DEFAULT_MAX_BYTES)} limit. "
                    f"Use bash: sed -n '{start_display}p' {params.path} | head -c {DEFAULT_MAX_BYTES}]"
                )
            else:
                next_offset = end_display + 1
                output_parts.append(
                    f"\n[Showing lines {start_display}-{end_display} of {total_lines}. "
                    f"Use offset={next_offset} to continue.]"
                )
        elif params.limit is not None and start + params.limit < total_lines:
            remaining = total_lines - (start + params.limit)
            next_offset = start + params.limit + 1
            output_parts.append(
                f"\n[{remaining} more lines in file. Use offset={next_offset} to continue.]"
            )

        return "\n".join(output_parts)

    except Exception as e:
        return f"Error reading file: {e}"


def tool_write(params: WriteParams, *, cwd: str = "") -> str:
    """Write content to a file."""
    path = _resolve(params.path, cwd)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(params.content, encoding="utf-8")
        return f"✅ Written {len(params.content)} chars to {path}"
    except Exception as e:
        return f"Error writing file: {e}"


def tool_edit(params: EditParams, *, cwd: str = "") -> str:
    """
    Edit a file with one or more exact text replacements.

    Supports both:
    - old_text + new_text (single replacement, backward compat)
    - edits[] (multiple replacements in one call)
    """
    path = _resolve(params.path, cwd)

    # Normalize to edits list
    edits: list[dict[str, str]] = []
    if params.edits:
        edits = [{"old_text": e.old_text, "new_text": e.new_text} for e in params.edits]
    elif params.old_text is not None and params.new_text is not None:
        edits = [{"old_text": params.old_text, "new_text": params.new_text}]
    else:
        return "Error: Provide either edits[] or old_text + new_text"

    if not edits:
        return "Error: No edits provided"

    try:
        if not path.exists():
            return f"Error: File not found: {path}"

        content = path.read_text(encoding="utf-8")
        original = content

        # Collect all match positions first (all against original content)
        matches: list[tuple[int, int, str]] = []  # (start, end, new_text)

        for i, edit in enumerate(edits):
            old = edit["old_text"]
            new = edit["new_text"]

            if not old:
                label = f"edits[{i}].old_text" if len(edits) > 1 else "old_text"
                return f"Error: {label} must not be empty in {path}"

            idx = content.find(old)
            if idx == -1:
                # Try in original (non-shifted) content
                idx = original.find(old)
                if idx == -1:
                    label = f"edits[{i}]" if len(edits) > 1 else "the text"
                    return (
                        f"Error: Could not find {label} in {path}. "
                        f"The old text must match exactly including whitespace and newlines."
                    )

            # Check uniqueness
            count = original.count(old)
            if count > 1:
                label = f"edits[{i}]" if len(edits) > 1 else "the text"
                return (
                    f"Error: Found {count} occurrences of {label} in {path}. "
                    f"Provide more context to make it unique."
                )

            matches.append((idx, idx + len(old), new))

        # Check for overlaps
        matches.sort(key=lambda m: m[0])
        for i in range(1, len(matches)):
            if matches[i][0] < matches[i - 1][1]:
                return (
                    f"Error: edits[{i-1}] and edits[{i}] overlap in {path}. "
                    f"Merge them into one edit or target disjoint regions."
                )

        # Apply edits in reverse order so offsets stay valid
        result = original
        for start, end, new_text in reversed(matches):
            result = result[:start] + new_text + result[end:]

        if result == original:
            return f"Error: No changes made to {path}. Check that old_text differs from new_text."

        path.write_text(result, encoding="utf-8")
        return f"✅ Edited {path} (replaced {len(matches)} block(s))"

    except Exception as e:
        return f"Error editing file: {e}"


def tool_grep(params: GrepParams, *, timeout: int = 30, cwd: str = "") -> str:
    """Search for a pattern in files using ripgrep."""
    search_path = _resolve(params.path or ".", cwd)
    try:
        if not search_path.exists():
            return f"Error: Path not found: {search_path}"

        if shutil.which("rg"):
            return _grep_rg(params, search_path, timeout)
        else:
            return _grep_fallback(params, search_path, timeout)

    except subprocess.TimeoutExpired:
        return f"Search timed out after {timeout}s"
    except Exception as e:
        return f"Error searching: {e}"


def _grep_rg(params: GrepParams, search_path: Path, timeout: int) -> str:
    """Grep using ripgrep."""
    cmd = ["rg", "--line-number", "--with-filename", "--no-heading", "--color=never"]

    if params.ignore_case:
        cmd.append("--ignore-case")
    if params.literal:
        cmd.append("--fixed-strings")
    if params.glob:
        cmd += ["--glob", params.glob]
    if params.context and params.context > 0:
        cmd += ["--context", str(params.context)]

    cmd += [params.pattern, str(search_path)]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    output = result.stdout.strip()

    if not output:
        if result.returncode == 1:
            return "No matches found."
        if result.stderr:
            return f"Error: {result.stderr.strip()}"
        return "No matches found."

    # Truncate long lines
    lines = output.splitlines()
    truncated = False
    processed = []
    for line in lines:
        t, was_truncated = truncate_line(line)
        if was_truncated:
            truncated = True
        processed.append(t)

    # Apply byte truncation
    full = "\n".join(processed)
    trunc = truncate_head(full)

    final = trunc["content"]
    notices = []
    if trunc["truncated"]:
        notices.append(f"{format_size(DEFAULT_MAX_BYTES)} limit")
    if truncated:
        notices.append("some lines truncated to 500 chars. Use read tool for full lines")
    if notices:
        final += f"\n\n[Truncated: {', '.join(notices)}]"

    return final


def _grep_fallback(params: GrepParams, search_path: Path, timeout: int) -> str:
    """Grep using standard grep (no rg)."""
    cmd = ["grep", "-rn", "--color=never", "-E"]
    if params.ignore_case:
        cmd.append("-i")
    if params.literal:
        cmd.remove("-E")  # Remove regex flag for literal
        cmd.append("-F")
    if params.glob:
        cmd += ["--include", f"*.{params.glob.rstrip('*').rstrip('.')}"]
    cmd += [params.pattern, str(search_path)]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    output = result.stdout.strip()

    if not output:
        return "No matches found."

    lines = output.splitlines()
    if len(lines) > 200:
        return "\n".join(lines[:200]) + f"\n... ({len(lines) - 200} more matches)"

    return output


def tool_find(params: FindParams, *, cwd: str = "") -> str:
    """Find files by glob pattern."""
    search_path = _resolve(params.path or ".", cwd)
    try:
        if not search_path.exists():
            return f"Error: Path not found: {search_path}"
        if not search_path.is_dir():
            return f"Error: Not a directory: {search_path}"

        # Use fd if available (fast, respects .gitignore)
        if shutil.which("fd"):
            return _find_fd(params, search_path)
        else:
            return _find_fallback(params, search_path)

    except Exception as e:
        return f"Error finding files: {e}"


def _find_fd(params: FindParams, search_path: Path) -> str:
    """Find using fd command."""
    cmd = ["fd", "--glob", "--color=never", "--hidden", "--type", "f"]

    # If pattern contains path separators, use full-path matching
    pattern = params.pattern
    if "/" in pattern:
        cmd.append("--full-path")
        if not pattern.startswith("/") and not pattern.startswith("**/"):
            pattern = f"**/{pattern}"

    cmd += [pattern, str(search_path)]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

    if result.returncode != 0 and not result.stdout.strip():
        if result.stderr:
            return f"Error: {result.stderr.strip()}"
        return "No files found matching pattern."

    output = result.stdout.strip()
    if not output:
        return "No files found matching pattern."

    # Make paths relative to search_path
    lines = output.splitlines()
    relative = []
    for line in lines:
        try:
            rel = Path(line).relative_to(search_path)
            relative.append(str(rel))
        except ValueError:
            relative.append(line)

    raw = "\n".join(relative)
    trunc = truncate_head(raw)

    out = trunc["content"]
    if trunc["truncated"]:
        out += f"\n\n[Truncated: {format_size(DEFAULT_MAX_BYTES)} limit]"

    return out


def _find_fallback(params: FindParams, search_path: Path) -> str:
    """Find using os.walk (no fd available)."""
    matches = []
    pattern = params.pattern

    for root, dirs, files in os.walk(search_path):
        # Skip hidden and common ignored dirs
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        dirs[:] = [d for d in dirs if d not in ("node_modules", "__pycache__", ".git")]

        for filename in files:
            if fnmatch.fnmatch(filename, pattern) or fnmatch.fnmatch(filename, f"**/{pattern}"):
                full = Path(root) / filename
                try:
                    rel = full.relative_to(search_path)
                    matches.append(str(rel))
                except ValueError:
                    matches.append(str(full))

        if len(matches) >= 1000:
            break

    if not matches:
        return "No files found matching pattern."

    raw = "\n".join(matches)
    trunc = truncate_head(raw)

    out = trunc["content"]
    if trunc["truncated"]:
        out += f"\n\n[Truncated: {format_size(DEFAULT_MAX_BYTES)} limit]"

    return out


def tool_ls(params: LsParams, *, cwd: str = "") -> str:
    """List directory contents."""
    dir_path = _resolve(params.path or ".", cwd)
    try:
        if not dir_path.exists():
            return f"Error: Path not found: {dir_path}"
        if not dir_path.is_dir():
            return f"Error: Not a directory: {dir_path}"

        return _list_dir(dir_path)

    except Exception as e:
        return f"Error listing directory: {e}"


def _list_dir(dir_path: Path) -> str:
    """Format directory listing."""
    try:
        entries = sorted(dir_path.iterdir(), key=lambda e: e.name.lower())
    except PermissionError:
        return f"Error: Permission denied: {dir_path}"

    lines = []
    for entry in entries[:500]:
        suffix = "/" if entry.is_dir() else ""
        lines.append(f"{entry.name}{suffix}")

    if not lines:
        return "(empty directory)"

    output = "\n".join(lines)
    trunc = truncate_head(output)

    result = trunc["content"]
    if trunc["truncated"]:
        result += f"\n\n[Truncated: {format_size(DEFAULT_MAX_BYTES)} limit]"
    if len(entries) > 500:
        result += f"\n... and {len(entries) - 500} more entries"

    return result


# ── Tool registry ───────────────────────────────────────────────────

TOOL_DEFINITIONS = {
    "bash": (BashParams, tool_bash),
    "read": (ReadParams, tool_read),
    "write": (WriteParams, tool_write),
    "edit": (EditParams, tool_edit),
    "grep": (GrepParams, tool_grep),
    "find": (FindParams, tool_find),
    "ls": (LsParams, tool_ls),
}


def get_openai_tools() -> list[dict]:
    """Get tool definitions in OpenAI function calling format."""
    tools = []
    for name, (params_model, _) in TOOL_DEFINITIONS.items():
        tools.append(pydantic_function_tool(params_model, name=name))
    return tools


def execute_tool(name: str, arguments: dict, *, timeout: int = 30, cwd: str = "") -> str:
    """Execute a tool by name with the given arguments."""
    if name not in TOOL_DEFINITIONS:
        return f"Error: Unknown tool '{name}'"

    params_model, handler = TOOL_DEFINITIONS[name]
    try:
        params = params_model.model_validate(arguments)
        kwargs: dict[str, Any] = {"cwd": cwd}
        if name in ("bash", "grep"):
            kwargs["timeout"] = timeout
        return handler(params, **kwargs)
    except Exception as e:
        return f"Error executing {name}: {e}"


# ── Helpers ─────────────────────────────────────────────────────────

def _resolve(path_str: str, cwd: str) -> Path:
    """Resolve a path relative to cwd."""
    p = Path(path_str)
    if p.is_absolute():
        return p
    return Path(cwd) / p if cwd else p
