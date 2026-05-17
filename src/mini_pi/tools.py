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
            stderr=subprocess.PIPE,
            cwd=str(path_obj),
        )

        # Read output in a thread so we can enforce timeout
        stdout_lines: list[str] = []
        stderr_lines: list[str] = []

        def _read():
            assert proc.stdout is not None
            assert proc.stderr is not None
            for line in proc.stdout:
                stdout_lines.append(line.decode("utf-8", errors="replace"))
            for line in proc.stderr:
                stderr_lines.append(line.decode("utf-8", errors="replace"))

        reader = threading.Thread(target=_read, daemon=True)
        reader.start()

        try:
            proc.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            reader.join(timeout=2)
            full = "".join(stdout_lines)
            result = truncate_tail(full)
            output = result["content"]
            if result["truncated"]:
                output += f"\n\n[Truncated: showing last {result['output_lines']} of {result['total_lines']} lines]"
            output += f"\n\nCommand timed out after {timeout_sec}s"
            return output

        reader.join(timeout=2)

        # Build output
        stdout_full = "".join(stdout_lines)
        stderr_full = "".join(stderr_lines)

        result = truncate_tail(stdout_full)
        output = result["content"]
        if result["truncated"]:
            output += f"\n\n[Showing last {result['output_lines']} of {result['total_lines']} lines ({format_size(DEFAULT_MAX_BYTES)} limit)]"

        # Append stderr if present
        if stderr_full.strip():
            stderr_trunc = truncate_tail(stderr_full)
            stderr_output = stderr_trunc["content"]
            if stderr_trunc["truncated"]:
                stderr_output += "\n[stderr truncated]"
            output += f"\n\n--- stderr ---\n{stderr_output}"

        if proc.returncode != 0:
            output += f"\n\nExit code: {proc.returncode}"

        return output.strip() or "(no output)"

    except Exception as e:
        return f"Error: {e}"


def tool_read(params: ReadParams, *, cwd: str = "") -> str:
    """Read file contents with truncation and continuation hints (pi-style)."""
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

        # Output raw content without line numbers (pi-style)
        output_parts = [truncation["content"]]

        if truncation["truncated"]:
            end_display = start + 1 + truncation["output_lines"] - 1
            if truncation["first_line_exceeds_limit"]:
                output_parts.append(
                    f"[Line {start + 1} exceeds {format_size(DEFAULT_MAX_BYTES)} limit. "
                    f"Use bash: sed -n '{start + 1}p' {params.path} | head -c {DEFAULT_MAX_BYTES}]"
                )
            else:
                next_offset = end_display + 1
                output_parts.append(
                    f"[Output truncated to {DEFAULT_MAX_LINES} lines or {format_size(DEFAULT_MAX_BYTES)}. "
                    f"Use offset={next_offset} to continue reading the remaining lines.]"
                )
        elif params.limit is not None and start + params.limit < total_lines:
            remaining = total_lines - (start + params.limit)
            next_offset = start + params.limit + 1
            output_parts.append(
                f"[{remaining} more lines in file. Use offset={next_offset} to continue.]"
            )

        return "\n\n".join(output_parts)

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

    Features:
    - Fuzzy matching: normalizes whitespace/unicode for close matches
    - Diff preview: shows unified diff of changes
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

            # Try exact match first
            idx = original.find(old)
            used_fuzzy = False

            if idx == -1:
                # Try fuzzy match: normalize whitespace/unicode
                idx, match_len = _fuzzy_find(original, old)
                if idx >= 0:
                    used_fuzzy = True
                    matches.append((idx, idx + match_len, new))
                    continue

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

        # Generate unified diff
        diff = _unified_diff(original, result, str(path))

        # Count changes
        changed_lines = sum(1 for l in diff.splitlines() if l.startswith('+') and not l.startswith('+++'))

        path.write_text(result, encoding="utf-8")

        output = f"✅ Edited {path} (replaced {len(matches)} block(s), ~{changed_lines} lines changed)"
        if diff:
            output += "\n" + diff
        return output

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


# ── Edit helpers ────────────────────────────────────────────────────

def _normalize_for_fuzzy(text: str) -> str:
    """Normalize text for fuzzy matching."""
    import unicodedata

    # NFKC normalization (compatibility decomposition)
    text = unicodedata.normalize("NFKC", text)

    # Strip trailing whitespace per line
    text = "\n".join(line.rstrip() for line in text.splitlines())

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    return text


def _fuzzy_find(content: str, search: str) -> tuple[int, int]:
    """
    Try to find search text in content using fuzzy matching.

    Strategies (in order):
    1. Exact match (fast path)
    2. CRLF → LF normalization
    3. Line-by-line trailing whitespace strip + CRLF normalization

    Returns (index, length) in original content, or (-1, 0) if not found.
    """
    if not search:
        return -1, 0

    # Strategy 1: Exact match
    idx = content.find(search)
    if idx >= 0:
        return idx, len(search)

    # Strategy 2: CRLF normalization
    norm_content = content.replace("\r\n", "\n").replace("\r", "\n")
    norm_search = search.replace("\r\n", "\n").replace("\r", "\n")
    idx = norm_content.find(norm_search)
    if idx >= 0:
        # Map back: count how many \r were before this position
        orig_idx = _map_norm_to_orig(content, idx)
        orig_end = _map_norm_to_orig(content, idx + len(norm_search))
        return orig_idx, orig_end - orig_idx

    # Strategy 3: Strip trailing whitespace per line
    stripped_content = "\n".join(line.rstrip() for line in content.splitlines())
    stripped_search = "\n".join(line.rstrip() for line in search.splitlines())
    stripped_norm_content = stripped_content.replace("\r\n", "\n").replace("\r", "\n")
    stripped_norm_search = stripped_search.replace("\r\n", "\n").replace("\r", "\n")

    sidx = stripped_norm_content.find(stripped_norm_search)
    if sidx == -1:
        return -1, 0

    # Approximate mapping: find the matching region in original content
    # by comparing line by line
    return _fuzzy_find_by_lines(content, search)


def _map_norm_to_orig(content: str, norm_pos: int) -> int:
    """Map a position in CRLF-normalized content back to original content."""
    orig_pos = 0
    count = 0
    while orig_pos < len(content) and count < norm_pos:
        if content[orig_pos] == "\r" and orig_pos + 1 < len(content) and content[orig_pos + 1] == "\n":
            orig_pos += 2
            count += 1  # \r\n → \n counts as 1
        elif content[orig_pos] == "\r":
            orig_pos += 1
            count += 1  # \r → \n counts as 1
        else:
            orig_pos += 1
            count += 1
    return orig_pos


def _fuzzy_find_by_lines(content: str, search: str) -> tuple[int, int]:
    """
    Find search in content by matching line-by-line with trailing whitespace stripped.

    Returns (start_idx, length) in original content.
    """
    content_lines = content.splitlines(True)  # keep line endings
    search_lines = search.splitlines(True)

    # Strip for comparison
    content_stripped = [l.strip() for l in content.splitlines()]
    search_stripped = [l.strip() for l in search.splitlines()]

    if not search_stripped:
        return -1, 0

    # Find first line match
    for start_line in range(len(content_stripped) - len(search_stripped) + 1):
        match = True
        for j, search_line in enumerate(search_stripped):
            if content_stripped[start_line + j] != search_line:
                match = False
                break
        if match:
            # Map back to original positions
            orig_start = sum(len(l) for l in content_lines[:start_line])
            orig_end = sum(len(l) for l in content_lines[:start_line + len(search_lines)])
            return orig_start, orig_end - orig_start

    return -1, 0


def _unified_diff(old: str, new: str, path: str, context: int = 3) -> str:
    """Generate a unified diff between old and new content."""
    import difflib

    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)

    diff = list(difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=f"{path} (before)",
        tofile=f"{path} (after)",
        n=context,
    ))

    if not diff:
        return ""

    # Truncate diff if too long (max 100 lines)
    if len(diff) > 100:
        diff = diff[:100]
        diff.append("... (diff truncated)\n")

    return "".join(diff).rstrip()
