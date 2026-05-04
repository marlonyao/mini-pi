"""
Core tools for mini-pi: bash, read, write, edit, grep.

These are the 5 primitives Pi uses — enough for most coding tasks.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from openai import pydantic_function_tool
from pydantic import BaseModel, Field


# ── Tool parameter schemas ──────────────────────────────────────────

class BashParams(BaseModel):
    """Run a bash command."""
    command: str = Field(description="The bash command to execute")


class ReadParams(BaseModel):
    """Read file contents."""
    path: str = Field(description="Path to the file to read")
    offset: int | None = Field(default=None, description="Line number to start reading from (1-indexed)")
    limit: int | None = Field(default=None, description="Maximum number of lines to read")


class WriteParams(BaseModel):
    """Write content to a file, creating parent directories as needed."""
    path: str = Field(description="Path to the file to write")
    content: str = Field(description="Content to write")


class EditParams(BaseModel):
    """Replace exact text in a file."""
    path: str = Field(description="Path to the file to edit")
    old_text: str = Field(description="Exact text to find and replace")
    new_text: str = Field(description="Replacement text")


class GrepParams(BaseModel):
    """Search for a pattern in files."""
    pattern: str = Field(description="The regex pattern to search for")
    path: str | None = Field(default=None, description="Directory or file to search in (default: .)")
    file_type: str | None = Field(default=None, description="File extension to filter (e.g. 'py', 'js'). Omit to search all files.")


# ── Tool implementations ────────────────────────────────────────────

def tool_bash(params: BashParams, *, timeout: int = 30, cwd: str = "") -> str:
    """Execute a bash command and return stdout + stderr."""
    try:
        result = subprocess.run(
            params.command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd or None,
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += f"\nSTDERR:\n{result.stderr}"
        if result.returncode != 0:
            output += f"\nExit code: {result.returncode}"
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return f"Command timed out after {timeout}s"
    except Exception as e:
        return f"Error: {e}"


def tool_read(params: ReadParams, *, cwd: str = "") -> str:
    """Read file contents, optionally a range of lines."""
    path = _resolve(params.path, cwd)
    try:
        if not path.exists():
            return f"Error: File not found: {path}"
        if path.is_dir():
            entries = sorted(path.iterdir())
            return "\n".join(
                f"{'📁' if e.is_dir() else '📄'} {e.name}"
                for e in entries[:100]
            ) + (f"\n... and {len(entries) - 100} more" if len(entries) > 100 else "")

        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        start = (params.offset or 1) - 1
        end = start + params.limit if params.limit else len(lines)
        selected = lines[start:end]

        # Add line numbers
        total = len(lines)
        result_lines = []
        for i, line in enumerate(selected, start=start + 1):
            result_lines.append(f"{i:>6} | {line}")

        header = f"File: {path} ({total} lines)"
        if params.offset or params.limit:
            header += f" [showing lines {start+1}-{min(start+len(selected), total)}]"
        return header + "\n" + "\n".join(result_lines)

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
    """Replace exact text in a file."""
    path = _resolve(params.path, cwd)
    try:
        if not path.exists():
            return f"Error: File not found: {path}"
        content = path.read_text(encoding="utf-8")
        if params.old_text not in content:
            # Try to give helpful feedback
            return f"Error: old_text not found in {path}. The exact text to replace must match."
        count = content.count(params.old_text)
        if count > 1:
            return f"Error: old_text found {count} times in {path}. Provide more context to make it unique."
        new_content = content.replace(params.old_text, params.new_text)
        path.write_text(new_content, encoding="utf-8")
        return f"✅ Edited {path} (replaced 1 occurrence)"
    except Exception as e:
        return f"Error editing file: {e}"


def tool_grep(params: GrepParams, *, timeout: int = 30, cwd: str = "") -> str:
    """Search for a pattern in files using ripgrep (rg) if available, else grep -rn."""
    search_path = _resolve(params.path or ".", cwd)
    try:
        if not search_path.exists():
            return f"Error: Path not found: {search_path}"

        # Build the command
        if shutil.which("rg"):
            # ripgrep is available — fast, respects .gitignore
            cmd = ["rg", "--line-number", "--with-filename", "--no-heading", "--color=never"]
            if params.file_type:
                cmd += ["--type", params.file_type]
            cmd += [params.pattern, str(search_path)]
        else:
            # Fall back to standard grep
            cmd = ["grep", "-rn", "--color=never", "-E"]
            if params.file_type:
                cmd += ["--include", f"*.{params.file_type}"]
            cmd += [params.pattern, str(search_path)]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        output = result.stdout.strip()
        if not output:
            if result.returncode == 1:
                return "No matches found."
            # rg returns 2 on errors; grep may return 2 as well
            if result.stderr:
                return f"Error: {result.stderr.strip()}"
            return "No matches found."

        # Truncate if output is very long
        lines = output.splitlines()
        max_lines = 200
        if len(lines) > max_lines:
            truncated = "\n".join(lines[:max_lines])
            truncated += f"\n... ({len(lines) - max_lines} more matches, showing first {max_lines})"
            return truncated

        return output

    except subprocess.TimeoutExpired:
        return f"Search timed out after {timeout}s"
    except Exception as e:
        return f"Error searching: {e}"


# ── Tool registry ───────────────────────────────────────────────────

TOOL_DEFINITIONS = {
    "bash": (BashParams, tool_bash),
    "read": (ReadParams, tool_read),
    "write": (WriteParams, tool_write),
    "edit": (EditParams, tool_edit),
    "grep": (GrepParams, tool_grep),
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
        # Only pass timeout to bash and grep, cwd to all
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
