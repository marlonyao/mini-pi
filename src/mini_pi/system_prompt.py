"""
System prompt builder for mini-pi.

Inspired by Pi's system prompt construction:
- Project context (AGENTS.md, README.md, etc.)
- Date and working directory
- Tool descriptions and guidelines
- Skill catalog (lightweight, full content loaded on-demand)
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any


# ── Project context file discovery ──────────────────────────────────

# Files to look for in the project root (in order of priority)
CONTEXT_FILENAMES = [
    "AGENTS.md",
    "SYSTEM.md",
    "CLAUDE.md",       # Claude Code compat
    ".cursorrules",    # Cursor compat
    "README.md",
]

# Maximum size per context file (50KB)
MAX_CONTEXT_FILE_SIZE = 50_000


def discover_context_files(cwd: str) -> list[tuple[str, str]]:
    """
    Discover and load project context files from the working directory.

    Returns list of (filename, content) tuples for files that exist.
    Skips files that are too large.
    """
    cwd_path = Path(cwd).resolve()
    results: list[tuple[str, str]] = []

    for filename in CONTEXT_FILENAMES:
        filepath = cwd_path / filename
        if filepath.is_file():
            try:
                size = filepath.stat().st_size
                if size > MAX_CONTEXT_FILE_SIZE:
                    continue
                content = filepath.read_text(encoding="utf-8").strip()
                if content:
                    results.append((filename, content))
            except (OSError, UnicodeDecodeError):
                continue

    return results


# ── System prompt building ──────────────────────────────────────────

def build_system_prompt(
    cwd: str = "",
    append: str = "",
    context_files: dict[str, str] | None = None,
    discover_files: bool = True,
    tool_snippets: dict[str, str] | None = None,
) -> str:
    """
    Build the system prompt.

    Args:
        cwd: Current working directory.
        append: Extra text to append to the system prompt.
        context_files: Dict of {filename: content} for project context.
        discover_files: If True, auto-discover AGENTS.md etc. from cwd.
        tool_snippets: One-line descriptions per tool for the tools section.
    """
    prompt_cwd = Path(cwd).as_posix() if cwd else Path.cwd().as_posix()
    date = datetime.now().strftime("%Y-%m-%d")
    weekday = datetime.now().strftime("%A")

    # Build tools section
    tools = tool_snippets or {
        "bash": "Execute shell commands (timeout 30s). Use for file operations, building, testing, git, etc.",
        "read": "Read file contents with optional line range. If given a directory, lists its contents.",
        "write": "Create or overwrite files (creates parent directories automatically).",
        "edit": "Find and replace exact text in a file. The old_text must be unique. Supports multiple edits in one call.",
        "grep": "Search for a pattern in files with regex, glob filtering, and context lines.",
        "find": "Find files by name pattern in a directory tree.",
        "ls": "List directory contents with file sizes.",
    }

    tools_section = "\n".join(f"- {name}: {desc}" for name, desc in tools.items())

    prompt = f"""\
You are an expert coding assistant. You help users by reading files, running commands, editing code, and writing new files.

# Available Tools

{tools_section}

# Guidelines

- Be concise and direct. Skip unnecessary preamble.
- Use bash for file exploration (ls, grep, find)
- Prefer edit over write for small changes to existing files
- Show file paths clearly when working with files
- When exploring a new codebase, start by listing files, then read key files
- Verify your changes by reading the file back or running tests
- If a command might be destructive, mention what it will do first
- For multi-file changes, make all edits before running tests
- When debugging, read the relevant code first, form a hypothesis, then test it

Current date: {weekday}, {date}
Current working directory: {prompt_cwd}"""

    # Project context files
    all_context: dict[str, str] = {}

    # Auto-discover from cwd
    if discover_files and cwd:
        for filename, content in _discover_context_files(cwd):
            all_context[filename] = content

    # Explicitly provided context files (override discovered ones)
    if context_files:
        all_context.update(context_files)

    if all_context:
        prompt += "\n\n# Project Context\n\n"
        for filepath, content in all_context.items():
            prompt += f"## {filepath}\n\n{content}\n\n"

    if append:
        prompt += f"\n\n{append}"

    return prompt


def _discover_context_files(cwd: str) -> list[tuple[str, str]]:
    """Discover context files, returned as (filename, content) pairs."""
    cwd_path = Path(cwd).resolve()
    results: list[tuple[str, str]] = []

    for filename in CONTEXT_FILENAMES:
        filepath = cwd_path / filename
        if filepath.is_file():
            try:
                size = filepath.stat().st_size
                if size > MAX_CONTEXT_FILE_SIZE:
                    continue
                content = filepath.read_text(encoding="utf-8").strip()
                if content:
                    results.append((filename, content))
            except (OSError, UnicodeDecodeError):
                continue

    return results
