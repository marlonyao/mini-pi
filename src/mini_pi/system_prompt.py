"""
System prompt builder for mini-pi.

Inspired by Pi's minimalist approach:
- Describe available tools clearly
- Give practical guidelines
- Keep it short to leave room for context
"""

from datetime import datetime
from pathlib import Path


def build_system_prompt(
    cwd: str = "",
    append: str = "",
    context_files: dict[str, str] | None = None,
) -> str:
    """
    Build the system prompt.

    Args:
        cwd: Current working directory.
        append: Extra text to append to the system prompt.
        context_files: Dict of {filename: content} for project context (like AGENTS.md).
    """
    prompt_cwd = Path(cwd).as_posix() if cwd else Path.cwd().as_posix()
    date = datetime.now().strftime("%Y-%m-%d")

    prompt = f"""\
You are an expert coding assistant. You help users by reading files, running commands, editing code, and writing new files.

Available tools:
- bash: Execute shell commands (timeout 30s). Use for file operations, building, testing, git, etc.
- read: Read file contents with optional line range. If given a directory, lists its contents.
- write: Create or overwrite files (creates parent directories automatically).
- edit: Find and replace exact text in a file. The old_text must be unique in the file.

Guidelines:
- Be concise and direct
- Use bash for file exploration (ls, grep, find)
- Prefer edit over write for small changes to existing files
- Show file paths clearly when working with files
- When exploring a new codebase, start by listing files, then read key files
- Verify your changes by reading the file back or running tests
- If a command might be destructive, mention what it will do first

Current date: {date}
Current working directory: {prompt_cwd}"""

    # Append project context files (like AGENTS.md, SYSTEM.md)
    if context_files:
        prompt += "\n\n# Project Context\n\n"
        for filepath, content in context_files.items():
            prompt += f"## {filepath}\n\n{content}\n\n"

    if append:
        prompt += f"\n\n{append}"

    return prompt
