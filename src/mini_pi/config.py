"""
Configuration for mini-pi coding agent.
"""

import os
from dataclasses import dataclass, field
from typing import Optional

from .compactor import CompactionConfig
from .context import PruningConfig


@dataclass
class Config:
    """Agent configuration, loaded from environment variables."""

    # LLM settings
    api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    base_url: str = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    model: str = field(default_factory=lambda: os.getenv("MINI_PI_MODEL", "gpt-4o"))

    # Agent settings
    max_steps: int = 20  # max tool call rounds per turn
    timeout: int = 30    # bash command timeout in seconds

    # Session
    session_dir: str = field(default_factory=lambda: os.path.expanduser("~/.mini-pi/sessions"))

    # Context management
    pruning: PruningConfig = field(default_factory=PruningConfig)
    compaction: CompactionConfig = field(default_factory=CompactionConfig)

    # System prompt
    system_prompt: str = ""

    # Working directory
    cwd: str = field(default_factory=os.getcwd)

    def validate(self) -> list[str]:
        """Return list of configuration issues."""
        issues = []
        if not self.api_key:
            issues.append("OPENAI_API_KEY not set")
        return issues
