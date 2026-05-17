"""
Configuration for mini-pi coding agent.
"""

import os
from dataclasses import dataclass, field
from typing import Any

from .compactor import CompactionConfig
from .context import PruningConfig
from .models import ModelRegistry, ModelInfo, create_llm, get_model_extra_kwargs


@dataclass
class Config:
    """Agent configuration, loaded from environment variables."""

    # LLM settings (legacy single-model, still supported)
    api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    base_url: str = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    model: str = field(default_factory=lambda: os.getenv("MINI_PI_MODEL", "gpt-4o"))

    # Agent settings
    max_steps: int = 99  # max tool call rounds per turn
    timeout: int = 30    # bash command timeout in seconds

    # Session
    session_dir: str = field(default_factory=lambda: os.path.expanduser("~/.mini-pi/sessions"))

    # Skills
    skill_dirs: list[str] = field(default_factory=lambda: [
        os.path.expanduser("~/.mini-pi/skills"),
    ])

    # Templates
    template_dirs: list[str] = field(default_factory=lambda: [
        os.path.expanduser("~/.mini-pi/templates"),
    ])

    # Context management
    pruning: PruningConfig = field(default_factory=PruningConfig)
    compaction: CompactionConfig = field(default_factory=CompactionConfig)

    # System prompt
    system_prompt: str = ""

    # Working directory
    cwd: str = field(default_factory=os.getcwd)

    # Multi-model registry
    model_registry: ModelRegistry = field(default_factory=ModelRegistry)
    _current_model_info: ModelInfo | None = field(default=None, init=False, repr=False)

    def validate(self) -> list[str]:
        """Return list of configuration issues."""
        issues = []

        # Try to resolve model from registry first
        resolved = self.resolve_model(self.model)
        if resolved is None and not self.api_key:
            issues.append("OPENAI_API_KEY not set (and model not found in registry)")
        elif resolved is None and self.api_key:
            # Legacy mode: just use OPENAI_API_KEY + OPENAI_BASE_URL + model name
            pass

        return issues

    def resolve_model(self, model_spec: str | None = None) -> ModelInfo | None:
        """
        Resolve a model spec to ModelInfo using the registry.

        Falls back to legacy env-based config if not found in registry.
        """
        spec = model_spec or self.model

        # Try registry first
        info = self.model_registry.resolve(spec)
        if info and info.api_key:
            self._current_model_info = info
            self.compaction.max_context_tokens = info.max_context_tokens
            return info

        # Also try the default from registry
        if not spec or spec == "gpt-4o":
            default = self.model_registry.get_default_spec()
            if default:
                info = self.model_registry.resolve(default)
                if info and info.api_key:
                    self._current_model_info = info
                    self.compaction.max_context_tokens = info.max_context_tokens
                    return info

        # Fallback: legacy env-based config
        if self.api_key:
            info = ModelInfo(
                provider="custom",
                model=spec,
                api_key=self.api_key,
                base_url=self.base_url,
                max_context_tokens=self.compaction.max_context_tokens,
            )
            self._current_model_info = info
            return info

        return None

    def get_current_model_info(self) -> ModelInfo | None:
        """Get the currently resolved model info."""
        if self._current_model_info is None:
            self.resolve_model()
        return self._current_model_info

    def get_extra_kwargs(self) -> dict[str, Any]:
        """Get provider-specific kwargs for the current model."""
        info = self.get_current_model_info()
        if info:
            return get_model_extra_kwargs(info)
        return {"temperature": 0}
