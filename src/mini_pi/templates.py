"""
Prompt templates for mini-pi.

Templates are reusable prompt configurations stored as YAML files.
They can include custom system prompt appendions, example user messages,
and model configuration overrides.

Template file format (~/.mini-pi/templates/<name>.yaml):
    name: code-review
    description: Review code for quality and bugs
    system_append: |
      Focus on security vulnerabilities and performance issues.
    model: deepseek/deepseek-v4-flash  # optional model override
    temperature: 0.3  # optional
    messages:  # optional starter messages
      - role: user
        content: Review the current codebase for common anti-patterns
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class PromptTemplate:
    """A reusable prompt template."""
    name: str
    description: str = ""
    system_append: str = ""
    model: str | None = None
    temperature: float | None = None
    messages: list[dict[str, str]] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PromptTemplate:
        """Create from a parsed YAML dict."""
        return cls(
            name=data.get("name", "unnamed"),
            description=data.get("description", ""),
            system_append=data.get("system_append", ""),
            model=data.get("model"),
            temperature=data.get("temperature"),
            messages=data.get("messages"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict suitable for YAML output."""
        d: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
        }
        if self.system_append:
            d["system_append"] = self.system_append
        if self.model:
            d["model"] = self.model
        if self.temperature is not None:
            d["temperature"] = self.temperature
        if self.messages:
            d["messages"] = self.messages
        return d


class TemplateManager:
    """
    Discovers and manages prompt templates.

    Template directories (in order of precedence):
    - .mini-pi/templates/ (project-level)
    - ~/.mini-pi/templates/ (global)
    """

    def __init__(self, template_dirs: list[str] | None = None):
        self.template_dirs = template_dirs or []
        self._templates: dict[str, PromptTemplate] = {}
        self._discovered = False

    def discover(self) -> list[PromptTemplate]:
        """Scan template directories and load templates."""
        self._templates = {}

        for dir_path in self.template_dirs:
            path = Path(dir_path)
            if not path.is_dir():
                continue

            for entry in sorted(path.iterdir()):
                if not entry.is_file():
                    continue
                if entry.suffix not in (".yaml", ".yml"):
                    continue

                template = self._load_template(entry)
                if template and template.name not in self._templates:
                    self._templates[template.name] = template

        self._discovered = True
        return list(self._templates.values())

    def get(self, name: str) -> PromptTemplate | None:
        """Get a template by name."""
        if not self._discovered:
            self.discover()
        return self._templates.get(name)

    @property
    def templates(self) -> list[PromptTemplate]:
        """All discovered templates."""
        if not self._discovered:
            self.discover()
        return list(self._templates.values())

    def _load_template(self, path: Path) -> PromptTemplate | None:
        """Load a single template from a YAML file."""
        try:
            import yaml
            content = path.read_text(encoding="utf-8")
            data = yaml.safe_load(content)
            if not isinstance(data, dict):
                return None
            return PromptTemplate.from_dict(data)
        except Exception:
            return None

    def format_list(self) -> str:
        """Format template list for display."""
        if not self.templates:
            return "No templates found."

        lines = ["[bold cyan]Templates:[/bold cyan]"]
        for t in self.templates:
            parts = [f"[green]{t.name}[/green]"]
            if t.description:
                parts.append(f"[dim]{t.description}[/dim]")
            if t.model:
                parts.append(f"[dim]model: {t.model}[/dim]")
            lines.append("  • " + " | ".join(parts))
        return "\n".join(lines)
