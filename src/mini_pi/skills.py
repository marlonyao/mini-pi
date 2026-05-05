"""
Skill framework for mini-pi.

Skills are directories containing a SKILL.md (required) and optional tools.py.
They provide domain-specific knowledge and tools that the agent can use.

Directory structure:
  skills/
  └── my-skill/
      ├── SKILL.md          # Required: skill description and instructions
      ├── tools.py          # Optional: additional tool definitions
      └── references/       # Optional: reference docs, examples
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel

from .tools import TOOL_DEFINITIONS


@dataclass
class Skill:
    """A skill is a directory with a SKILL.md and optional tools."""

    name: str
    description: str              # First line / heading from SKILL.md
    skill_dir: Path
    skill_md_content: str         # Full SKILL.md content

    @classmethod
    def from_dir(cls, skill_dir: Path | str) -> Skill | None:
        """
        Load a skill from a directory.

        Returns None if the directory doesn't exist or has no SKILL.md.
        """
        skill_dir = Path(skill_dir)
        if not skill_dir.is_dir():
            return None

        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            return None

        content = skill_md.read_text(encoding="utf-8")
        # Extract description from first heading
        description = ""
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("# "):
                description = line[2:].strip()
                break

        return cls(
            name=skill_dir.name,
            description=description,
            skill_dir=skill_dir,
            skill_md_content=content,
        )

    def system_prompt_addition(self) -> str:
        """Get the content to inject into the system prompt when this skill is active."""
        return self.skill_md_content

    def load_tools(self) -> list[dict]:
        """
        Load tool definitions from tools.py if it exists.

        Returns a list of OpenAI-compatible tool definitions.
        """
        tools_py = self.skill_dir / "tools.py"
        if not tools_py.exists():
            return []

        try:
            # Dynamically import the tools module
            spec = importlib.util.spec_from_file_location(
                f"skill_{self.name}_tools",
                tools_py,
            )
            if spec is None or spec.loader is None:
                return []

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Look for TOOL_DEFINITIONS dict
            if not hasattr(module, "TOOL_DEFINITIONS"):
                return []

            defs = module.TOOL_DEFINITIONS
            tools = []
            for tool_name, (params_model, handler) in defs.items():
                # Convert Pydantic model to OpenAI function tool format
                schema = params_model.model_json_schema()
                tools.append({
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": params_model.__doc__ or f"Tool: {tool_name}",
                        "parameters": schema,
                    },
                })

            return tools

        except Exception:
            return []


class SkillManager:
    """
    Discovers and manages skills from configured directories.
    """

    def __init__(self, skill_dirs: list[str] | None = None):
        self.skill_dirs = skill_dirs or []
        self._skills: list[Skill] = []
        self._discovered = False

    def discover(self) -> list[Skill]:
        """
        Scan skill directories and load all valid skills.

        Returns the list of discovered skills.
        """
        self._skills = []

        for dir_path in self.skill_dirs:
            path = Path(dir_path)
            if not path.is_dir():
                continue

            for entry in sorted(path.iterdir()):
                if entry.is_dir():
                    skill = Skill.from_dir(entry)
                    if skill is not None:
                        self._skills.append(skill)

        self._discovered = True
        return self._skills

    def get_skill(self, name: str) -> Skill | None:
        """Get a skill by name."""
        for skill in self._skills:
            if skill.name == name:
                return skill
        return None

    @property
    def skills(self) -> list[Skill]:
        """All discovered skills."""
        if not self._discovered:
            self.discover()
        return self._skills

    def match(self, user_message: str) -> Skill | None:
        """
        Find the most relevant skill for a user message.

        Uses simple keyword matching against skill names and descriptions.
        Returns None if no skill matches.
        """
        message_lower = user_message.lower()

        best_match: Skill | None = None
        best_score = 0

        for skill in self.skills:
            score = self._match_score(skill, message_lower)
            if score > best_score:
                best_score = score
                best_match = skill

        # Require minimum score to avoid false matches
        return best_match if best_score >= 1 else None

    def _match_score(self, skill: Skill, message_lower: str) -> int:
        """Calculate a simple match score for a skill against a message."""
        score = 0

        # Check skill name keywords
        name_words = skill.name.replace("-", " ").split()
        for word in name_words:
            if word in message_lower or message_lower in word:
                score += 2
            # Stem matching: "test" matches "testing"
            elif len(word) >= 4 and word[:4] in message_lower:
                score += 1

        # Check description keywords
        desc_words = skill.description.lower().split()
        for word in desc_words:
            word = word.strip("#:-.,")
            if len(word) > 2 and (word in message_lower or message_lower in word):
                score += 1
            elif len(word) >= 4 and word[:4] in message_lower:
                score += 1

        # Check SKILL.md content for "Use when:" patterns
        content_lower = skill.skill_md_content.lower()
        if "use when:" in content_lower:
            for line in skill.skill_md_content.splitlines():
                if "use when:" in line.lower():
                    condition = line.lower().split("use when:", 1)[1]
                    condition_words = condition.replace(",", " ").split()
                    for word in condition_words:
                        word = word.strip(". ")
                        if len(word) > 2 and word in message_lower:
                            score += 3

        return score

    def get_all_tools(self) -> list[dict]:
        """Get OpenAI tool definitions from all skills that have tools."""
        tools = []
        for skill in self.skills:
            tools.extend(skill.load_tools())
        return tools

    def build_skill_prompt_addition(self) -> str:
        """
        Build a system prompt addition with all skill descriptions.

        This lets the agent know what skills are available.
        """
        if not self.skills:
            return ""

        parts = ["\n\n# Available Skills\n"]
        parts.append("When the user's request matches a skill, read and follow its instructions.\n")

        for skill in self.skills:
            parts.append(f"## {skill.description}\n")
            parts.append(skill.skill_md_content)
            parts.append("")

        return "\n".join(parts)
