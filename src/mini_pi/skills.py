"""
Skill framework for mini-pi (v2 — Progressive Disclosure).

Follows the Agent Skills standard (https://agentskills.io/specification).
Inspired by Pi's implementation: LLM decides which skill to use,
then uses the `read` tool to load full instructions on-demand.

Key changes from v1:
- Only name + description + file_path are loaded at startup (Tier 1: Catalog)
- Full SKILL.md is NOT injected into system prompt
- System prompt only gets a lightweight XML catalog of available skills
- LLM uses the `read` tool to load SKILL.md when needed (Tier 2: Instructions)
- No keyword matching — LLM decides relevance autonomously
- Skill tools are NOT pre-registered; they load on demand
- Supports YAML frontmatter parsing (name, description fields)

Directory structure:
  skills/
  └── my-skill/
      ├── SKILL.md          # Required: frontmatter (name, description) + instructions
      ├── tools.py          # Optional: additional tool definitions
      └── references/       # Optional: reference docs, examples
"""

from __future__ import annotations

import importlib.util
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Skill:
    """A skill is a directory with a SKILL.md.

    Only metadata is loaded at startup. The full SKILL.md body
    is read on-demand by the agent via the `read` tool.
    """

    name: str
    description: str
    skill_dir: Path
    skill_md_path: Path  # Absolute path for the `read` tool

    @classmethod
    def from_dir(cls, skill_dir: Path | str) -> Skill | None:
        """
        Load skill metadata from a directory.

        Only reads frontmatter (name + description), NOT the full body.
        Returns None if the directory doesn't exist or has no SKILL.md.
        """
        skill_dir = Path(skill_dir)
        if not skill_dir.is_dir():
            return None

        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            return None

        content = skill_md.read_text(encoding="utf-8")

        # Parse YAML frontmatter
        name, description = _parse_frontmatter(content)

        # Fallback: extract description from first heading if no frontmatter
        if not description:
            for line in content.splitlines():
                line = line.strip()
                if line.startswith("# "):
                    description = line[2:].strip()
                    break

        # Must have a description to be useful
        if not description or not description.strip():
            return None

        # Use directory name as fallback for skill name
        if not name:
            name = skill_dir.name

        return cls(
            name=name,
            description=description.strip(),
            skill_dir=skill_dir,
            skill_md_path=skill_md.resolve(),
        )

    def load_tools(self) -> list[dict]:
        """
        Load tool definitions from tools.py if it exists.

        Called only when the skill is activated, not at startup.
        Returns a list of OpenAI-compatible tool definitions.
        """
        tools_py = self.skill_dir / "tools.py"
        if not tools_py.exists():
            return []

        try:
            spec = importlib.util.spec_from_file_location(
                f"skill_{self.name}_tools",
                tools_py,
            )
            if spec is None or spec.loader is None:
                return []

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            if not hasattr(module, "TOOL_DEFINITIONS"):
                return []

            defs = module.TOOL_DEFINITIONS
            tools = []
            for tool_name, (params_model, _handler) in defs.items():
                from pydantic import BaseModel
                if not issubclass(params_model, BaseModel):
                    continue
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


def _parse_frontmatter(content: str) -> tuple[str, str]:
    """Parse YAML frontmatter from SKILL.md content.

    Returns (name, description). Either may be empty string.
    Handles common YAML formatting issues (unquoted colons, etc.)
    """
    name = ""
    description = ""

    # Match frontmatter between --- delimiters
    match = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
    if not match:
        return name, description

    yaml_block = match.group(1)

    for line in yaml_block.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # Simple key: value parsing (handles common YAML)
        colon_idx = line.find(":")
        if colon_idx < 1:
            continue

        key = line[:colon_idx].strip().lower()
        value = line[colon_idx + 1:].strip()

        # Remove surrounding quotes if present
        if (value.startswith('"') and value.endswith('"')) or \
           (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]

        if key == "name":
            name = value
        elif key == "description":
            description = value

    return name, description


class SkillManager:
    """
    Discovers and manages skills from configured directories.

    Uses Progressive Disclosure:
    - Tier 1 (Catalog): name + description loaded at startup → injected into system prompt
    - Tier 2 (Instructions): full SKILL.md loaded on-demand by LLM via `read` tool
    - Tier 3 (Resources): scripts/references loaded when instructions reference them
    """

    def __init__(self, skill_dirs: list[str] | None = None):
        self.skill_dirs = skill_dirs or []
        self._skills: list[Skill] = []
        self._discovered = False

    def discover(self) -> list[Skill]:
        """Scan skill directories and load metadata (name + description only)."""
        self._skills = []
        seen_names: set[str] = set()

        for dir_path in self.skill_dirs:
            path = Path(dir_path)
            if not path.is_dir():
                continue

            for entry in sorted(path.iterdir()):
                if not entry.is_dir():
                    continue
                skill = Skill.from_dir(entry)
                if skill is not None:
                    # Handle name collisions: first found wins
                    if skill.name not in seen_names:
                        self._skills.append(skill)
                        seen_names.add(skill.name)

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

    def format_skills_for_prompt(self) -> str:
        """
        Build a lightweight XML catalog for the system prompt.

        This follows the Agent Skills standard format.
        Only includes name + description + file location (Tier 1).
        The LLM reads the full SKILL.md via the `read` tool when needed (Tier 2).

        Token cost: ~50-100 tokens per skill (vs thousands for full content).
        """
        if not self.skills:
            return ""

        lines = [
            "\n\nThe following skills provide specialized instructions for specific tasks.",
            "Use the read tool to load a skill's file when the task matches its description.",
            "When a skill file references a relative path, resolve it against the skill directory.",
            "",
            "<available_skills>",
        ]

        for skill in self.skills:
            lines.append("  <skill>")
            lines.append(f"    <name>{_escape_xml(skill.name)}</name>")
            lines.append(f"    <description>{_escape_xml(skill.description)}</description>")
            lines.append(f"    <location>{_escape_xml(str(skill.skill_md_path))}</location>")
            lines.append("  </skill>")

        lines.append("</available_skills>")
        return "\n".join(lines)


def _escape_xml(text: str) -> str:
    """Escape special XML characters."""
    return (
        text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )
