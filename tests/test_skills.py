"""
Tests for the skill framework.

Skills are directories with a SKILL.md and optional tool definitions.
The SkillManager should:
1. Discover skills from configured directories
2. Load SKILL.md descriptions
3. Load optional tool definitions from tools.py
4. Match skills to user messages
5. Inject skill context into system prompt
"""

import pytest
from pathlib import Path

from mini_pi.skills import Skill, SkillManager


# ── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def skill_dir(tmp_path):
    """Create a temporary skill directory with a sample skill."""
    my_skill = tmp_path / "my-skill"
    my_skill.mkdir()
    (my_skill / "SKILL.md").write_text(
        "# Code Review Skill\n\n"
        "Use when: user asks to review code.\n\n"
        "## Steps\n"
        "1. Read the file\n"
        "2. Analyze for issues\n"
        "3. Suggest improvements\n"
    )
    return tmp_path


@pytest.fixture
def multi_skill_dir(tmp_path):
    """Create a directory with multiple skills."""
    for name, desc in [
        ("code-review", "# Code Review\n\nUse when: reviewing code"),
        ("git-workflow", "# Git Workflow\n\nUse when: git operations"),
        ("testing", "# Testing\n\nUse when: writing or running tests"),
    ]:
        skill_path = tmp_path / name
        skill_path.mkdir()
        (skill_path / "SKILL.md").write_text(desc)
    return tmp_path


# ── Tests ───────────────────────────────────────────────────────────

class TestSkill:
    def test_load_skill(self, skill_dir):
        skill = Skill.from_dir(skill_dir / "my-skill")
        assert skill is not None
        assert skill.name == "my-skill"
        assert "Code Review" in skill.description
        assert skill.skill_md_content  # Should have loaded the file content

    def test_load_skill_missing_dir(self, tmp_path):
        skill = Skill.from_dir(tmp_path / "nonexistent")
        assert skill is None

    def test_load_skill_no_skill_md(self, tmp_path):
        empty_skill = tmp_path / "empty-skill"
        empty_skill.mkdir()
        skill = Skill.from_dir(empty_skill)
        assert skill is None

    def test_skill_dir_path(self, skill_dir):
        skill = Skill.from_dir(skill_dir / "my-skill")
        assert skill.skill_dir == skill_dir / "my-skill"

    def test_skill_has_system_prompt_addition(self, skill_dir):
        skill = Skill.from_dir(skill_dir / "my-skill")
        assert skill.system_prompt_addition()
        assert "Code Review" in skill.system_prompt_addition()


class TestSkillManager:
    def test_discover_skills(self, multi_skill_dir):
        manager = SkillManager(skill_dirs=[str(multi_skill_dir)])
        skills = manager.discover()
        assert len(skills) == 3
        names = {s.name for s in skills}
        assert "code-review" in names
        assert "git-workflow" in names
        assert "testing" in names

    def test_discover_empty_dir(self, tmp_path):
        manager = SkillManager(skill_dirs=[str(tmp_path)])
        skills = manager.discover()
        assert skills == []

    def test_discover_nonexistent_dir(self):
        manager = SkillManager(skill_dirs=["/nonexistent/path"])
        skills = manager.discover()
        assert skills == []

    def test_discover_multiple_dirs(self, tmp_path):
        dir1 = tmp_path / "skills1"
        dir2 = tmp_path / "skills2"
        dir1.mkdir()
        dir2.mkdir()

        (dir1 / "skill-a").mkdir()
        (dir1 / "skill-a" / "SKILL.md").write_text("# Skill A")
        (dir2 / "skill-b").mkdir()
        (dir2 / "skill-b" / "SKILL.md").write_text("# Skill B")

        manager = SkillManager(skill_dirs=[str(dir1), str(dir2)])
        skills = manager.discover()
        assert len(skills) == 2

    def test_get_skill_by_name(self, multi_skill_dir):
        manager = SkillManager(skill_dirs=[str(multi_skill_dir)])
        manager.discover()
        skill = manager.get_skill("code-review")
        assert skill is not None
        assert skill.name == "code-review"

    def test_get_skill_not_found(self, multi_skill_dir):
        manager = SkillManager(skill_dirs=[str(multi_skill_dir)])
        manager.discover()
        skill = manager.get_skill("nonexistent")
        assert skill is None

    def test_get_all_tools_empty(self, multi_skill_dir):
        """Skills without tools.py should return empty tool list."""
        manager = SkillManager(skill_dirs=[str(multi_skill_dir)])
        manager.discover()
        tools = manager.get_all_tools()
        assert tools == []

    def test_match_skill_by_keyword(self, multi_skill_dir):
        """Match skill based on user message content."""
        manager = SkillManager(skill_dirs=[str(multi_skill_dir)])
        manager.discover()

        matched = manager.match("please review this code")
        assert matched is not None
        assert matched.name == "code-review"

    def test_match_skill_no_match(self, multi_skill_dir):
        manager = SkillManager(skill_dirs=[str(multi_skill_dir)])
        manager.discover()

        matched = manager.match("what's the weather today?")
        assert matched is None

    def test_match_returns_most_relevant(self, multi_skill_dir):
        """When multiple skills could match, return the most relevant."""
        manager = SkillManager(skill_dirs=[str(multi_skill_dir)])
        manager.discover()

        matched = manager.match("run the test suite")
        assert matched is not None
        assert matched.name == "testing"


class TestSkillWithTools:
    def test_skill_with_tools_py(self, tmp_path):
        """Skills can have a tools.py that registers additional tools."""
        skill_path = tmp_path / "my-tools"
        skill_path.mkdir()
        (skill_path / "SKILL.md").write_text("# Tool Skill")
        (skill_path / "tools.py").write_text(
            "from pydantic import BaseModel, Field\n\n"
            "class SearchParams(BaseModel):\n"
            "    query: str = Field(description='Search query')\n\n"
            "def tool_search(params, **kwargs):\n"
            "    return f'Searching for: {params.query}'\n\n"
            "TOOL_DEFINITIONS = {\n"
            "    'search': (SearchParams, tool_search),\n"
            "}\n"
        )

        skill = Skill.from_dir(skill_path)
        assert skill is not None
        # Should be able to load tool definitions
        tools = skill.load_tools()
        assert len(tools) == 1
        assert tools[0]["function"]["name"] == "search"

    def test_skill_without_tools_py(self, skill_dir):
        """Skills without tools.py should work fine."""
        skill = Skill.from_dir(skill_dir / "my-skill")
        tools = skill.load_tools()
        assert tools == []


class TestSkillSystemPrompt:
    def test_skill_prompt_injection(self, skill_dir):
        """Skill content should be injectable into system prompt."""
        manager = SkillManager(skill_dirs=[str(skill_dir)])
        manager.discover()

        addition = manager.build_skill_prompt_addition()
        assert "Code Review" in addition

    def test_no_skills_prompt(self, tmp_path):
        """Empty skills should produce empty prompt addition."""
        manager = SkillManager(skill_dirs=[str(tmp_path)])
        manager.discover()
        addition = manager.build_skill_prompt_addition()
        assert addition == ""
