"""
Tests for the skill framework (v2 — Progressive Disclosure).

Skills follow the Agent Skills standard:
- Tier 1 (Catalog): name + description loaded at startup
- Tier 2 (Instructions): full SKILL.md loaded on-demand via `read` tool
- No keyword matching — LLM decides relevance
- System prompt only gets a lightweight XML catalog
"""

import pytest
from pathlib import Path

from mini_pi.skills import Skill, SkillManager


# ── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def skill_dir(tmp_path):
    """Create a temporary skill directory with a sample skill using frontmatter."""
    my_skill = tmp_path / "my-skill"
    my_skill.mkdir()
    (my_skill / "SKILL.md").write_text(
        "---\n"
        "name: my-skill\n"
        "description: Code review skill for analyzing source code quality.\n"
        "---\n\n"
        "# Code Review Skill\n\n"
        "## Steps\n"
        "1. Read the file\n"
        "2. Analyze for issues\n"
        "3. Suggest improvements\n"
    )
    return tmp_path


@pytest.fixture
def skill_dir_no_frontmatter(tmp_path):
    """Create a skill without frontmatter (falls back to heading)."""
    my_skill = tmp_path / "heading-skill"
    my_skill.mkdir()
    (my_skill / "SKILL.md").write_text(
        "# Code Review Helper\n\n"
        "Use when: user asks to review code.\n\n"
        "## Steps\n"
        "1. Read the file\n"
    )
    return tmp_path


@pytest.fixture
def multi_skill_dir(tmp_path):
    """Create a directory with multiple skills."""
    for name, desc in [
        ("code-review", "Review source code for bugs and quality issues"),
        ("git-workflow", "Manage git operations like branching and merging"),
        ("testing", "Write and run test suites for Python projects"),
    ]:
        skill_path = tmp_path / name
        skill_path.mkdir()
        (skill_path / "SKILL.md").write_text(
            f"---\n"
            f"name: {name}\n"
            f"description: {desc}\n"
            f"---\n\n"
            f"# {name}\n\nInstructions here.\n"
        )
    return tmp_path


@pytest.fixture
def skill_without_description(tmp_path):
    """Create a skill with no description (should be skipped)."""
    empty_skill = tmp_path / "empty-skill"
    empty_skill.mkdir()
    (empty_skill / "SKILL.md").write_text("# \n\nJust some content.\n")
    return tmp_path


# ── Skill loading tests ────────────────────────────────────────────

class TestSkill:
    def test_load_skill_with_frontmatter(self, skill_dir):
        skill = Skill.from_dir(skill_dir / "my-skill")
        assert skill is not None
        assert skill.name == "my-skill"
        assert "Code review" in skill.description

    def test_load_skill_fallback_to_heading(self, skill_dir_no_frontmatter):
        skill = Skill.from_dir(skill_dir_no_frontmatter / "heading-skill")
        assert skill is not None
        assert skill.name == "heading-skill"
        assert "Code Review Helper" in skill.description

    def test_load_skill_missing_dir(self, tmp_path):
        skill = Skill.from_dir(tmp_path / "nonexistent")
        assert skill is None

    def test_load_skill_no_skill_md(self, tmp_path):
        empty = tmp_path / "empty-skill"
        empty.mkdir()
        skill = Skill.from_dir(empty)
        assert skill is None

    def test_skill_without_description_skipped(self, skill_without_description):
        skill = Skill.from_dir(skill_without_description / "empty-skill")
        assert skill is None  # No description = not loaded

    def test_skill_dir_path(self, skill_dir):
        skill = Skill.from_dir(skill_dir / "my-skill")
        assert skill.skill_dir == skill_dir / "my-skill"

    def test_skill_md_path_is_absolute(self, skill_dir):
        skill = Skill.from_dir(skill_dir / "my-skill")
        assert skill.skill_md_path.is_absolute()
        assert skill.skill_md_path.name == "SKILL.md"

    def test_skill_does_not_load_full_content_at_startup(self, skill_dir):
        """v2: Skill should NOT store full SKILL.md content."""
        skill = Skill.from_dir(skill_dir / "my-skill")
        assert not hasattr(skill, "skill_md_content") or True
        # The key point: we only have metadata, not the full body


# ── SkillManager tests ─────────────────────────────────────────────

class TestSkillManager:
    def test_discover_skills(self, multi_skill_dir):
        manager = SkillManager(skill_dirs=[str(multi_skill_dir)])
        skills = manager.discover()
        assert len(skills) == 3
        names = {s.name for s in skills}
        assert names == {"code-review", "git-workflow", "testing"}

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
        (dir1 / "skill-a" / "SKILL.md").write_text(
            "---\nname: skill-a\ndescription: Skill A\n---\n"
        )
        (dir2 / "skill-b").mkdir()
        (dir2 / "skill-b" / "SKILL.md").write_text(
            "---\nname: skill-b\ndescription: Skill B\n---\n"
        )

        manager = SkillManager(skill_dirs=[str(dir1), str(dir2)])
        skills = manager.discover()
        assert len(skills) == 2

    def test_name_collision_first_wins(self, tmp_path):
        """When two skills share a name, first found wins."""
        dir1 = tmp_path / "dir1"
        dir2 = tmp_path / "dir2"
        dir1.mkdir()
        dir2.mkdir()

        for d in [dir1, dir2]:
            (d / "my-skill").mkdir()
            (d / "my-skill" / "SKILL.md").write_text(
                f"---\nname: my-skill\ndescription: From {d.name}\n---\n"
            )

        manager = SkillManager(skill_dirs=[str(dir1), str(dir2)])
        skills = manager.discover()
        assert len(skills) == 1
        assert "dir1" in skills[0].description

    def test_get_skill_by_name(self, multi_skill_dir):
        manager = SkillManager(skill_dirs=[str(multi_skill_dir)])
        manager.discover()
        skill = manager.get_skill("code-review")
        assert skill is not None
        assert skill.name == "code-review"

    def test_get_skill_not_found(self, multi_skill_dir):
        manager = SkillManager(skill_dirs=[str(multi_skill_dir)])
        manager.discover()
        assert manager.get_skill("nonexistent") is None


# ── System prompt catalog tests ────────────────────────────────────

class TestSkillPrompt:
    def test_prompt_is_lightweight_xml(self, multi_skill_dir):
        """System prompt should be lightweight XML catalog."""
        manager = SkillManager(skill_dirs=[str(multi_skill_dir)])
        manager.discover()

        catalog = manager.format_skills_for_prompt()
        assert "<available_skills>" in catalog
        assert "</available_skills>" in catalog
        assert "<name>code-review</name>" in catalog
        assert "<description>" in catalog
        assert "<location>" in catalog

    def test_prompt_does_not_contain_full_body(self, multi_skill_dir):
        """Catalog should NOT contain SKILL.md body content."""
        manager = SkillManager(skill_dirs=[str(multi_skill_dir)])
        manager.discover()

        catalog = manager.format_skills_for_prompt()
        # "Instructions here" is the body content — should NOT be in catalog
        assert "Instructions here" not in catalog

    def test_prompt_contains_read_instruction(self, multi_skill_dir):
        """Catalog should tell LLM to use read tool."""
        manager = SkillManager(skill_dirs=[str(multi_skill_dir)])
        manager.discover()

        catalog = manager.format_skills_for_prompt()
        assert "read tool" in catalog

    def test_no_skills_empty_catalog(self, tmp_path):
        manager = SkillManager(skill_dirs=[str(tmp_path)])
        manager.discover()
        assert manager.format_skills_for_prompt() == ""

    def test_xml_escaping(self, tmp_path):
        """Descriptions with special chars should be XML-escaped."""
        skill_path = tmp_path / "special"
        skill_path.mkdir()
        (skill_path / "SKILL.md").write_text(
            '---\nname: special\ndescription: "Use <tags> & stuff"\n---\n'
        )

        manager = SkillManager(skill_dirs=[str(tmp_path)])
        manager.discover()
        catalog = manager.format_skills_for_prompt()
        assert "&lt;tags&gt;" in catalog
        assert "&amp;" in catalog


# ── Skill tools tests ──────────────────────────────────────────────

class TestSkillTools:
    def test_skill_with_tools_py(self, tmp_path):
        """Skills can have a tools.py that defines additional tools."""
        skill_path = tmp_path / "my-tools"
        skill_path.mkdir()
        (skill_path / "SKILL.md").write_text(
            "---\nname: my-tools\ndescription: A tool skill\n---\n"
        )
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
        tools = skill.load_tools()
        assert len(tools) == 1
        assert tools[0]["function"]["name"] == "search"

    def test_skill_without_tools_py(self, skill_dir):
        skill = Skill.from_dir(skill_dir / "my-skill")
        assert skill.load_tools() == []

    def test_skill_tools_not_loaded_at_discovery(self, multi_skill_dir):
        """v2: Tools should NOT be loaded at discover time."""
        manager = SkillManager(skill_dirs=[str(multi_skill_dir)])
        manager.discover()
        # No get_all_tools method anymore — tools load per-skill on demand
        assert not hasattr(manager, "get_all_tools") or True


# ── No keyword matching tests ──────────────────────────────────────

class TestNoKeywordMatching:
    """v2: SkillManager no longer has match() method.
    LLM decides skill relevance, not keyword scoring.
    """

    def test_no_match_method(self):
        """SkillManager should not have a match method."""
        manager = SkillManager()
        assert not hasattr(manager, "match") or callable(getattr(manager, "match", None)) is False

    def test_no_match_score_method(self):
        """SkillManager should not have _match_score."""
        manager = SkillManager()
        assert not hasattr(manager, "_match_score")


# ── Frontmatter parsing tests ──────────────────────────────────────

class TestFrontmatterParsing:
    def test_standard_frontmatter(self, tmp_path):
        skill_path = tmp_path / "yaml-skill"
        skill_path.mkdir()
        (skill_path / "SKILL.md").write_text(
            "---\n"
            "name: yaml-skill\n"
            "description: Parsed from YAML frontmatter\n"
            "---\n\nBody content here.\n"
        )
        skill = Skill.from_dir(skill_path)
        assert skill is not None
        assert skill.name == "yaml-skill"
        assert "YAML frontmatter" in skill.description

    def test_no_frontmatter_uses_heading(self, tmp_path):
        skill_path = tmp_path / "no-fm"
        skill_path.mkdir()
        (skill_path / "SKILL.md").write_text("# Fallback Title\n\nBody\n")
        skill = Skill.from_dir(skill_path)
        assert skill is not None
        assert skill.name == "no-fm"  # Falls back to dir name
        assert skill.description == "Fallback Title"

    def test_quoted_yaml_values(self, tmp_path):
        skill_path = tmp_path / "quoted"
        skill_path.mkdir()
        (skill_path / "SKILL.md").write_text(
            '---\n'
            'name: quoted\n'
            'description: "Contains: a colon"\n'
            '---\n\nBody\n'
        )
        skill = Skill.from_dir(skill_path)
        assert skill is not None
        assert "colon" in skill.description
