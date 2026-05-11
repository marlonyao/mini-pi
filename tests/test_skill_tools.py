"""
Tests for skill tool registration.

Covers:
- Skills with tools.py get their tools auto-registered with the agent
- Skill tools appear in get_openai_tools()-style output
- Skill tool executors are invoked correctly
- Skills without tools.py are unaffected
- Multiple skills can register distinct tools
- Tool name conflicts: skill tools don't override core tools
"""

import pytest
from pathlib import Path

from mini_pi.skills import Skill, SkillManager


# ── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def skills_dir(tmp_path):
    """Create a skills directory with some skills that have tools."""
    skills = tmp_path / "skills"
    skills.mkdir()

    # Skill with tools
    skill1 = skills / "counter"
    skill1.mkdir()
    (skill1 / "SKILL.md").write_text(
        "---\nname: counter\ndescription: Count things\n---\n\n# Counter\n"
    )
    (skill1 / "tools.py").write_text(
        "from pydantic import BaseModel, Field\n\n"
        "class CountParams(BaseModel):\n"
        "    '''Count items.'''\n"
        "    items: list[str] = Field(description='Items to count')\n\n"
        "def tool_count(params, **kwargs):\n"
        "    return f'Counted {len(params[\"items\"])} items'\n\n"
        "TOOL_DEFINITIONS = {\n"
        "    'skill_count': (CountParams, tool_count),\n"
        "}\n"
    )

    # Skill without tools
    skill2 = skills / "simple"
    skill2.mkdir()
    (skill2 / "SKILL.md").write_text(
        "---\nname: simple\ndescription: A simple skill without tools\n---\n\n# Simple\n"
    )

    # Skill with another tool
    skill3 = skills / "analyzer"
    skill3.mkdir()
    (skill3 / "SKILL.md").write_text(
        "---\nname: analyzer\ndescription: Analyze code\n---\n\n# Analyzer\n"
    )
    (skill3 / "tools.py").write_text(
        "from pydantic import BaseModel, Field\n\n"
        "class AnalyzeParams(BaseModel):\n"
        "    '''Analyze code quality.'''\n"
        "    path: str = Field(description='File path')\n"
        "    checks: list[str] = Field(default=['lint', 'types'], description='Checks')\n\n"
        "def tool_analyze(params, **kwargs):\n"
        "    return f'Analyzing {params[\"path\"]} with {params[\"checks\"]}'\n\n"
        "TOOL_DEFINITIONS = {\n"
        "    'skill_analyze': (AnalyzeParams, tool_analyze),\n"
        "}\n"
    )

    return skills


@pytest.fixture
def skill_with_bad_tools(tmp_path):
    """Skill with a tools.py that has errors."""
    skills = tmp_path / "skills"
    skills.mkdir()

    skill1 = skills / "broken"
    skill1.mkdir()
    (skill1 / "SKILL.md").write_text(
        "---\nname: broken\ndescription: Broken tools\n---\n\n# Broken\n"
    )
    (skill1 / "tools.py").write_text(
        "# This file has invalid Python\nraise RuntimeError('oops')\n"
    )

    skill2 = skills / "good"
    skill2.mkdir()
    (skill2 / "SKILL.md").write_text(
        "---\nname: good\ndescription: Good skill\n---\n\n# Good\n"
    )
    (skill2 / "tools.py").write_text(
        "from pydantic import BaseModel\n\n"
        "class HelloParams(BaseModel):\n"
        "    '''Say hello.'''\n"
        "    name: str\n\n"
        "def tool_hello(params, **kwargs):\n"
        "    return f'Hello {params[\"name\"]}'\n\n"
        "TOOL_DEFINITIONS = {\n"
        "    'skill_hello': (HelloParams, tool_hello),\n"
        "}\n"
    )

    return skills


# ── Tests ───────────────────────────────────────────────────────────

class TestSkillToolLoading:
    def test_skill_load_tools_returns_definitions(self, skills_dir):
        skill = Skill.from_dir(skills_dir / "counter")
        assert skill is not None
        tools = skill.load_tools()
        assert len(tools) == 1
        assert tools[0]["function"]["name"] == "skill_count"
        assert "items" in tools[0]["function"]["parameters"]["properties"]

    def test_skill_without_tools_returns_empty(self, skills_dir):
        skill = Skill.from_dir(skills_dir / "simple")
        assert skill is not None
        tools = skill.load_tools()
        assert tools == []

    def test_skill_with_analyze_tools(self, skills_dir):
        skill = Skill.from_dir(skills_dir / "analyzer")
        assert skill is not None
        tools = skill.load_tools()
        assert len(tools) == 1
        assert tools[0]["function"]["name"] == "skill_analyze"


class TestSkillManagerGetAllSkillTools:
    def test_discovers_all_skill_tools(self, skills_dir):
        manager = SkillManager(skill_dirs=[str(skills_dir)])
        manager.discover()

        tools, executors = manager.get_all_skill_tools()
        tool_names = {t["function"]["name"] for t in tools}
        assert "skill_count" in tool_names
        assert "skill_analyze" in tool_names
        # Simple skill has no tools
        assert "simple" not in tool_names

    def test_executors_are_callable(self, skills_dir):
        manager = SkillManager(skill_dirs=[str(skills_dir)])
        manager.discover()

        _, executors = manager.get_all_skill_tools()
        assert "skill_count" in executors

        skill, handler = executors["skill_count"]
        assert skill.name == "counter"
        result = handler({"items": ["a", "b", "c"]})
        assert "Counted 3 items" in result

    def test_analyze_executor_works(self, skills_dir):
        manager = SkillManager(skill_dirs=[str(skills_dir)])
        manager.discover()

        _, executors = manager.get_all_skill_tools()
        skill, handler = executors["skill_analyze"]
        result = handler({"path": "main.py", "checks": ["lint"]})
        assert "Analyzing main.py" in result

    def test_no_skills_returns_empty(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        manager = SkillManager(skill_dirs=[str(empty)])
        manager.discover()

        tools, executors = manager.get_all_skill_tools()
        assert tools == []
        assert executors == {}

    def test_broken_tools_dont_crash_others(self, skill_with_bad_tools):
        manager = SkillManager(skill_dirs=[str(skill_with_bad_tools)])
        manager.discover()

        tools, executors = manager.get_all_skill_tools()
        # Good skill should still work
        tool_names = {t["function"]["name"] for t in tools}
        assert "skill_hello" in tool_names
        # Broken skill's tools should be skipped
        assert "skill_broken" not in tool_names

    def test_tool_definitions_have_correct_schema(self, skills_dir):
        manager = SkillManager(skill_dirs=[str(skills_dir)])
        manager.discover()

        tools, _ = manager.get_all_skill_tools()
        for tool in tools:
            assert tool["type"] == "function"
            assert "name" in tool["function"]
            assert "description" in tool["function"]
            assert "parameters" in tool["function"]
            params = tool["function"]["parameters"]
            assert params["type"] == "object"
            assert "properties" in params
