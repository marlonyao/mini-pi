"""
Integration tests for the skill framework with the agent.

Tests that skills are properly discovered, loaded, and integrated
into the agent's system prompt and tool set.
"""

import io
import sys
import pytest
from pathlib import Path
from unittest.mock import patch

from mini_pi.agent import Agent
from mini_pi.config import Config
from mini_pi.main import StreamingDisplay
from mini_pi.session import Session
from mini_pi.skills import Skill, SkillManager


# ── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def project_root():
    """Return the mini-pi project root path."""
    return Path(__file__).parent.parent


@pytest.fixture
def bundled_skills_dir(project_root):
    """Return the path to bundled skills."""
    return project_root / "skills"


# ── Tests ───────────────────────────────────────────────────────────

class TestBundledCodeReviewSkill:
    """Test the bundled code-review skill."""

    def test_skill_exists(self, bundled_skills_dir):
        """The code-review skill should exist in the skills directory."""
        assert (bundled_skills_dir / "code-review" / "SKILL.md").exists()

    def test_skill_loads(self, bundled_skills_dir):
        """The code-review skill should load correctly."""
        skill = Skill.from_dir(bundled_skills_dir / "code-review")
        assert skill is not None
        assert skill.name == "code-review"
        assert "review" in skill.description.lower() or "Code Review" in skill.description

    def test_skill_has_content(self, bundled_skills_dir):
        """The skill should have meaningful content."""
        skill = Skill.from_dir(bundled_skills_dir / "code-review")
        assert len(skill.skill_md_content) > 100
        assert "Checklist" in skill.skill_md_content or "checklist" in skill.skill_md_content.lower()

    def test_skill_has_system_prompt(self, bundled_skills_dir):
        """The skill should produce a system prompt addition."""
        skill = Skill.from_dir(bundled_skills_dir / "code-review")
        prompt = skill.system_prompt_addition()
        assert len(prompt) > 50
        assert "review" in prompt.lower()


class TestSkillManagerWithBundledSkills:
    """Test SkillManager with the bundled skills directory."""

    def test_discover_bundled_skills(self, bundled_skills_dir):
        """Should discover the bundled code-review skill."""
        manager = SkillManager(skill_dirs=[str(bundled_skills_dir)])
        skills = manager.discover()
        names = {s.name for s in skills}
        assert "code-review" in names

    def test_match_code_review(self, bundled_skills_dir):
        """Should match code-review skill for review requests."""
        manager = SkillManager(skill_dirs=[str(bundled_skills_dir)])
        manager.discover()

        # Various ways users might ask for code review
        matched = manager.match("please review my code")
        assert matched is not None
        assert matched.name == "code-review"

    def test_match_review_this_file(self, bundled_skills_dir):
        manager = SkillManager(skill_dirs=[str(bundled_skills_dir)])
        manager.discover()
        matched = manager.match("review this file for bugs")
        assert matched is not None
        assert matched.name == "code-review"

    def test_build_prompt_addition(self, bundled_skills_dir):
        """System prompt should include skill instructions."""
        manager = SkillManager(skill_dirs=[str(bundled_skills_dir)])
        manager.discover()

        addition = manager.build_skill_prompt_addition()
        assert "review" in addition.lower()
        assert "Checklist" in addition or "checklist" in addition.lower()


class TestAgentSkillIntegration:
    """Test that skills are integrated into the Agent."""

    def test_agent_loads_skills_from_config(self, bundled_skills_dir, tmp_path):
        """Agent should discover skills from configured skill_dirs."""
        config = Config(
            api_key="test-key",
            skill_dirs=[str(bundled_skills_dir)],
        )
        session = Session(tmp_path / "test.jsonl")

        # We need to mock the OpenAI client since we don't have a real API key
        with patch("mini_pi.agent.OpenAI"):
            agent = Agent(config, session)

        assert len(agent.skill_manager.skills) >= 1
        names = {s.name for s in agent.skill_manager.skills}
        assert "code-review" in names

    def test_agent_system_prompt_includes_skills(self, bundled_skills_dir, tmp_path):
        """Agent system prompt should include skill context."""
        config = Config(
            api_key="test-key",
            skill_dirs=[str(bundled_skills_dir)],
        )
        session = Session(tmp_path / "test.jsonl")

        with patch("mini_pi.agent.OpenAI"):
            agent = Agent(config, session)

        assert "review" in agent.system_prompt.lower()

    def test_agent_has_core_and_skill_tools(self, bundled_skills_dir, tmp_path):
        """Agent should have both core tools and any skill tools."""
        config = Config(
            api_key="test-key",
            skill_dirs=[str(bundled_skills_dir)],
        )
        session = Session(tmp_path / "test.jsonl")

        with patch("mini_pi.agent.OpenAI"):
            agent = Agent(config, session)

        tool_names = {t["function"]["name"] for t in agent.tools}
        # Core tools
        assert "bash" in tool_names
        assert "read" in tool_names
        assert "write" in tool_names
        assert "edit" in tool_names
        assert "grep" in tool_names

    def test_streaming_display_writes_to_original_stdout(self, monkeypatch):
        """Interactive streaming should bypass captured sys.stdout and write immediately."""
        original_stdout = io.StringIO()
        captured_stdout = io.StringIO()
        display = StreamingDisplay(original_stdout)

        monkeypatch.setattr(sys, "stdout", captured_stdout)
        display.start()
        display.add_reasoning("thinking")
        display.add("answer")
        display.stop()

        assert captured_stdout.getvalue() == ""
        assert "thinking" in original_stdout.getvalue()
        assert "\n\nanswer" in original_stdout.getvalue()


class TestCustomSkill:
    """Test creating and using a custom skill."""

    def test_custom_skill_in_temp_dir(self, tmp_path):
        """Should be able to create a skill on the fly."""
        # Create a custom skill
        skill_dir = tmp_path / "my-custom-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "# Custom Skill\n\n"
            "Use when: doing something custom.\n\n"
            "## Instructions\n"
            "1. Do step one\n"
            "2. Do step two\n"
        )

        # Load it
        skill = Skill.from_dir(skill_dir)
        assert skill is not None
        assert skill.name == "my-custom-skill"

        # Use in manager
        manager = SkillManager(skill_dirs=[str(tmp_path)])
        manager.discover()
        assert len(manager.skills) == 1

        # Match
        matched = manager.match("do something custom")
        assert matched is not None
        assert matched.name == "my-custom-skill"

    def test_custom_skill_with_tools(self, tmp_path):
        """Custom skill with tools.py should register tools."""
        skill_dir = tmp_path / "tool-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("# Tool Skill\n\nUse when: need tools.")
        (skill_dir / "tools.py").write_text(
            "from pydantic import BaseModel, Field\n\n"
            "class CountParams(BaseModel):\n"
            "    '''Count things.'''\n"
            "    what: str = Field(description='What to count')\n\n"
            "def tool_count(params, **kwargs):\n"
            "    return f'Counting {params.what}'\n\n"
            "TOOL_DEFINITIONS = {\n"
            "    'count': (CountParams, tool_count),\n"
            "}\n"
        )

        skill = Skill.from_dir(skill_dir)
        tools = skill.load_tools()
        assert len(tools) == 1
        assert tools[0]["function"]["name"] == "count"

        # Integrate with manager
        manager = SkillManager(skill_dirs=[str(tmp_path)])
        manager.discover()
        all_tools = manager.get_all_tools()
        assert len(all_tools) == 1
        assert all_tools[0]["function"]["name"] == "count"
