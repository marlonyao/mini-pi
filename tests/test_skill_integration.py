"""
Integration tests for the skill framework (v2) with the agent.

Tests that skills are properly discovered and integrated
using Progressive Disclosure:
- Catalog (name + description) in system prompt
- Full SKILL.md loaded on-demand by LLM via `read` tool
- No keyword matching
- No full content injection
"""

import io
import sys
import pytest
from pathlib import Path
from unittest.mock import patch

from mini_pi.agent import Agent
from mini_pi.config import Config
from mini_pi.llm import FakeLLM
from mini_pi.main import StreamingDisplay
from mini_pi.session import Session
from mini_pi.skills import Skill, SkillManager


# ── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def project_root():
    return Path(__file__).parent.parent


@pytest.fixture
def bundled_skills_dir(project_root):
    return project_root / "skills"


@pytest.fixture
def skills_dir_with_frontmatter(tmp_path):
    """Create skills with proper frontmatter."""
    code_review = tmp_path / "code-review"
    code_review.mkdir()
    (code_review / "SKILL.md").write_text(
        "---\n"
        "name: code-review\n"
        "description: Review source code for bugs, style issues, and improvements.\n"
        "---\n\n"
        "# Code Review\n\n"
        "## Checklist\n"
        "- [ ] Correctness\n"
        "- [ ] Performance\n"
        "- [ ] Security\n"
    )
    return tmp_path


# ── Bundled skill tests ────────────────────────────────────────────

class TestBundledCodeReviewSkill:
    def test_skill_exists(self, bundled_skills_dir):
        assert (bundled_skills_dir / "code-review" / "SKILL.md").exists()

    def test_skill_loads(self, bundled_skills_dir):
        skill = Skill.from_dir(bundled_skills_dir / "code-review")
        assert skill is not None
        assert skill.name == "code-review"

    def test_skill_has_description(self, bundled_skills_dir):
        skill = Skill.from_dir(bundled_skills_dir / "code-review")
        assert skill.description
        assert len(skill.description) > 10


# ── Agent integration tests ────────────────────────────────────────

class TestAgentSkillIntegration:
    def test_agent_discovers_skills(self, skills_dir_with_frontmatter, tmp_path):
        config = Config(
            api_key="test-key",
            skill_dirs=[str(skills_dir_with_frontmatter)],
        )
        session = Session(tmp_path / "test.jsonl")
        fake_llm = FakeLLM()
        agent = Agent(config, session, llm=fake_llm)

        assert len(agent.skill_manager.skills) >= 1
        names = {s.name for s in agent.skill_manager.skills}
        assert "code-review" in names

    def test_system_prompt_contains_xml_catalog(self, skills_dir_with_frontmatter, tmp_path):
        """System prompt should contain lightweight XML catalog, not full content."""
        config = Config(
            api_key="test-key",
            skill_dirs=[str(skills_dir_with_frontmatter)],
        )
        session = Session(tmp_path / "test.jsonl")
        fake_llm = FakeLLM()
        agent = Agent(config, session, llm=fake_llm)

        prompt = agent.system_prompt
        # Should have XML catalog
        assert "<available_skills>" in prompt
        assert "<name>code-review</name>" in prompt
        assert "<description>" in prompt
        # Should NOT have full skill body
        assert "## Checklist" not in prompt

    def test_no_full_skill_content_in_prompt(self, skills_dir_with_frontmatter, tmp_path):
        """Full SKILL.md body must NOT be in system prompt."""
        config = Config(
            api_key="test-key",
            skill_dirs=[str(skills_dir_with_frontmatter)],
        )
        session = Session(tmp_path / "test.jsonl")
        fake_llm = FakeLLM()
        agent = Agent(config, session, llm=fake_llm)

        # The detailed instructions should not be injected
        assert "- [ ] Correctness" not in agent.system_prompt
        assert "- [ ] Performance" not in agent.system_prompt

    def test_system_prompt_has_read_instruction(self, skills_dir_with_frontmatter, tmp_path):
        """System prompt should tell LLM to use read tool for skills."""
        config = Config(
            api_key="test-key",
            skill_dirs=[str(skills_dir_with_frontmatter)],
        )
        session = Session(tmp_path / "test.jsonl")
        fake_llm = FakeLLM()
        agent = Agent(config, session, llm=fake_llm)

        assert "read tool" in agent.system_prompt

    def test_core_tools_present(self, skills_dir_with_frontmatter, tmp_path):
        """Core tools (bash, read, write, edit, grep) should always be present."""
        config = Config(
            api_key="test-key",
            skill_dirs=[str(skills_dir_with_frontmatter)],
        )
        session = Session(tmp_path / "test.jsonl")
        fake_llm = FakeLLM()
        agent = Agent(config, session, llm=fake_llm)

        tool_names = {t["function"]["name"] for t in agent.tools}
        # Core tools must be present
        assert {"bash", "read", "write", "edit", "grep", "find", "ls"}.issubset(tool_names)

    def test_skill_tools_auto_registered(self, tmp_path):
        """Skill tools should be auto-registered at startup."""
        # Create a skill with tools.py
        skill_dir = tmp_path / "tool-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: tool-skill\ndescription: Has tools\n---\n"
        )
        (skill_dir / "tools.py").write_text(
            "from pydantic import BaseModel, Field\n\n"
            "class CountParams(BaseModel):\n"
            "    '''Count things.'''\n"
            "    what: str = Field(description='What to count')\n\n"
            "def tool_count(params, **kwargs):\n"
            "    return f'Counting {params[\"what\"]}'\n\n"
            "TOOL_DEFINITIONS = {\n"
            "    'count': (CountParams, tool_count),\n"
            "}\n"
        )

        config = Config(
            api_key="test-key",
            skill_dirs=[str(tmp_path)],
        )
        session = Session(tmp_path / "test.jsonl")
        fake_llm = FakeLLM()
        agent = Agent(config, session, llm=fake_llm)

        tool_names = {t["function"]["name"] for t in agent.tools}
        # Core tools + skill tool
        assert "count" in tool_names
        assert "bash" in tool_names

        # Executor should be available
        assert "count" in agent._skill_executors
        skill, handler = agent._skill_executors["count"]
        assert skill.name == "tool-skill"
        result = handler({"what": "lines"})
        assert "Counting lines" in result


# ── StreamingDisplay test (unchanged) ──────────────────────────────

class TestStreamingDisplay:
    def test_writes_to_original_stdout(self, monkeypatch):
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
