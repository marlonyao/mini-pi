"""
Tests for prompt templates.

Covers:
- Template discovery from YAML files
- Template loading (system_append, model, messages)
- TemplateManager get/list
- Agent.apply_template integration
- Missing/broken YAML files don't crash
"""

import pytest
from pathlib import Path

from mini_pi.templates import PromptTemplate, TemplateManager
from mini_pi.agent import Agent
from mini_pi.config import Config
from mini_pi.llm import FakeLLM
from mini_pi.session import Session


# ── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def template_dir(tmp_path):
    """Create a template directory with sample templates."""
    tdir = tmp_path / "templates"
    tdir.mkdir()

    (tdir / "code-review.yaml").write_text(
        "name: code-review\n"
        "description: Review code for quality and bugs\n"
        "system_append: |\n"
        "  Focus on security vulnerabilities and performance.\n"
        "  Report issues in severity order.\n"
    )

    (tdir / "refactor.yml").write_text(
        "name: refactor\n"
        "description: Refactor code with clean architecture\n"
        "system_append: |\n"
        "  Apply SOLID principles and design patterns.\n"
        "model: deepseek/deepseek-v4-flash\n"
        "temperature: 0.3\n"
    )

    (tdir / "with-messages.yaml").write_text(
        "name: starter\n"
        "description: Session with starter messages\n"
        "messages:\n"
        "  - role: user\n"
        "    content: Analyze the current project structure\n"
        "  - role: assistant\n"
        "    content: I'll analyze the project structure for you.\n"
    )

    return tdir


@pytest.fixture
def template_dir_with_bad(tmp_path):
    """Template dir with some broken files."""
    tdir = tmp_path / "templates"
    tdir.mkdir()

    (tdir / "good.yaml").write_text(
        "name: good\n"
        "description: A good template\n"
    )

    # Bad: not valid YAML dict
    (tdir / "bad.yaml").write_text(
        "- just\n- a\n- list\n"
    )

    # Bad: syntax error (unbalanced quotes)
    (tdir / "broken.yaml").write_text(
        "name: broken\ndescription: 'unclosed quote\n"
    )

    return tdir


# ── PromptTemplate Tests ────────────────────────────────────────────

class TestPromptTemplate:
    def test_from_dict_minimal(self):
        t = PromptTemplate.from_dict({"name": "test"})
        assert t.name == "test"
        assert t.description == ""
        assert t.system_append == ""
        assert t.model is None
        assert t.temperature is None
        assert t.messages is None

    def test_from_dict_full(self):
        t = PromptTemplate.from_dict({
            "name": "full",
            "description": "Full template",
            "system_append": "Be concise.",
            "model": "kimi/kimi-k2.6",
            "temperature": 0.7,
            "messages": [{"role": "user", "content": "hello"}],
        })
        assert t.name == "full"
        assert t.system_append == "Be concise."
        assert t.model == "kimi/kimi-k2.6"
        assert t.temperature == 0.7
        assert len(t.messages) == 1

    def test_to_dict_roundtrip(self):
        original = PromptTemplate(
            name="test",
            description="desc",
            system_append="append",
            model="model",
            temperature=0.5,
        )
        d = original.to_dict()
        restored = PromptTemplate.from_dict(d)
        assert restored.name == original.name
        assert restored.description == original.description
        assert restored.system_append == original.system_append
        assert restored.model == original.model
        assert restored.temperature == original.temperature

    def test_to_dict_skips_empty(self):
        t = PromptTemplate(name="minimal")
        d = t.to_dict()
        assert "system_append" not in d
        assert "model" not in d
        assert "temperature" not in d
        assert "messages" not in d


# ── TemplateManager Discovery Tests ─────────────────────────────────

class TestTemplateDiscovery:
    def test_discovers_yaml_files(self, template_dir):
        manager = TemplateManager(template_dirs=[str(template_dir)])
        templates = manager.discover()
        names = {t.name for t in templates}
        assert "code-review" in names
        assert "refactor" in names
        assert "starter" in names

    def test_empty_dir(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        manager = TemplateManager(template_dirs=[str(empty)])
        assert manager.discover() == []

    def test_nonexistent_dir(self):
        manager = TemplateManager(template_dirs=["/nonexistent"])
        assert manager.discover() == []

    def test_skips_bad_templates(self, template_dir_with_bad):
        manager = TemplateManager(template_dirs=[str(template_dir_with_bad)])
        templates = manager.discover()
        names = {t.name for t in templates}
        assert "good" in names
        assert len(templates) == 1

    def test_name_collision_first_wins(self, tmp_path):
        dir1 = tmp_path / "d1"
        dir2 = tmp_path / "d2"
        dir1.mkdir()
        dir2.mkdir()

        (dir1 / "t.yaml").write_text("name: t\ndescription: first\n")
        (dir2 / "t.yaml").write_text("name: t\ndescription: second\n")

        manager = TemplateManager(template_dirs=[str(dir1), str(dir2)])
        templates = manager.discover()
        assert len(templates) == 1
        assert templates[0].description == "first"


# ── TemplateManager Get/List Tests ──────────────────────────────────

class TestTemplateManagerOps:
    def test_get_existing(self, template_dir):
        manager = TemplateManager(template_dirs=[str(template_dir)])
        t = manager.get("code-review")
        assert t is not None
        assert t.name == "code-review"
        assert "security" in t.system_append

    def test_get_nonexistent(self, template_dir):
        manager = TemplateManager(template_dirs=[str(template_dir)])
        assert manager.get("nope") is None

    def test_templates_property(self, template_dir):
        manager = TemplateManager(template_dirs=[str(template_dir)])
        assert len(manager.templates) == 3

    def test_refactor_has_model(self, template_dir):
        manager = TemplateManager(template_dirs=[str(template_dir)])
        t = manager.get("refactor")
        assert t.model == "deepseek/deepseek-v4-flash"
        assert t.temperature == 0.3

    def test_starter_has_messages(self, template_dir):
        manager = TemplateManager(template_dirs=[str(template_dir)])
        t = manager.get("starter")
        assert t.messages is not None
        assert len(t.messages) == 2
        assert t.messages[0]["role"] == "user"


# ── Agent Integration Tests ─────────────────────────────────────────

class TestAgentTemplateIntegration:
    def test_apply_template_appends_system_prompt(self, tmp_path, template_dir):
        config = Config(api_key="test-key", template_dirs=[str(template_dir)])
        session = Session(tmp_path / "test.jsonl")
        agent = Agent(config, session, llm=FakeLLM())

        original_prompt_len = len(agent.system_prompt)
        result = agent.apply_template("code-review")
        assert result is not None
        assert len(agent.system_prompt) > original_prompt_len
        assert "security" in agent.system_prompt

    def test_apply_template_injects_messages(self, tmp_path, template_dir):
        config = Config(api_key="test-key", template_dirs=[str(template_dir)])
        session = Session(tmp_path / "test.jsonl")
        agent = Agent(config, session, llm=FakeLLM())

        agent.apply_template("starter")

        user_msgs = [m for m in session.messages if m.get("role") == "user"]
        asst_msgs = [m for m in session.messages if m.get("role") == "assistant"]
        assert len(user_msgs) == 1
        assert "Analyze the current project structure" in user_msgs[0]["content"]
        assert len(asst_msgs) == 1

    def test_apply_nonexistent_template(self, tmp_path, template_dir):
        config = Config(api_key="test-key", template_dirs=[str(template_dir)])
        session = Session(tmp_path / "test.jsonl")
        agent = Agent(config, session, llm=FakeLLM())

        result = agent.apply_template("nope")
        assert result is None

    def test_apply_template_returns_description(self, tmp_path, template_dir):
        config = Config(api_key="test-key", template_dirs=[str(template_dir)])
        session = Session(tmp_path / "test.jsonl")
        agent = Agent(config, session, llm=FakeLLM())

        result = agent.apply_template("refactor")
        assert result == "Refactor code with clean architecture"
