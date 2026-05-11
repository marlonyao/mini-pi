"""
Tests for the extension system.

Covers:
- Extension discovery and loading
- Event dispatch
- Tool registration from extensions
- Command registration
- Error handling (bad extensions don't crash the agent)
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock

from mini_pi.extensions import Extension, ExtensionManager, EventContext


# ── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def ext_dir(tmp_path):
    """Create a temporary extension directory with sample extensions."""
    exts = tmp_path / "extensions"
    exts.mkdir()

    # Simple event-only extension
    (exts / "logger.py").write_text(
        "log = []\n\n"
        "def extension(config):\n"
        "    return {\n"
        "        'name': 'logger',\n"
        "        'on_before_tool_call': lambda ctx: log.append(f'before:{ctx.tool_name}'),\n"
        "        'on_after_tool_call': lambda ctx: log.append(f'after:{ctx.tool_name}:{ctx.tool_result}'),\n"
        "    }\n"
    )

    # Extension with tools
    (exts / "tools-ext.py").write_text(
        "from pydantic import BaseModel, Field\n\n"
        "class CountParams(BaseModel):\n"
        "    '''Count things.'''\n"
        "    what: str = Field(description='What to count')\n\n"
        "def tool_count(params, **kwargs):\n"
        "    return f'Counting {params[\"what\"]}'\n\n"
        "def extension(config):\n"
        "    return {\n"
        "        'name': 'counter',\n"
        "        'tools': [{\n"
        "            'type': 'function',\n"
        "            'function': {\n"
        "                'name': 'count',\n"
        "                'description': 'Count things',\n"
        "                'parameters': {'type': 'object', 'properties': {'what': {'type': 'string'}}},\n"
        "            },\n"
        "        }],\n"
        "        'tool_executors': {'count': tool_count},\n"
        "    }\n"
    )

    # Extension with commands
    (exts / "cmd-ext.py").write_text(
        "def extension(config):\n"
        "    return {\n"
        "        'name': 'cmd-demo',\n"
        "        'commands': {\n"
        "            'hello': lambda args, **kw: f'Hello, {args}!' if args else 'Hello!',\n"
        "        },\n"
        "    }\n"
    )

    return exts


@pytest.fixture
def ext_dir_with_bad(tmp_path):
    """Extension dir with one good and one bad extension."""
    exts = tmp_path / "extensions"
    exts.mkdir()

    (exts / "good.py").write_text(
        "def extension(config):\n"
        "    return {'name': 'good', 'on_start': lambda ctx: None}\n"
    )

    # Bad: no extension() function
    (exts / "bad_no_func.py").write_text(
        "# This file has no extension() function\n"
        "x = 42\n"
    )

    # Bad: extension() raises exception
    (exts / "bad_crash.py").write_text(
        "def extension(config):\n"
        "    raise RuntimeError('I crash on load')\n"
    )

    # Bad: extension() returns non-dict
    (exts / "bad_return.py").write_text(
        "def extension(config):\n"
        "    return 'not a dict'\n"
    )

    return exts


@pytest.fixture
def dir_ext(tmp_path):
    """Extension as a directory with __init__.py."""
    exts = tmp_path / "extensions"
    exts.mkdir()

    dir_ext = exts / "my-dir-ext"
    dir_ext.mkdir()
    (dir_ext / "__init__.py").write_text(
        "def extension(config):\n"
        "    return {\n"
        "        'name': 'dir-extension',\n"
        "        'on_start': lambda ctx: None,\n"
        "    }\n"
    )

    return exts


# ── Discovery Tests ─────────────────────────────────────────────────

class TestDiscovery:
    def test_discover_finds_extensions(self, ext_dir):
        manager = ExtensionManager(extension_dirs=[str(ext_dir)])
        exts = manager.discover()
        names = {e.name for e in exts}
        assert "logger" in names
        assert "counter" in names
        assert "cmd-demo" in names

    def test_discover_empty_dir(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        manager = ExtensionManager(extension_dirs=[str(empty)])
        exts = manager.discover()
        assert exts == []

    def test_discover_nonexistent_dir(self):
        manager = ExtensionManager(extension_dirs=["/nonexistent/path"])
        exts = manager.discover()
        assert exts == []

    def test_discover_directory_extension(self, dir_ext):
        manager = ExtensionManager(extension_dirs=[str(dir_ext)])
        exts = manager.discover()
        assert len(exts) == 1
        assert exts[0].name == "dir-extension"

    def test_skips_bad_extensions(self, ext_dir_with_bad):
        manager = ExtensionManager(extension_dirs=[str(ext_dir_with_bad)])
        exts = manager.discover()
        names = {e.name for e in exts}
        assert "good" in names
        assert len(exts) == 1  # Only the good one loaded

    def test_name_collision_first_wins(self, tmp_path):
        dir1 = tmp_path / "dir1"
        dir2 = tmp_path / "dir2"
        dir1.mkdir()
        dir2.mkdir()

        (dir1 / "same.py").write_text(
            "def extension(config):\n"
            "    return {'name': 'same', 'on_start': lambda ctx: None}\n"
        )
        (dir2 / "same.py").write_text(
            "def extension(config):\n"
            "    return {'name': 'same', 'on_end': lambda ctx: None}\n"
        )

        manager = ExtensionManager(extension_dirs=[str(dir1), str(dir2)])
        exts = manager.discover()
        assert len(exts) == 1
        assert "on_start" in exts[0].handlers

    def test_skips_non_py_files(self, tmp_path):
        exts = tmp_path / "extensions"
        exts.mkdir()
        (exts / "readme.md").write_text("# Not an extension")
        (exts / "data.json").write_text("{}")

        manager = ExtensionManager(extension_dirs=[str(exts)])
        assert manager.discover() == []


# ── Event Dispatch Tests ────────────────────────────────────────────

class TestEventDispatch:
    def test_emit_calls_handler(self, ext_dir):
        manager = ExtensionManager(extension_dirs=[str(ext_dir)])
        manager.discover()

        ctx = EventContext(event="on_before_tool_call", tool_name="bash")
        manager.emit("on_before_tool_call", ctx)

        # Logger extension should have recorded the event
        logger_mod = _get_loaded_module(manager, "logger")
        assert "before:bash" in logger_mod.log

    def test_emit_without_ctx_creates_default(self):
        manager = ExtensionManager()
        # No extensions → just returns a fresh context
        ctx = manager.emit("on_start")
        assert ctx.event == "on_start"

    def test_emit_to_multiple_extensions(self, tmp_path):
        exts = tmp_path / "extensions"
        exts.mkdir()

        for i in range(3):
            (exts / f"ext{i}.py").write_text(
                f"results = []\n\n"
                f"def extension(config):\n"
                f"    return {{\n"
                f"        'name': 'ext{i}',\n"
                f"        'on_start': lambda ctx: results.append('ext{i}'),\n"
                f"    }}\n"
            )

        manager = ExtensionManager(extension_dirs=[str(exts)])
        manager.discover()
        manager.emit("on_start")

        # All three should have been called
        for i in range(3):
            mod = _get_loaded_module(manager, f"ext{i}")
            assert mod is not None
            assert f"ext{i}" in mod.results

    def test_emit_handler_error_doesnt_crash(self, tmp_path):
        exts = tmp_path / "extensions"
        exts.mkdir()

        (exts / "crasher.py").write_text(
            "def extension(config):\n"
            "    return {\n"
            "        'name': 'crasher',\n"
            "        'on_start': lambda ctx: 1/0,\n"
            "    }\n"
        )
        (exts / "ok.py").write_text(
            "called = []\n"
            "def extension(config):\n"
            "    return {\n"
            "        'name': 'ok',\n"
            "        'on_start': lambda ctx: called.append(True),\n"
            "    }\n"
        )

        manager = ExtensionManager(extension_dirs=[str(exts)])
        manager.discover()

        # Should not raise, and ok extension should still be called
        manager.emit("on_start")

        ok_mod = _get_loaded_module(manager, "ok")
        assert len(ok_mod.called) == 1

    def test_event_context_mutation(self, tmp_path):
        exts = tmp_path / "extensions"
        exts.mkdir()

        (exts / "mutator.py").write_text(
            "def extension(config):\n"
            "    def on_before_llm_call(ctx):\n"
            "        if ctx.extra_kwargs:\n"
            "            ctx.extra_kwargs['temperature'] = 0.5\n"
            "    return {\n"
            "        'name': 'mutator',\n"
            "        'on_before_llm_call': on_before_llm_call,\n"
            "    }\n"
        )

        manager = ExtensionManager(extension_dirs=[str(exts)])
        manager.discover()

        ctx = EventContext(
            event="on_before_llm_call",
            extra_kwargs={"temperature": 0},
        )
        result = manager.emit("on_before_llm_call", ctx)
        assert result.extra_kwargs["temperature"] == 0.5


# ── Tool Registration Tests ─────────────────────────────────────────

class TestToolRegistration:
    def test_get_all_tools(self, ext_dir):
        manager = ExtensionManager(extension_dirs=[str(ext_dir)])
        manager.discover()

        tools = manager.get_all_tools()
        tool_names = {t["function"]["name"] for t in tools}
        assert "count" in tool_names

    def test_get_tool_executor(self, ext_dir):
        manager = ExtensionManager(extension_dirs=[str(ext_dir)])
        manager.discover()

        executor = manager.get_tool_executor("count")
        assert executor is not None
        result = executor({"what": "lines"})
        assert "Counting lines" in result

    def test_get_tool_executor_not_found(self, ext_dir):
        manager = ExtensionManager(extension_dirs=[str(ext_dir)])
        manager.discover()

        assert manager.get_tool_executor("nonexistent") is None

    def test_no_tools_when_no_extensions(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        manager = ExtensionManager(extension_dirs=[str(empty)])
        manager.discover()
        assert manager.get_all_tools() == []


# ── Command Registration Tests ──────────────────────────────────────

class TestCommandRegistration:
    def test_get_all_commands(self, ext_dir):
        manager = ExtensionManager(extension_dirs=[str(ext_dir)])
        manager.discover()

        commands = manager.get_all_commands()
        assert "hello" in commands
        ext, handler = commands["hello"]
        assert ext.name == "cmd-demo"

    def test_command_execution(self, ext_dir):
        manager = ExtensionManager(extension_dirs=[str(ext_dir)])
        manager.discover()

        commands = manager.get_all_commands()
        _, handler = commands["hello"]
        result = handler("world")
        assert "Hello, world!" in result

    def test_command_with_empty_args(self, ext_dir):
        manager = ExtensionManager(extension_dirs=[str(ext_dir)])
        manager.discover()

        commands = manager.get_all_commands()
        _, handler = commands["hello"]
        result = handler("")
        assert "Hello!" in result


# ── Status Display Tests ────────────────────────────────────────────

class TestStatusDisplay:
    def test_format_status_empty(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        manager = ExtensionManager(extension_dirs=[str(empty)])
        manager.discover()
        assert "No extensions" in manager.format_status()

    def test_format_status_shows_extensions(self, ext_dir):
        manager = ExtensionManager(extension_dirs=[str(ext_dir)])
        manager.discover()
        status = manager.format_status()
        assert "logger" in status
        assert "counter" in status
        assert "cmd-demo" in status


# ── EventContext Tests ──────────────────────────────────────────────

class TestEventContext:
    def test_default_values(self):
        ctx = EventContext(event="on_start")
        assert ctx.tool_name == ""
        assert ctx.tool_args is None
        assert ctx.tool_result is None
        assert ctx.messages is None
        assert ctx.data == {}

    def test_mutation(self):
        ctx = EventContext(event="on_before_tool_call")
        ctx.tool_name = "bash"
        ctx.data["custom"] = "value"
        assert ctx.tool_name == "bash"
        assert ctx.data["custom"] == "value"


# ── Helper ──────────────────────────────────────────────────────────

def _get_loaded_module(manager: ExtensionManager, name: str):
    """Get the loaded Python module for an extension by name."""
    import sys
    for ext in manager._extensions:
        mod_name = f"mini_pi_ext_{name}"
        if mod_name in sys.modules:
            return sys.modules[mod_name]
    return None
