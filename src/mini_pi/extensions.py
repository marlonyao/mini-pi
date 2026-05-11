"""
Extension system for mini-pi.

Provides a lightweight hook mechanism inspired by Pi Coding Agent's extensions.
Extensions are Python modules that can register event handlers, custom tools,
and commands without modifying mini-pi internals.

Lifecycle events (in order of typical occurrence):
  - on_start: Agent session starts
  - on_before_llm_call: Before sending messages to LLM
  - on_after_llm_call: After receiving LLM response
  - on_before_tool_call: Before executing a tool
  - on_after_tool_call: After a tool returns
  - on_message: Any message added to session
  - on_before_compact: Before context compaction
  - on_after_compact: After compaction completes
  - on_final_response: Agent produces final text response
  - on_end: Agent session ends

Extension directory structure:
  ~/.mini-pi/extensions/
  └── my-extension.py         # Single file extension
  └── my-extension/           # Or directory
      ├── __init__.py          # Must export extension(config) function
      └── helpers.py

Usage in extension:
  # ~/.mini-pi/extensions/my-extension.py
  def extension(config):
      return {
          "name": "my-extension",
          "on_after_tool_call": lambda ctx: print(f"Tool: {ctx.tool_name}"),
      }
"""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


# ── Event Context ───────────────────────────────────────────────────

@dataclass
class EventContext:
    """Context passed to event handlers.

    Handlers can read any field and mutate some (messages, extra_kwargs).
    """
    event: str                          # Event name
    agent: Any = None                   # Reference to Agent instance (read-only recommended)

    # Tool call fields (on_before_tool_call, on_after_tool_call)
    tool_name: str = ""
    tool_args: dict | None = None
    tool_result: str | None = None      # Only in on_after_tool_call

    # LLM fields (on_before_llm_call, on_after_llm_call)
    messages: list[dict] | None = None  # Can be mutated in on_before_llm_call
    response: Any = None                # ChatResponse in on_after_llm_call
    extra_kwargs: dict | None = None    # Can be mutated in on_before_llm_call

    # Compaction fields
    compaction_result: Any = None       # CompactResult in on_after_compact

    # Generic
    data: dict = field(default_factory=dict)  # Extension-specific data


# ── Extension Definition ────────────────────────────────────────────

@dataclass
class Extension:
    """A loaded extension with its event handlers."""
    name: str
    handlers: dict[str, Callable] = field(default_factory=dict)
    tools: list[dict] = field(default_factory=list)          # OpenAI tool definitions
    tool_executors: dict[str, Callable] = field(default_factory=dict)  # name → executor
    commands: dict[str, Callable] = field(default_factory=dict)  # name → handler
    path: Path | None = None


# ── Extension Manager ───────────────────────────────────────────────

class ExtensionManager:
    """
    Discovers, loads, and dispatches events to extensions.

    Extension discovery order:
    1. ~/.mini-pi/extensions/ (global)
    2. .mini-pi/extensions/ (project-level)
    """

    def __init__(self, extension_dirs: list[str] | None = None):
        self.extension_dirs = extension_dirs or [
            str(Path.home() / ".mini-pi" / "extensions"),
            str(Path.cwd() / ".mini-pi" / "extensions"),
        ]
        self._extensions: list[Extension] = []
        self._discovered = False

    # ── Discovery & Loading ──────────────────────────────────────

    def discover(self) -> list[Extension]:
        """Scan extension directories and load all valid extensions."""
        self._extensions = []
        seen_names: set[str] = set()

        for dir_path in self.extension_dirs:
            path = Path(dir_path)
            if not path.is_dir():
                continue

            for entry in sorted(path.iterdir()):
                ext = self._load_extension(entry)
                if ext and ext.name not in seen_names:
                    self._extensions.append(ext)
                    seen_names.add(ext.name)

        self._discovered = True
        return self._extensions

    def _load_extension(self, path: Path) -> Extension | None:
        """Load a single extension from a file or directory."""
        if path.is_file() and path.suffix == ".py":
            return self._load_module(path, path.stem)
        elif path.is_dir() and (path / "__init__.py").exists():
            return self._load_module(path / "__init__.py", path.name)
        return None

    def _load_module(self, module_path: Path, name: str) -> Extension | None:
        """Load an extension module and extract its config."""
        try:
            spec = importlib.util.spec_from_file_location(
                f"mini_pi_ext_{name}",
                module_path,
            )
            if spec is None or spec.loader is None:
                return None

            module = importlib.util.module_from_spec(spec)

            # Register in sys.modules so tests (and reload) can find it
            import sys
            mod_name = f"mini_pi_ext_{name}"
            sys.modules[mod_name] = module

            spec.loader.exec_module(module)

            # Extension must define an extension() function
            if not hasattr(module, "extension"):
                return None

            # Call extension() — config param is optional
            import inspect
            sig = inspect.signature(module.extension)
            if len(sig.parameters) > 0:
                config_result = module.extension(None)
            else:
                config_result = module.extension()

            if not isinstance(config_result, dict):
                return None

            ext_name = config_result.get("name", name)

            # Extract handlers (on_xxx functions)
            handlers = {}
            for key, value in config_result.items():
                if key.startswith("on_") and callable(value):
                    handlers[key] = value

            # Extract tool definitions
            tools = config_result.get("tools", [])
            tool_executors = config_result.get("tool_executors", {})

            # Extract commands
            commands = config_result.get("commands", {})

            return Extension(
                name=ext_name,
                handlers=handlers,
                tools=tools,
                tool_executors=tool_executors,
                commands=commands,
                path=module_path.parent if module_path.name == "__init__.py" else module_path,
            )

        except Exception as e:
            print(f"  ⚠ Failed to load extension {name}: {e}")
            return None

    # ── Event Dispatch ───────────────────────────────────────────

    def emit(self, event: str, ctx: EventContext | None = None) -> EventContext:
        """
        Emit an event to all extensions.

        Handlers are called in discovery order. Each handler receives
        the same EventContext, which can be mutated.

        Returns the (possibly mutated) EventContext.
        """
        if ctx is None:
            ctx = EventContext(event=event)
        else:
            ctx.event = event

        for ext in self._extensions:
            handler = ext.handlers.get(event)
            if handler:
                try:
                    handler(ctx)
                except Exception as e:
                    print(f"  ⚠ Extension {ext.name} error in {event}: {e}")

        return ctx

    # ── Tool Aggregation ─────────────────────────────────────────

    def get_all_tools(self) -> list[dict]:
        """Get OpenAI tool definitions from all extensions."""
        tools = []
        for ext in self._extensions:
            tools.extend(ext.tools)
        return tools

    def get_tool_executor(self, tool_name: str) -> Callable | None:
        """Find the executor for a tool provided by an extension."""
        for ext in self._extensions:
            if tool_name in ext.tool_executors:
                return ext.tool_executors[tool_name]
        return None

    def get_all_commands(self) -> dict[str, tuple[Extension, Callable]]:
        """Get all registered commands: name → (extension, handler)."""
        commands = {}
        for ext in self._extensions:
            for name, handler in ext.commands.items():
                if name not in commands:
                    commands[name] = (ext, handler)
        return commands

    # ── Properties ───────────────────────────────────────────────

    @property
    def extensions(self) -> list[Extension]:
        if not self._discovered:
            self.discover()
        return self._extensions

    def format_status(self) -> str:
        """Format extension status for display."""
        if not self.extensions:
            return "No extensions loaded."

        lines = []
        for ext in self.extensions:
            parts = [ext.name]
            if ext.handlers:
                parts.append(f"events: {', '.join(sorted(ext.handlers.keys()))}")
            if ext.tools:
                tool_names = [t["function"]["name"] for t in ext.tools]
                parts.append(f"tools: {', '.join(tool_names)}")
            if ext.commands:
                parts.append(f"cmds: {', '.join(sorted(ext.commands.keys()))}")
            lines.append("  • " + " | ".join(parts))

        return "\n".join(lines)
