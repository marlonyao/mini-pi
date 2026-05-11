"""
Mini Pi — A minimal coding agent in Python.

Usage:
    mini-pi                  # Start interactive session
    mini-pi -c               # Continue most recent session
    mini-pi -p "query"       # One-shot query (print mode)
    mini-pi --session PATH   # Load specific session
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TextIO

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from .agent import Agent
from .config import Config
from .llm import OpenAILLM
from .session import Session, create_session, fork_session, list_sessions

console = Console()


def _load_dotenv_files() -> None:
    """Load ``.env`` from the repository root, then the current working directory (later wins)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    repo_root = Path(__file__).resolve().parents[2]
    paths: list[Path] = []
    repo_env = repo_root / ".env"
    cwd_env = Path.cwd() / ".env"
    if repo_env.is_file():
        paths.append(repo_env)
    if cwd_env.is_file() and (not paths or cwd_env.resolve() != paths[0].resolve()):
        paths.append(cwd_env)
    for i, path in enumerate(paths):
        load_dotenv(path, override=bool(i))


class StreamingDisplay:
    """Writes streamed assistant deltas directly to the real terminal.

    Rich ``Live`` panels render nicely, but in some terminals they only paint their
    final frame when ``stop()`` runs. Direct writes keep interactive mode genuinely
    token-streamed, matching ``-p`` mode.
    """

    def __init__(self, output: TextIO) -> None:
        self.output = output
        self._saw_reasoning = False
        self._answer_opened = False

    def start(self) -> None:
        """Start the live display."""
        self._saw_reasoning = False
        self._answer_opened = False
        self.output.write("🤖 Assistant\n")
        self.flush()

    def add(self, text: str) -> None:
        """Add streamed answer text chunk."""
        if not text:
            return
        if self._saw_reasoning and not self._answer_opened:
            self.output.write("\n\n")
            self._answer_opened = True
        self.output.write(text)
        self.flush()

    def add_reasoning(self, text: str) -> None:
        """Add streamed reasoning content chunk (DeepSeek thinking mode)."""
        if not text:
            return
        self._saw_reasoning = True
        if self.output.isatty():
            self.output.write(f"\033[2;3m{text}\033[0m")
        else:
            self.output.write(text)
        self.flush()

    def flush(self) -> None:
        """Force immediate flush of streamed output."""
        self.output.flush()

    def stop(self) -> None:
        """Stop the live display."""
        self.flush()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="mini-pi",
        description="A minimal coding agent inspired by Pi",
    )
    parser.add_argument("-p", "--print", dest="query", help="One-shot query (non-interactive)")
    parser.add_argument("-c", "--continue", dest="continue_last", action="store_true", help="Continue most recent session")
    parser.add_argument("--session", type=str, help="Path to session file to load")
    parser.add_argument("--model", type=str, help="Model to use (overrides MINI_PI_MODEL env)")
    parser.add_argument("--cwd", type=str, help="Working directory (default: current)")
    args = parser.parse_args()

    _load_dotenv_files()

    # Load config
    config = Config()
    if args.model:
        config.model = args.model
    if args.cwd:
        config.cwd = args.cwd

    issues = config.validate()
    if issues:
        for issue in issues:
            console.print(f"[red]⚠ {issue}[/red]")
        console.print("\nSet environment variables or pass flags. Example:")
        console.print("  export OPENAI_API_KEY=sk-xxx")
        console.print("  mini-pi")
        sys.exit(1)

    # Load or create session
    if args.session:
        session = Session(Path(args.session))
        console.print(f"[dim]Loaded session: {args.session}[/dim]")
    elif args.continue_last:
        sessions = list_sessions(config.session_dir)
        if not sessions:
            console.print("[yellow]No previous sessions found. Starting new one.[/yellow]")
            session = create_session(config.session_dir)
        else:
            session = Session(Path(sessions[0]["path"]))
            console.print(f"[dim]Resuming: {sessions[0]['name']}[/dim]")
    else:
        session = create_session(config.session_dir)

    # Create agent
    agent = Agent(config, session)

    # One-shot mode
    if args.query:
        response = agent.chat(args.query)
        console.print(Panel(Markdown(response), title="🤖 Assistant", border_style="green"))
        return

    # Interactive REPL
    _print_banner(config)
    _repl(agent, config)


def _print_banner(config: Config) -> None:
    """Print startup banner."""
    model_info = config.get_current_model_info()
    if model_info:
        model_display = f"{model_info.provider}/{model_info.model}"
        ctx = f"{model_info.max_context_tokens // 1000}K"
    else:
        model_display = config.model
        ctx = "?"

    # Show available providers
    available = config.model_registry.list_available()
    available_count = len(available)

    console.print(Panel.fit(
        f"[bold cyan]mini-pi[/bold cyan] [dim]v0.3.0[/dim]\n"
        f"Model: [green]{model_display}[/green] [dim]({ctx} ctx)[/dim]\n"
        f"CWD: [dim]{config.cwd}[/dim]\n"
        f"Providers: [dim]{available_count} models available[/dim]\n"
        f"[dim]Streaming enabled ✨[/dim]\n"
        f"Type [bold]exit[/bold] to quit, [bold]/model[/bold] to switch, [bold]/compact[/bold] to compress, [bold]/new[/bold]/[bold]/resume[/bold]/[bold]/fork[/bold] for sessions, [bold]status[/bold] for info",
        title="🤖 Mini Pi",
        border_style="cyan",
    ))


def _repl(agent: Agent, config: Config) -> None:
    """Interactive read-eval-print loop with streaming display."""
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import FileHistory

        history_file = Path(config.session_dir) / ".repl_history"
        history_file.parent.mkdir(parents=True, exist_ok=True)
        prompt_session = PromptSession(history=FileHistory(str(history_file)))
        prompt_func = lambda: prompt_session.prompt("You > ")
    except ImportError:
        prompt_func = lambda: input("You > ")

    while True:
        try:
            user_input = prompt_func()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Bye! 👋[/dim]")
            break

        user_input = user_input.strip()
        if not user_input:
            continue

        if user_input.lower() == "exit":
            console.print("[dim]Bye! 👋[/dim]")
            break

        # Handle /new command — start a new session
        if user_input.lower() == "/new":
            _handle_new_session(agent, config)
            continue

        # Handle /resume command — switch to a different session
        if user_input.lower() == "/resume" or user_input.lower().startswith("/resume "):
            _handle_resume(agent, config, user_input)
            continue

        # Handle /fork command — fork current session
        if user_input.lower() == "/fork" or user_input.lower().startswith("/fork "):
            _handle_fork(agent, config, user_input)
            continue

        # Handle /sessions command — list sessions
        if user_input.lower() == "/sessions":
            _handle_sessions(config)
            continue

        # Handle /model command
        if user_input.lower() == "/model" or user_input.lower().startswith("/model "):
            _handle_model_command(agent, config, user_input)
            continue

        # Handle /models command (list all)
        if user_input.lower() == "/models":
            _handle_models_list(config)
            continue

        # Handle /compact command
        if user_input.lower() == "/compact" or user_input.lower().startswith("/compact "):
            _handle_compact(agent, user_input)
            continue

        # Handle extension commands: /ext:<name> [args]
        if user_input.lower().startswith("/ext:"):
            _handle_extension_command(agent, user_input)
            continue

        if user_input.lower() == "/extensions":
            _handle_extensions_list(agent)
            continue

        if user_input.lower().startswith("/steer "):
            _handle_steer_command(agent, user_input)
            continue

        if user_input.lower() == "/templates":
            _handle_templates_list(agent)
            continue

        if user_input.lower().startswith("/template "):
            _handle_template_apply(agent, user_input)
            continue

        if user_input.lower() == "clear":
            agent.session.messages.clear()
            console.print("[dim]Session cleared.[/dim]")
            continue

        if user_input.lower() == "status":
            usage = agent.session.token_usage
            model_info = config.get_current_model_info()
            if model_info:
                model_str = f"{model_info.provider}/{model_info.model}"
            else:
                model_str = config.model
            console.print(f"[dim]Model: {model_str} | Messages: {len(agent.session.messages)} | "
                         f"Tokens: {usage['total']} (prompt: {usage['prompt']}, completion: {usage['completion']})[/dim]")
            continue

        # Send to agent with streaming display
        try:
            _run_streaming(agent, user_input)
        except KeyboardInterrupt:
            # Instead of just interrupting, offer steering
            _handle_interrupt_with_steering(agent)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


def _handle_compact(agent: Agent, user_input: str) -> None:
    """Handle /compact command — manually trigger context compaction."""
    config = agent.config
    if not config.compaction.enabled:
        console.print("[yellow]Compaction is disabled.[/yellow]")
        return

    messages = agent.session.get_openai_messages()
    if not messages:
        console.print("[dim]No messages to compact.[/dim]")
        return

    # Parse optional custom instructions
    parts = user_input.strip().split(maxsplit=1)
    instructions = parts[1].strip() if len(parts) > 1 else ""

    # Check token usage
    ratio = agent.token_estimator.usage_ratio(messages)
    count = len(messages)
    console.print(f"[dim]Compacting {count} messages (context usage: {ratio:.0%})...[/dim]")

    existing_summary = getattr(agent.session, "_last_compaction_summary", "")
    result = agent.compactor.compact(
        messages,
        existing_summary=existing_summary,
        instructions=instructions,
    )
    if result.success:
        agent.session.record_compaction(result)
        console.print(
            f"[green]✅ Compacted {result.original_count} → {result.compacted_count} messages[/green]"
        )
    else:
        console.print(f"[yellow]Compaction skipped: {result.error}[/yellow]")


def _handle_new_session(agent: Agent, config: Config) -> None:
    """Handle /new command — start a fresh session."""
    # Save current session first
    agent.session.save()

    # Create a new session
    new_session = create_session(config.session_dir)
    agent.session = new_session

    console.print(f"[green]New session started: {new_session.path.stem}[/green]")
    console.print("[dim]Previous session saved.[/dim]")


def _handle_resume(agent: Agent, config: Config, user_input: str) -> None:
    """Handle /resume command — switch to a different session."""
    parts = user_input.strip().split(maxsplit=1)

    if len(parts) == 1:
        # No arg — show session list for selection
        _handle_sessions(config)
        console.print("[dim]Usage: /resume <name-or-number>[/dim]")
        return

    arg = parts[1].strip()
    sessions = list_sessions(config.session_dir)
    if not sessions:
        console.print("[yellow]No sessions found.[/yellow]")
        return

    # Try to match by number (1-based index)
    try:
        idx = int(arg) - 1
        if 0 <= idx < len(sessions):
            target = sessions[idx]
        else:
            console.print(f"[red]Invalid session number: {arg}[/red]")
            return
    except ValueError:
        # Match by name prefix
        matches = [s for s in sessions if s["name"].startswith(arg)]
        if len(matches) == 1:
            target = matches[0]
        elif len(matches) > 1:
            console.print(f"[yellow]Ambiguous: {len(matches)} sessions match '{arg}'[/yellow]")
            for m in matches[:5]:
                console.print(f"  [dim]{m['name']}[/dim]")
            return
        else:
            console.print(f"[red]Session not found: {arg}[/red]")
            return

    # Save current and switch
    agent.session.save()
    new_session = Session(Path(target["path"]))
    agent.session = new_session

    msg_count = len(new_session.messages)
    console.print(f"[green]Resumed: {target['name']}[/green] [dim]({msg_count} messages)[/dim]")

    # Show context summary so user knows what this session is about
    _print_session_summary(new_session)


def _handle_fork(agent: Agent, config: Config, user_input: str) -> None:
    """Handle /fork command — fork the current session."""
    parts = user_input.strip().split(maxsplit=1)
    name = parts[1].strip() if len(parts) > 1 else None

    if not agent.session.messages:
        console.print("[yellow]Nothing to fork — current session is empty.[/yellow]")
        return

    forked = fork_session(agent.session, config.session_dir, name)
    agent.session = forked

    console.print(f"[green]Forked session: {forked.path.stem}[/green]")
    console.print(f"[dim]({len(forked.messages)} messages copied)[/dim]")


def _handle_sessions(config: Config) -> None:
    """Handle /sessions command — list all sessions."""
    sessions = list_sessions(config.session_dir)
    if not sessions:
        console.print("[dim]No sessions found.[/dim]")
        return

    console.print("[bold cyan]Sessions:[/bold cyan]")
    for i, s in enumerate(sessions[:20], 1):
        size_kb = s["size"] / 1024
        modified = s["modified"][:16]  # YYYY-MM-DDTHH:MM

        # Try to get first user message as topic
        topic = _get_session_topic(s["path"])
        topic_str = f" — {topic}" if topic else ""

        console.print(f"  {i:>2}. [dim]{s['name']}[/dim]  [dim]({size_kb:.1f}KB, {modified})[/dim][dim]{topic_str}[/dim]")

    if len(sessions) > 20:
        console.print(f"  [dim]... and {len(sessions) - 20} more[/dim]")


def _get_session_topic(path: str, max_chars: int = 60) -> str | None:
    """Extract the first user message from a session file as topic."""
    try:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("type") == "message":
                data = entry.get("data", {})
                if data.get("role") == "user" and data.get("content", "").strip():
                    topic = data["content"].strip().replace("\n", " ")
                    if len(topic) > max_chars:
                        topic = topic[:max_chars - 3] + "..."
                    return topic
    except (OSError, json.JSONDecodeError):
        pass
    return None


def _replay_history(messages: list[dict], max_messages: int = 10) -> None:
    """
    Replay recent conversation history in a compact format.

    Shows the last `max_messages` messages with role labels and
    truncated content, giving the user immediate context on resume.
    """
    # Show last N messages
    start = max(0, len(messages) - max_messages)
    recent = messages[start:]

    if start > 0:
        console.print(f"  [dim]... ({start} earlier messages skipped) ...[/dim]")

    for msg in recent:
        role = msg.get("role", "")
        content = msg.get("content", "") or ""
        tool_calls = msg.get("tool_calls")

        if role == "system" and "[Compaction Summary]" in content:
            # Show compaction summary
            preview = content.replace("[Compaction Summary]", "").strip()
            preview = preview.replace("\n", " ")[:100]
            console.print(f"  [dim]📋 Summary: {preview}...[/dim]")
            continue

        if role == "user":
            preview = content.strip().replace("\n", " ")
            if len(preview) > 100:
                preview = preview[:97] + "..."
            console.print(f"  [cyan]You:[/cyan] {preview}")

        elif role == "assistant":
            if tool_calls:
                # Summarize tool calls
                tool_names = [tc.get("function", {}).get("name", "?") for tc in tool_calls]
                console.print(f"  [dim]🤖 → {', '.join(tool_names)}[/dim]")
            elif content.strip():
                preview = content.strip().replace("\n", " ")
                if len(preview) > 100:
                    preview = preview[:97] + "..."
                console.print(f"  [green]🤖:[/green] {preview}")

        elif role == "tool":
            preview = content.strip().replace("\n", " ")[:60]
            console.print(f"  [dim]  → {preview}[/dim]")

    console.print()


def _print_session_summary(session: Session) -> None:
    """Print a summary of a session's content for context on resume."""
    messages = session.messages
    if not messages:
        console.print("  [dim](empty session)[/dim]")
        return

    # Show compaction status
    compaction_count = getattr(session, "_compaction_count", 0)
    if compaction_count:
        console.print(f"  [dim](compacted {compaction_count} time(s))[/dim]")

    # Show token usage
    usage = session.token_usage
    if usage["total"] > 0:
        console.print(f"  [dim]Tokens: {usage['total']:,} (prompt: {usage['prompt']:,}, completion: {usage['completion']:,})[/dim]")

    # Replay last N messages so user sees the conversation context
    console.print()
    _replay_history(messages, max_messages=10)


def _handle_model_command(agent: Agent, config: Config, user_input: str) -> None:
    """Handle /model command — switch model or show current."""
    parts = user_input.strip().split(maxsplit=1)

    if len(parts) == 1:
        # No arg: show current and available
        model_info = config.get_current_model_info()
        if model_info:
            console.print(f"[green]Current: {model_info.provider}/{model_info.model}[/green]")
        else:
            console.print(f"[green]Current: {config.model}[/green]")
        console.print("[dim]Usage: /model <provider/model> | /models to list all[/dim]")
        return

    model_spec = parts[1].strip()
    _switch_model(agent, config, model_spec)


def _switch_model(agent: Agent, config: Config, model_spec: str) -> None:
    """Switch the agent to a different model."""
    model_info = config.model_registry.resolve(model_spec)
    if model_info is None:
        console.print(f"[red]Unknown model: {model_spec}[/red]")
        console.print("[dim]Use /models to see available models[/dim]")
        return

    if not model_info.api_key:
        api_key_env = "?"
        # Find the api_key_env for this provider
        prov_data = config.model_registry._providers.get(model_info.provider, {})
        api_key_env = prov_data.get("api_key_env", "API_KEY")
        console.print(f"[red]API key not set for {model_info.provider}[/red]")
        console.print(f"[dim]Set environment variable: [bold]{api_key_env}=<your-key>[/bold][/dim]")
        return

    # Update config
    config._current_model_info = model_info
    config.model = model_info.model

    # Create new LLM instance
    from .models import create_llm, get_model_extra_kwargs
    agent.llm = create_llm(model_info)
    agent._extra_kwargs = get_model_extra_kwargs(model_info)

    # Update compactor client
    if isinstance(agent.llm, OpenAILLM):
        agent.compactor.client = agent.llm.client

    # Update token estimator
    agent.token_estimator.max_context_tokens = model_info.max_context_tokens

    thinking_str = " [dim](thinking on)[/dim]" if model_info.thinking else ""
    console.print(f"[green]Switched to {model_info.provider}/{model_info.model} ({model_info.max_context_tokens // 1000}K ctx){thinking_str}[/green]")


def _handle_models_list(config: Config) -> None:
    """List all configured models."""
    all_models = config.model_registry.list_models()

    if not all_models:
        console.print("[dim]No models configured.[/dim]")
        return

    current_info = config.get_current_model_info()
    current_spec = f"{current_info.provider}/{current_info.model}" if current_info else None

    # Group by provider
    by_provider: dict[str, list] = {}
    for m in all_models:
        by_provider.setdefault(m["provider"], []).append(m)

    for prov_name, models in sorted(by_provider.items()):
        console.print(f"\n[bold cyan]{prov_name}[/bold cyan]")
        for m in models:
            spec = m["spec"]
            ctx = m["max_context_tokens"] // 1000
            key_status = "🔑" if m["api_key_set"] else "❌"
            thinking = " 🧠" if m["thinking"] else ""
            current = " [green]← current[/green]" if spec == current_spec else ""
            console.print(f"  {key_status} {spec} [dim]({ctx}K ctx){thinking}[/dim]{current}")

    console.print("\n[dim]🔑 = API key set  ❌ = API key missing  🧠 = thinking mode[/dim]")
    console.print("[dim]Use /model <provider/model> to switch[/dim]")


def _run_streaming(agent: Agent, user_message: str) -> None:
    """Run agent with a streaming display + background steering thread.

    A background thread reads stdin during agent execution.
    Type `/steer <msg>` at any time to inject guidance mid-execution.
    """
    import io
    import threading

    old_stdout = sys.stdout
    display = StreamingDisplay(old_stdout)
    display.start()

    class StreamCapture(io.TextIOBase):
        """Capture streamed text from the agent, pass to Rich display."""

        def __init__(self, fallback, display: StreamingDisplay):
            self.fallback = fallback
            self.display = display
            self.in_reasoning = False

        def write(self, text: str) -> int:
            if text:
                if self.in_reasoning:
                    self.display.add_reasoning(text)
                else:
                    self.display.add(text)
            return len(text) if text else 0

        def flush(self):
            self.display.flush()
            self.fallback.flush()

        def isatty(self) -> bool:
            return False

    # Background thread: read stdin for steering commands during execution
    stop_event = threading.Event()

    def _stdin_reader():
        """Background thread that reads /steer commands from stdin."""
        while not stop_event.is_set():
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                line = line.strip()
                if line.lower().startswith("/steer "):
                    msg = line[7:].strip()
                    if msg:
                        agent.steer(msg)
                        old_stdout.write(
                            f"\n  📣 Steering queued: {msg[:60]}"
                            f"{'...' if len(msg) > 60 else ''}\n"
                        )
                        old_stdout.flush()
                    else:
                        old_stdout.write("  [dim]Usage: /steer <message>[/dim]\n")
                        old_stdout.flush()
            except Exception:
                break

    reader_thread = threading.Thread(target=_stdin_reader, daemon=True)
    reader_thread.start()

    try:
        sys.stdout = StreamCapture(old_stdout, display)  # type: ignore
        agent.chat(user_message)
    finally:
        stop_event.set()
        sys.stdout = old_stdout
        display.stop()
        console.print()


def _handle_extension_command(agent: Agent, user_input: str) -> None:
    """Handle /ext:<name> commands registered by extensions."""
    parts = user_input[5:].strip().split(maxsplit=1)  # strip "/ext:"
    cmd_name = parts[0] if parts else ""
    cmd_args = parts[1] if len(parts) > 1 else ""

    if not cmd_name:
        _handle_extensions_list(agent)
        return

    ext_commands = agent.extension_manager.get_all_commands()
    if cmd_name in ext_commands:
        ext, handler = ext_commands[cmd_name]
        try:
            result = handler(cmd_args, agent=agent)
            if result:
                console.print(str(result))
        except Exception as e:
            console.print(f"[red]Extension command error ({ext.name}): {e}[/red]")
    else:
        console.print(f"[yellow]Unknown extension command: {cmd_name}[/yellow]")
        console.print("[dim]Use /extensions to list available commands[/dim]")


def _handle_extensions_list(agent: Agent) -> None:
    """List loaded extensions and their capabilities."""
    exts = agent.extension_manager.extensions

    if not exts:
        console.print("[dim]No extensions loaded.[/dim]")
        console.print("[dim]Place .py files in ~/.mini-pi/extensions/ to add extensions[/dim]")
        return

    console.print("[bold cyan]Extensions:[/bold cyan]")
    for ext in exts:
        parts = [f"[green]{ext.name}[/green]"]
        if ext.handlers:
            events = ", ".join(sorted(ext.handlers.keys()))
            parts.append(f"[dim]events: {events}[/dim]")
        if ext.tools:
            tool_names = ", ".join(t["function"]["name"] for t in ext.tools)
            parts.append(f"[dim]tools: {tool_names}[/dim]")
        if ext.commands:
            cmd_names = ", ".join(f"/ext:{n}" for n in sorted(ext.commands.keys()))
            parts.append(f"[dim]cmds: {cmd_names}[/dim]")
        console.print("  • " + " | ".join(parts))


def _handle_steer_command(agent: Agent, user_input: str) -> None:
    """Queue a steering message for the running agent loop."""
    msg = user_input[7:].strip()  # strip "/steer "
    if not msg:
        console.print("[dim]Usage: /steer <message> — inject guidance into the running agent[/dim]")
        return
    agent.steer(msg)
    console.print(f"[green]✓ Steering message queued: {msg[:60]}{'...' if len(msg) > 60 else ''}[/green]")


def _handle_interrupt_with_steering(agent: Agent) -> None:
    """Handle Ctrl+C during agent execution."""
    console.print("\n[yellow]Agent interrupted.[/yellow]")
    console.print("[dim]Type /steer <msg> to inject guidance and continue[/dim]")


def _handle_templates_list(agent: Agent) -> None:
    """List available prompt templates."""
    templates = agent.template_manager.templates
    if not templates:
        console.print("[dim]No templates found.[/dim]")
        console.print("[dim]Place .yaml files in ~/.mini-pi/templates/ to add templates[/dim]")
        return

    console.print("[bold cyan]Templates:[/bold cyan]")
    for t in templates:
        parts = [f"[green]{t.name}[/green]"]
        if t.description:
            parts.append(f"[dim]{t.description}[/dim]")
        if t.model:
            parts.append(f"[dim]model: {t.model}[/dim]")
        console.print("  • " + " | ".join(parts))
    console.print("\n[dim]Use /template <name> to apply a template[/dim]")


def _handle_template_apply(agent: Agent, user_input: str) -> None:
    """Apply a prompt template to the current session."""
    name = user_input[10:].strip()  # strip "/template "
    if not name:
        _handle_templates_list(agent)
        return

    result = agent.apply_template(name)
    if result is None:
        console.print(f"[yellow]Template not found: {name}[/yellow]")
        _handle_templates_list(agent)
    else:
        console.print(f"[green]✓ Template '{name}' applied: {result}[/green]")


if __name__ == "__main__":
    main()
