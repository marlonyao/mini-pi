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
import sys
from pathlib import Path
from typing import TextIO

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from .agent import Agent
from .config import Config
from .llm import OpenAILLM
from .session import Session, create_session, list_sessions

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
        f"Type [bold]exit[/bold] to quit, [bold]/model[/bold] to switch, [bold]status[/bold] for info",
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

        # Handle /model command
        if user_input.lower() == "/model" or user_input.lower().startswith("/model "):
            _handle_model_command(agent, config, user_input)
            continue

        # Handle /models command (list all)
        if user_input.lower() == "/models":
            _handle_models_list(config)
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
            console.print("\n[yellow]Interrupted.[/yellow]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


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
    """Run agent with a streaming display.

    We monkey-patch sys.stdout to capture streamed text chunks
    from the agent, and write them to the original stdout immediately.
    """
    import io

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

    try:
        sys.stdout = StreamCapture(old_stdout, display)  # type: ignore
        agent.chat(user_message)
    finally:
        sys.stdout = old_stdout
        display.stop()
        console.print()


if __name__ == "__main__":
    main()
