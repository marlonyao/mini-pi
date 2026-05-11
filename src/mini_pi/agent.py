"""
Agent loop for mini-pi.

The core cycle:
  1. Send messages + system prompt to LLM (via LLMBase abstraction)
  2. If LLM returns tool calls → execute them → add results → go to 1
  3. If LLM returns text only → done, display to user

The LLMBase abstraction handles streaming internally and returns
complete ChatResponse objects. Real-time output uses callbacks.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from .compactor import Compactor, CompactionConfig
from .config import Config
from .context import prune_messages, PruningConfig
from .extensions import ExtensionManager, EventContext
from .llm import LLMBase, OpenAILLM
from .models import create_llm
from .session import Session
from .skills import SkillManager
from .system_prompt import build_system_prompt
from .templates import TemplateManager
from .tools import execute_tool, get_openai_tools
from .token_estimator import TokenEstimator


class Agent:
    """The coding agent: orchestrates LLM calls and tool execution."""

    def __init__(self, config: Config, session: Session, llm: LLMBase | None = None):
        self.config = config
        self.session = session

        # LLM abstraction — inject, resolve from registry, or legacy fallback
        if llm is not None:
            self.llm = llm
        else:
            model_info = config.get_current_model_info()
            if model_info:
                self.llm = create_llm(model_info)
            else:
                self.llm = OpenAILLM(
                    api_key=config.api_key,
                    base_url=config.base_url,
                    model=config.model,
                )

        self.tools = get_openai_tools()
        self.system_prompt = build_system_prompt(cwd=config.cwd)

        # Provider-specific kwargs (thinking mode, temperature, etc.)
        self._extra_kwargs = config.get_extra_kwargs()

        # Skill support (Progressive Disclosure)
        # Only load skill catalog (name + description) at startup.
        # Full SKILL.md is loaded on-demand by LLM via `read` tool.
        self.skill_manager = SkillManager(skill_dirs=config.skill_dirs)
        self.skill_manager.discover()

        # Inject lightweight skill catalog into system prompt
        skill_catalog = self.skill_manager.format_skills_for_prompt()
        if skill_catalog:
            self.system_prompt += skill_catalog

        # Register skill tools (from skills that define tools.py)
        skill_tools, self._skill_executors = self.skill_manager.get_all_skill_tools()
        if skill_tools:
            self.tools.extend(skill_tools)

        # Steering: mid-execution user interventions
        self._steering_messages: list[str] = []

        # Prompt templates
        self.template_manager = TemplateManager(
            template_dirs=config.template_dirs,
        )
        self.template_manager.discover()

        # Extension system — hooks, custom tools, commands
        self.extension_manager = ExtensionManager()
        self.extension_manager.discover()

        # Register extension tools alongside core tools
        ext_tools = self.extension_manager.get_all_tools()
        if ext_tools:
            self.tools.extend(ext_tools)

        # Compaction support — reuse the LLM's OpenAI client for summarization
        # (compactor still uses raw OpenAI client for simplicity)
        self.compactor = Compactor(config.compaction)
        if isinstance(self.llm, OpenAILLM):
            self.compactor.client = self.llm.client

        self.token_estimator = TokenEstimator(
            max_context_tokens=config.compaction.max_context_tokens,
        )

        # Emit startup event
        self.extension_manager.emit("on_start", EventContext(
            event="on_start", agent=self,
        ))

    def chat(self, user_message: str) -> str:
        """
        Process a user message through the agent loop.
        Returns the final assistant text response.
        """
        self.session.add_user(user_message)
        return self._agent_loop()

    def steer(self, message: str) -> None:
        """
        Inject a steering message into the running agent loop.

        The message is queued and consumed at the next loop iteration,
        appearing as a user message that guides the agent's behavior.
        """
        self._steering_messages.append(message)

    def apply_template(self, template_name: str) -> str | None:
        """
        Apply a prompt template to the current session.

        - Appends to the system prompt
        - Optionally switches model
        - Optionally injects starter messages

        Returns the template description, or None if not found.
        """
        template = self.template_manager.get(template_name)
        if template is None:
            return None

        if template.system_append:
            self.system_prompt += f"\n\n{template.system_append}"

        if template.messages:
            for msg in template.messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    self.session.add_user(content)
                elif role == "assistant":
                    self.session.add_assistant(content)

        return template.description or template.name

    def _agent_loop(self) -> str:
        """Run the agent loop until a final text response is produced."""
        for step in range(self.config.max_steps):
            # Consume any steering messages injected mid-execution
            while self._steering_messages:
                msg = self._steering_messages.pop(0)
                self.session.add_user(msg)
                print(f"\n  📣 Steering: {msg[:80]}{'...' if len(msg) > 80 else ''}")
            # Auto-compaction: check if context is approaching limit
            self._maybe_compact()

            # Build messages for this turn
            raw_messages = self.session.get_openai_messages()
            pruned = prune_messages(raw_messages, self.config.pruning)
            messages = [
                {"role": "system", "content": self.system_prompt},
                *pruned,
            ]

            # Extension hook: on_before_llm_call (can mutate messages/kwargs)
            ctx = self.extension_manager.emit("on_before_llm_call", EventContext(
                event="on_before_llm_call", agent=self,
                messages=messages, extra_kwargs=self._extra_kwargs,
            ))
            # Use possibly-mutated messages and kwargs
            messages = ctx.messages or messages
            call_kwargs = ctx.extra_kwargs if ctx.extra_kwargs else self._extra_kwargs

            # Call LLM via abstraction (with provider-specific kwargs)
            response = self.llm.chat(
                messages=messages,
                tools=self.tools,
                on_text=self._on_text,
                on_reasoning=self._on_reasoning,
                **call_kwargs,
            )

            # Extension hook: on_after_llm_call
            self.extension_manager.emit("on_after_llm_call", EventContext(
                event="on_after_llm_call", agent=self,
                response=response, messages=messages,
            ))

            # Track token usage
            if response.usage:
                self.session.update_token_usage(response.usage.to_dict())

            if response.has_tool_calls:
                # Add assistant message with tool calls
                assistant_kwargs: dict[str, Any] = {
                    "content": response.content,
                    "tool_calls": [tc.to_openai_dict() for tc in response.tool_calls],
                }
                if response.reasoning_content:
                    assistant_kwargs["reasoning_content"] = response.reasoning_content
                self.session.add_assistant(**assistant_kwargs)

                # Execute each tool call
                for tc in response.tool_calls:
                    args = json.loads(tc.arguments)
                    print(f"\n  🔧 {tc.name}({self._format_args(args)})")

                    # Extension hook: on_before_tool_call
                    self.extension_manager.emit("on_before_tool_call", EventContext(
                        event="on_before_tool_call", agent=self,
                        tool_name=tc.name, tool_args=args,
                    ))

                    # Try extension tool → skill tool → core tools
                    ext_executor = self.extension_manager.get_tool_executor(tc.name)
                    if ext_executor:
                        try:
                            result = ext_executor(args, cwd=self.config.cwd)
                        except Exception as e:
                            result = f"Error: {e}"
                    elif tc.name in self._skill_executors:
                        _skill, handler = self._skill_executors[tc.name]
                        try:
                            result = handler(args, cwd=self.config.cwd)
                        except Exception as e:
                            result = f"Error: {e}"
                    else:
                        result = execute_tool(
                            tc.name,
                            args,
                            timeout=self.config.timeout,
                            cwd=self.config.cwd,
                        )

                    # Extension hook: on_after_tool_call
                    self.extension_manager.emit("on_after_tool_call", EventContext(
                        event="on_after_tool_call", agent=self,
                        tool_name=tc.name, tool_args=args,
                        tool_result=result,
                    ))

                    # Show brief result
                    preview = result[:200].replace("\n", " ")
                    if len(result) > 200:
                        preview += "..."
                    print(f"     → {preview}")

                    self.session.add_tool_result(tc.id, result)

                print()  # spacing before next LLM call
            else:
                # No tool calls — final response
                assistant_kwargs = {"content": response.content}
                if response.reasoning_content:
                    assistant_kwargs["reasoning_content"] = response.reasoning_content
                self.session.add_assistant(**assistant_kwargs)

                # Extension hook: on_final_response
                self.extension_manager.emit("on_final_response", EventContext(
                    event="on_final_response", agent=self,
                    response=response,
                ))

                self.session.save()
                return response.content

        self.session.save()
        return "(agent reached max tool call steps without producing a final response)"

    def _on_text(self, text: str) -> None:
        """Callback: print text as it streams in."""
        sys.stdout.write(text)
        sys.stdout.flush()

    def _on_reasoning(self, text: str) -> None:
        """Callback: print reasoning content (DeepSeek thinking mode)."""
        sys.stdout.write(text)
        sys.stdout.flush()

    @staticmethod
    def _format_args(args: dict) -> str:
        """Format tool arguments for display."""
        parts = []
        for k, v in args.items():
            v_str = str(v)
            if len(v_str) > 60:
                v_str = v_str[:57] + "..."
            parts.append(f"{k}={v_str!r}")
        return ", ".join(parts)

    def _maybe_compact(self) -> None:
        """Check if context needs compaction and run it if so."""
        config = self.config.compaction
        if not config.enabled:
            return

        messages = self.session.get_openai_messages()
        if self.token_estimator.should_compact(messages, threshold=config.threshold):
            # Extension hook: on_before_compact
            self.extension_manager.emit("on_before_compact", EventContext(
                event="on_before_compact", agent=self,
                messages=messages,
            ))

            print("\n  🧹 Auto-compacting conversation...")

            # Support incremental update if we have a previous summary
            existing_summary = getattr(self.session, "_last_compaction_summary", "")

            result = self.compactor.compact(
                messages,
                existing_summary=existing_summary,
            )
            if result.success:
                self.session.record_compaction(result)
                print(f"     Compacted {result.original_count} → {result.compacted_count} messages")
            else:
                print(f"     Compaction skipped: {result.error}")

            # Extension hook: on_after_compact
            self.extension_manager.emit("on_after_compact", EventContext(
                event="on_after_compact", agent=self,
                compaction_result=result,
            ))
