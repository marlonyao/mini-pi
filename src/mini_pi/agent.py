"""
Agent loop for mini-pi.

The core cycle:
  1. Send messages + system prompt to LLM (streaming)
  2. If LLM returns tool calls → execute them → add results → go to 1
  3. If LLM returns text only → done, display to user

Uses streaming for real-time output — you see the response as it's generated.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Generator

from openai import OpenAI

from .compactor import Compactor, CompactionConfig
from .config import Config
from .context import prune_messages, PruningConfig
from .session import Session
from .system_prompt import build_system_prompt
from .tools import execute_tool, get_openai_tools
from .token_estimator import TokenEstimator


class Agent:
    """The coding agent: orchestrates LLM calls and tool execution."""

    def __init__(self, config: Config, session: Session):
        self.config = config
        self.session = session

        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )

        self.tools = get_openai_tools()
        self.system_prompt = build_system_prompt(cwd=config.cwd)

        # Compaction support
        self.compactor = Compactor(config.compaction, client=self.client)
        self.token_estimator = TokenEstimator(
            max_context_tokens=config.compaction.max_context_tokens,
        )

    def chat(self, user_message: str) -> str:
        """
        Process a user message through the agent loop.
        Returns the final assistant text response.
        """
        self.session.add_user(user_message)
        return self._agent_loop()

    def _agent_loop(self) -> str:
        """Run the agent loop until a final text response is produced."""
        for step in range(self.config.max_steps):
            # Auto-compaction: check if context is approaching limit
            self._maybe_compact()

            text_content, tool_calls, usage = self._stream_llm()

            # Track token usage
            if usage:
                self.session.update_token_usage({
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                    "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
                    "total_tokens": getattr(usage, "total_tokens", 0) or 0,
                })

            if tool_calls:
                # Add assistant message with tool calls
                self.session.add_assistant(
                    content=text_content,
                    tool_calls=tool_calls,
                )

                # Execute each tool call
                for tc in tool_calls:
                    fn = tc["function"]
                    args = json.loads(fn["arguments"])
                    print(f"\n  🔧 {fn['name']}({self._format_args(args)})")

                    result = execute_tool(
                        fn["name"],
                        args,
                        timeout=self.config.timeout,
                        cwd=self.config.cwd,
                    )

                    # Show brief result
                    preview = result[:200].replace("\n", " ")
                    if len(result) > 200:
                        preview += "..."
                    print(f"     → {preview}")

                    self.session.add_tool_result(tc["id"], result)

                print()  # spacing before next LLM call
            else:
                # No tool calls — final response
                self.session.add_assistant(content=text_content)
                self.session.save()
                return text_content

        self.session.save()
        return "(agent reached max tool call steps without producing a final response)"

    def _stream_llm(self) -> tuple[str, list[dict], Any]:
        """
        Stream an LLM response. Returns (text, tool_calls, usage).
        
        Text content is printed to stdout in real-time.
        Tool calls are collected but not printed (we print them when executing).
        """
        # Prune old tool results to reduce context size
        raw_messages = self.session.get_openai_messages()
        pruned = prune_messages(raw_messages, self.config.pruning)

        messages = [
            {"role": "system", "content": self.system_prompt},
            *pruned,
        ]

        # Try streaming first; fall back to non-stream if provider doesn't support it
        try:
            stream = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                tools=self.tools,
                temperature=0,
                stream=True,
                stream_options={"include_usage": True},
            )
        except Exception:
            # Fallback to non-streaming
            return self._call_llm_sync()

        text_parts: list[str] = []
        tool_calls_map: dict[int, dict] = {}  # index -> {id, function: {name, arguments}}
        usage = None

        for chunk in stream:
            if not chunk.choices and hasattr(chunk, "usage") and chunk.usage:
                usage = chunk.usage
                continue

            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            # Text content — print in real-time
            if delta.content:
                text_parts.append(delta.content)
                sys.stdout.write(delta.content)
                sys.stdout.flush()

            # Tool calls — collect incrementally
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_calls_map:
                        tool_calls_map[idx] = {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    tc = tool_calls_map[idx]
                    if tc_delta.id:
                        tc["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            tc["function"]["name"] += tc_delta.function.name
                        if tc_delta.function.arguments:
                            tc["function"]["arguments"] += tc_delta.function.arguments

            # Usage from last chunk
            if hasattr(chunk.choices[0], "usage") and chunk.choices[0].usage:
                usage = chunk.choices[0].usage

        text_content = "".join(text_parts)
        tool_calls = [tool_calls_map[i] for i in sorted(tool_calls_map.keys())]

        # Add newline if we streamed text
        if text_content:
            print()  # newline after streamed text

        return text_content, tool_calls, usage

    def _call_llm_sync(self) -> tuple[str, list[dict], Any]:
        """Non-streaming fallback for providers that don't support streaming."""
        # Prune old tool results to reduce context size
        raw_messages = self.session.get_openai_messages()
        pruned = prune_messages(raw_messages, self.config.pruning)

        messages = [
            {"role": "system", "content": self.system_prompt},
            *pruned,
        ]

        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            tools=self.tools,
            temperature=0,
        )

        msg = response.choices[0].message
        text_content = msg.content or ""
        tool_calls = []
        if msg.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]

        if text_content:
            print(text_content)

        return text_content, tool_calls, response.usage

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
            print("\n  🧹 Auto-compacting conversation...")
            result = self.compactor.compact(messages)
            if result.success:
                self.session.record_compaction(result)
                print(f"     Compacted {result.original_count} → {result.compacted_count} messages")
            else:
                print(f"     Compaction skipped: {result.error}")
