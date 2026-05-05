"""
Compaction for mini-pi.

When the conversation approaches the model's context window limit, compaction
summarizes older messages into a condensed summary so the conversation can continue.

Inspired by Pi's compaction strategy:
- Split messages into "old" (to summarize) and "recent" (to keep)
- Send old messages to LLM for summarization
- Replace old messages with a single compaction summary
- Full history is preserved on disk (never lost)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from .token_estimator import TokenEstimator


@dataclass
class CompactionConfig:
    """Configuration for context compaction."""

    enabled: bool = True
    keep_recent_messages: int = 6      # Number of recent messages to keep intact
    threshold: float = 0.8             # Context usage ratio that triggers compaction
    max_context_tokens: int = 128000   # Model context window size
    model: str | None = None           # Model for summarization (None = use session model)


@dataclass
class CompactResult:
    """Result of a compaction operation."""

    success: bool
    summary: str = ""
    error: str = ""
    original_count: int = 0
    compacted_count: int = 0

    def get_messages(self) -> list[dict[str, Any]]:
        """
        Get the compacted message list.

        Returns:
            A list with the summary + recent messages.
        """
        if not self.success:
            return []

        summary_msg = {
            "role": "user",
            "content": (
                "[Previous conversation summary]\n\n"
                f"{self.summary}\n\n"
                "[End of summary. Continue from here.]"
            ),
        }

        # We need to store the recent messages somewhere
        # For now, this is a signal to the caller
        return [summary_msg]


# ── Compaction Prompt ───────────────────────────────────────────────

COMPACTION_SYSTEM_PROMPT = """\
You are a conversation summarizer. Your job is to create a concise but complete
summary of a conversation history.

## What to preserve:
- Key decisions and their rationale
- File paths that were created, modified, or read
- Important code changes (describe what was done, not the code itself)
- Unresolved issues or open questions
- Current task state and remaining work
- Any constraints or requirements mentioned

## What to skip:
- Verbose tool outputs (command results, file contents)
- Exploration that didn't lead anywhere useful
- Repeated attempts at the same thing
- Back-and-forth debugging noise

## Format:
Use markdown with clear sections. Be concise but complete.
Include specific file paths and key details.
"""


def build_compaction_prompt(
    messages: list[dict[str, Any]],
    instructions: str = "",
) -> str:
    """
    Build the prompt for the summarization LLM call.

    Args:
        messages: The old messages to summarize.
        instructions: Optional custom instructions for what to focus on.

    Returns:
        The prompt string to send to the LLM.
    """
    parts = ["Please summarize the following conversation history.\n"]

    if instructions:
        parts.append(f"Special instructions: {instructions}\n")

    parts.append("\n--- Conversation History ---\n")

    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content") or ""

        # For tool calls, include the function info
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                content += f"\n[Tool call: {fn.get('name', '?')}({fn.get('arguments', '')})]"

        if content:
            # Truncate very long content in the prompt
            if len(content) > 500:
                content = content[:500] + "... [truncated]"
            parts.append(f"[{role}]: {content}\n")

    parts.append("\n--- End of History ---\n")
    parts.append("Provide a concise summary in markdown.")

    return "".join(parts)


class Compactor:
    """
    Handles context compaction for a conversation.

    Usage:
        config = CompactionConfig(keep_recent_messages=6)
        compactor = Compactor(config)
        result = compactor.compact(messages)

        if result.success:
            # Use result.get_messages() for the new context
    """

    def __init__(
        self,
        config: CompactionConfig | None = None,
        client: OpenAI | None = None,
    ):
        self.config = config or CompactionConfig()
        self.client = client

    def compact(
        self,
        messages: list[dict[str, Any]],
        instructions: str = "",
    ) -> CompactResult:
        """
        Compact a conversation by summarizing old messages.

        Args:
            messages: The full conversation history.
            instructions: Optional custom instructions for summarization focus.

        Returns:
            A CompactResult with the summary and compacted message count.
        """
        if not self.config.enabled:
            return CompactResult(
                success=False,
                error="Compaction is disabled",
                original_count=len(messages),
            )

        if len(messages) <= self.config.keep_recent_messages:
            return CompactResult(
                success=False,
                error="Conversation too short to compact",
                original_count=len(messages),
            )

        # Split into old and recent
        old_messages, recent_messages = self._split_messages(messages)

        if not old_messages:
            return CompactResult(
                success=False,
                error="Nothing to compact",
                original_count=len(messages),
            )

        # Build prompt and get summary
        prompt = build_compaction_prompt(old_messages, instructions)

        try:
            summary = self._call_llm_for_summary(prompt, instructions)
        except Exception as e:
            return CompactResult(
                success=False,
                error=f"LLM summarization failed: {e}",
                original_count=len(messages),
            )

        if not summary:
            return CompactResult(
                success=False,
                error="LLM returned empty summary",
                original_count=len(messages),
            )

        result = CompactResult(
            success=True,
            summary=summary,
            original_count=len(messages),
            compacted_count=1 + len(recent_messages),  # summary + recent
        )

        # Store the recent messages on the result for retrieval
        result._recent_messages = recent_messages

        return result

    def _split_messages(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """
        Split messages into old and recent sections.

        Ensures the split point doesn't break tool call pairs:
        - If the split would land on a tool result, move it to recent
        - The recent section always starts with user or assistant (not tool)
        """
        keep = self.config.keep_recent_messages
        split_idx = len(messages) - keep

        if split_idx <= 0:
            return [], messages

        # Adjust split point to avoid breaking tool call sequences
        # A tool result (role=tool) must follow its assistant tool_call
        # Walk backward from split_idx to find a clean boundary
        while split_idx < len(messages):
            msg = messages[split_idx]
            # Good start if it's a user or assistant message
            if msg.get("role") in ("user", "assistant") and not msg.get("tool_calls"):
                break
            # If it's an assistant with tool_calls, include the whole sequence
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                split_idx += 1
                continue
            # If it's a tool result, move forward
            if msg.get("role") == "tool":
                split_idx += 1
                continue
            break

        old = messages[:split_idx]
        recent = messages[split_idx:]

        return old, recent

    def _call_llm_for_summary(
        self,
        prompt: str,
        instructions: str = "",
    ) -> str:
        """
        Call the LLM to generate a summary.

        Args:
            prompt: The compaction prompt with conversation history.
            instructions: Optional custom instructions.

        Returns:
            The summary text.
        """
        if not self.client:
            raise RuntimeError("No OpenAI client configured for compaction")

        model = self.config.model or "gpt-4o-mini"

        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": COMPACTION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )

        return response.choices[0].message.content or ""
