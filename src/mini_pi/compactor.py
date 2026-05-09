"""
Compaction for mini-pi.

When the conversation approaches the model's context window limit, compaction
summarizes older messages into a condensed summary so the conversation can continue.

Key principles (learned from Pi + fixing previous bugs):
- Recent tool outputs are NEVER truncated or summarized
- Split point respects turn boundaries (never breaks tool call + result pairs)
- Supports incremental summary updates (merge new messages into existing summary)
- Full history is preserved on disk (compaction is non-destructive)
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from openai import OpenAI


@dataclass
class CompactionConfig:
    """Configuration for context compaction."""

    enabled: bool = True
    keep_recent_messages: int = 8       # Number of recent messages to keep intact
    threshold: float = 0.8              # Context usage ratio that triggers compaction
    max_context_tokens: int = 128000    # Model context window size
    model: str | None = None            # Model for summarization (None = use session model)


@dataclass
class CompactResult:
    """Result of a compaction operation."""

    success: bool
    summary: str = ""
    error: str = ""
    original_count: int = 0
    compacted_count: int = 0
    _recent_messages: list[dict[str, Any]] | None = None

    def get_messages(self) -> list[dict[str, Any]]:
        """
        Get the compacted message list: summary + recent messages.
        """
        if not self.success:
            return []

        summary_msg = {
            "role": "system",
            "content": (
                "[Previous conversation summary]\n\n"
                f"{self.summary}\n\n"
                "[End of summary. Continue from here.]"
            ),
        }

        recent = self._recent_messages or []
        return [summary_msg] + list(recent)


# ── Compaction Prompt ───────────────────────────────────────────────

COMPACTION_SYSTEM_PROMPT = """\
You are a conversation summarizer. Create a concise but complete summary.

## What to preserve:
- Key decisions and their rationale
- File paths that were created, modified, or read
- Important code changes (describe what was done, not the full code)
- Unresolved issues or open questions
- Current task state and remaining work
- Any constraints or requirements mentioned

## What to skip:
- Verbose tool outputs (file contents, command results)
- Exploration that didn't lead anywhere
- Repeated attempts at the same thing

## Format:
Use markdown with clear sections. Be concise but complete.
Include specific file paths and key details.
"""

INCREMENTAL_SYSTEM_PROMPT = """\
You are updating an existing conversation summary with new information.

## Existing summary:
{existing_summary}

## New conversation to merge:
{new_messages}

## Instructions:
Update the summary to incorporate the new information. Keep the same format.
Remove information that is no longer relevant. Add new decisions, files, and context.
"""


def _format_messages_for_summary(messages: list[dict[str, Any]], max_chars: int = 2000) -> str:
    """Format messages for inclusion in the summarization prompt.

    Truncates individual tool outputs to max_chars, but keeps all messages
    so the LLM sees the full conversation flow.
    """
    parts = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content") or ""

        # For tool calls, include the function info
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                fn = tc.get("function", {})
                content += f"\n[Tool call: {fn.get('name', '?')}({fn.get('arguments', '')})]"

        if content:
            # Truncate individual tool outputs that are too long
            if role == "tool" and len(content) > max_chars:
                content = content[:max_chars] + f"\n... [truncated, {len(content)} chars total]"
            parts.append(f"[{role}]: {content}\n")

    return "".join(parts)


def build_compaction_prompt(
    messages: list[dict[str, Any]],
    instructions: str = "",
) -> str:
    """Build the prompt for the summarization LLM call (full rewrite)."""
    parts = ["Please summarize the following conversation history.\n"]

    if instructions:
        parts.append(f"Special instructions: {instructions}\n")

    parts.append("\n--- Conversation History ---\n")
    parts.append(_format_messages_for_summary(messages))
    parts.append("\n--- End of History ---\n")
    parts.append("Provide a concise summary in markdown.")

    return "".join(parts)


def build_incremental_prompt(
    existing_summary: str,
    new_messages: list[dict[str, Any]],
    instructions: str = "",
) -> str:
    """Build the prompt for incremental summary update."""
    parts = ["Please update the existing summary with the new conversation below.\n"]

    if instructions:
        parts.append(f"Special instructions: {instructions}\n")

    parts.append("\n--- Existing Summary ---\n")
    parts.append(existing_summary)
    parts.append("\n\n--- New Conversation ---\n")
    parts.append(_format_messages_for_summary(new_messages))
    parts.append("\n--- End ---\n")
    parts.append("Provide the updated summary in markdown.")

    return "".join(parts)


class Compactor:
    """Handles context compaction for a conversation."""

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
        existing_summary: str = "",
    ) -> CompactResult:
        """
        Compact a conversation by summarizing old messages.

        Args:
            messages: The full conversation history.
            instructions: Optional custom instructions for summarization focus.
            existing_summary: If provided, do incremental update instead of full rewrite.

        Returns:
            A CompactResult with the summary and recent messages.
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

        # Split into old and recent (respects turn boundaries)
        old_messages, recent_messages = self._split_messages(messages)

        if not old_messages:
            return CompactResult(
                success=False,
                error="Nothing to compact",
                original_count=len(messages),
            )

        # Build prompt: incremental if we have an existing summary, else full
        try:
            if existing_summary:
                prompt = build_incremental_prompt(
                    existing_summary, old_messages, instructions
                )
            else:
                prompt = build_compaction_prompt(old_messages, instructions)

            summary = self._call_llm_for_summary(prompt)
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

        return CompactResult(
            success=True,
            summary=summary,
            original_count=len(messages),
            compacted_count=1 + len(recent_messages),
            _recent_messages=recent_messages,
        )

    def _split_messages(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """
        Split messages into old (to summarize) and recent (to keep intact).

        CRITICAL: The split point must NOT break tool call sequences.
        An assistant message with tool_calls MUST be followed by all its
        tool results in the same section (old or recent).

        Algorithm:
        1. Start with split_idx = len(messages) - keep_recent_messages
        2. Walk backward to find a clean boundary:
           - Prefer splitting at a user message boundary
           - Never split in the middle of a tool call sequence
        """
        keep = self.config.keep_recent_messages
        split_idx = max(0, len(messages) - keep)

        if split_idx == 0:
            return [], list(messages)

        # Walk backward from split_idx to find a clean cut point.
        # A clean cut is at a user message (start of a new turn).
        # We must ensure any tool_call + tool_result sequence is not split.
        best_idx = 0  # Fallback: compact everything

        for idx in range(split_idx, 0, -1):
            if self._is_clean_boundary(messages, idx):
                best_idx = idx
                break

        # If no clean boundary found before split_idx, use split_idx
        # but verify it doesn't break tool pairs
        if best_idx == 0:
            best_idx = self._adjust_for_tool_pairs(messages, split_idx)

        old = messages[:best_idx]
        recent = messages[best_idx:]

        return old, recent

    def _is_clean_boundary(self, messages: list[dict], idx: int) -> bool:
        """Check if idx is a clean cut point.

        A clean boundary is:
        - The message at idx is a user message (start of turn)
        - The message at idx-1 is NOT a tool result (end of previous turn)
        """
        if idx <= 0 or idx >= len(messages):
            return False

        # Must start with user message
        if messages[idx].get("role") != "user":
            return False

        # Previous message should not be a tool result (would orphan the tool call)
        prev = messages[idx - 1]
        if prev.get("role") == "tool":
            return False

        return True

    def _adjust_for_tool_pairs(self, messages: list[dict], idx: int) -> int:
        """
        Adjust split point forward if it would break a tool call + result pair.

        If messages[idx] is:
        - A tool result: walk forward past all consecutive tool results,
          then past the assistant that holds the tool_calls
        - An assistant with tool_calls: walk forward past all its tool results
        """
        while idx < len(messages):
            msg = messages[idx]

            if msg.get("role") == "tool":
                # Walk forward past all consecutive tool results
                while idx < len(messages) and messages[idx].get("role") == "tool":
                    idx += 1
                # Also skip the assistant that has the tool_calls
                # (it should be just before the tool results, but we're
                #  walking forward, so it's already included)
                continue

            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                # This assistant has tool calls — include it and all its results
                idx += 1
                # Walk forward past all tool results that follow
                while idx < len(messages) and messages[idx].get("role") == "tool":
                    idx += 1
                continue

            # Clean boundary: user or plain assistant
            break

        return idx

    def _call_llm_for_summary(self, prompt: str) -> str:
        """Call the LLM to generate a summary."""
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
