"""
Compaction for mini-pi (Pi-aligned strategy).

When the conversation approaches the model's context window limit, compaction
summarizes older messages into a condensed summary so the conversation can
continue.

Key principles (aligned with Pi coding agent):
- Trigger: contextTokens > contextWindow - reserveTokens (absolute, not ratio)
- Keep strategy: walk backwards accumulating tokens until keepRecentTokens budget
- Split point respects turn boundaries (never breaks tool call + result pairs)
- Split turn: when a single turn exceeds budget, cut mid-turn at assistant boundary
- Structured summary format (Goal/Progress/Decisions/NextSteps + file tracking)
- Cumulative file tracking across compactions
- Full history is preserved on disk (compaction is non-destructive)
- No "summary of summary" — always summarize from raw messages
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from .token_estimator import estimate_tokens, _extract_message_texts

logger = logging.getLogger(__name__)


# ── Configuration ───────────────────────────────────────────────────


@dataclass
class CompactionConfig:
    """Configuration for context compaction (Pi-aligned).

    Attributes:
        enabled: Whether auto-compaction is enabled.
        reserve_tokens: Tokens reserved for LLM response. Compaction triggers
            when contextTokens > max_context_tokens - reserve_tokens.
        keep_recent_tokens: Token budget for recent messages. The compactor
            walks backwards from the newest message, accumulating tokens until
            this budget is reached. All messages within the budget are kept.
        max_context_tokens: Model's total context window size.
        model: Model for summarization. None = use session model.
    """

    enabled: bool = True
    reserve_tokens: int = 16384
    keep_recent_tokens: int = 20000
    max_context_tokens: int = 128000
    model: str | None = None


@dataclass
class CompactResult:
    """Result of a compaction operation."""

    success: bool
    summary: str = ""
    error: str = ""
    original_count: int = 0
    compacted_count: int = 0
    tokens_before: int = 0
    first_kept_index: int = 0
    read_files: list[str] = field(default_factory=list)
    modified_files: list[str] = field(default_factory=list)
    is_split_turn: bool = False
    _recent_messages: list[dict[str, Any]] | None = None
    _summary_messages: list[dict[str, Any]] | None = None


# ── Summary Prompts ─────────────────────────────────────────────────

COMPACTION_SYSTEM_PROMPT = """\
You are a conversation summarizer for a coding agent. Create a structured \
summary that preserves all information needed to continue the task.

You MUST use this exact markdown format:

## Goal
[What the user is trying to accomplish — be specific]

## Constraints & Preferences
- [Requirements or preferences mentioned by the user]
- [Technical constraints that must be respected]

## Progress
### Done
- [x] [Completed tasks with specific details]

### In Progress
- [ ] [Current work that has started but not finished]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Rationale for the decision]

## Next Steps
1. [What should happen next, in order]

## Critical Context
- [Specific data needed to continue: variable names, paths, configurations]
- [Any partial results or intermediate values]

<read-files>
[file paths that were read, one per line]
</read-files>

<modified-files>
[file paths that were modified, one per line]
</modified-files>

## Rules:
- Include specific file paths, variable names, and values
- Describe code changes at a high level (what was done, not full code)
- Never invent information not present in the conversation
- If a section has no content, omit it entirely
- The <read-files> and <modified-files> tags are REQUIRED even if empty
"""

SPLIT_TURN_SYSTEM_PROMPT = """\
You are summarizing the early portion of an in-progress task. The agent was \
in the middle of a multi-step operation when context limits were reached.

Summarize what has been discovered and done so far, but do NOT assume the \
task is complete. Use the same structured format as the system prompt.

Pay special attention to:
- What the user asked for (the Goal)
- What files were examined and what was found
- Any partial changes made
- What remains to be done (Next Steps)
"""


# ── Message Serialization ──────────────────────────────────────────


def serialize_message(msg: dict[str, Any], max_chars: int = 2000) -> str:
    """Serialize a single message to text for summarization.

    Truncates tool outputs to max_chars. Returns text in a format
    that prevents the summarizer from treating it as a conversation
    to continue.
    """
    role = msg.get("role", "unknown")
    content = msg.get("content") or ""

    tool_calls = msg.get("tool_calls")

    # Build the serialized output
    parts: list[str] = []

    # Truncate long tool outputs before serializing
    display_content = content
    if role == "tool" and len(content) > max_chars:
        display_content = content[:max_chars] + f"\n... [truncated, {len(content)} chars total]"

    # Always include content if present (even when tool_calls exist)
    if display_content:
        role_label = {
            "user": "User",
            "assistant": "Assistant",
            "tool": "Tool result",
            "system": "System",
        }.get(role, role.capitalize())
        parts.append(f"[{role_label}]: {display_content}\n")

    # Append tool call info if present
    if tool_calls:
        tc_parts = []
        for tc in tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", "?")
            args_str = fn.get("arguments", "")
            tc_parts.append(f"{name}({args_str})")
        parts.append(f"[Assistant tool calls]: {'; '.join(tc_parts)}\n")

    if not parts:
        return ""

    return "".join(parts)


def serialize_conversation(
    messages: list[dict[str, Any]],
    max_chars: int = 2000,
) -> str:
    """Serialize a list of messages to text for summarization.

    Produces a flat text representation that the summarizer LLM
    can read without treating it as a conversation to continue.
    """
    parts: list[str] = []
    for msg in messages:
        text = serialize_message(msg, max_chars=max_chars)
        if text:
            parts.append(text)
    return "".join(parts)


# ── File Tracking ───────────────────────────────────────────────────


def extract_file_ops(
    messages: list[dict[str, Any]],
    prev_read: list[str] | None = None,
    prev_modified: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Extract file operations from messages, merging with previous lists.

    Scans assistant messages for tool_calls and extracts file paths
    from read, write, edit, grep, and find tools. Merges cumulatively
    with previous compaction file lists.

    Returns:
        (sorted_read_files, sorted_modified_files)
    """
    read_files = set(prev_read or [])
    modified_files = set(prev_modified or [])

    for msg in messages:
        tool_calls = msg.get("tool_calls")
        if not tool_calls:
            continue

        for tc in tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            args_str = fn.get("arguments", "{}")
            try:
                args = json.loads(args_str)
            except (json.JSONDecodeError, TypeError):
                logger.warning(
                    "Failed to parse tool call arguments for file tracking: "
                    "name=%s, args_prefix=%r",
                    name, args_str[:80],
                )
                continue

            path = args.get("path", "")

            if name == "read":
                if path:
                    read_files.add(path)
            elif name in ("write", "edit"):
                if path:
                    modified_files.add(path)
            elif name == "grep":
                grep_path = args.get("path", "")
                if grep_path:
                    read_files.add(grep_path)
            elif name == "find":
                find_path = args.get("path", "")
                if find_path:
                    read_files.add(find_path)
            elif name == "bash":
                # Bash commands may read/modify files — skip heuristic extraction
                # as it's unreliable. The LLM summary will capture key paths.
                pass

    return sorted(read_files), sorted(modified_files)


def append_file_tags(
    summary: str,
    read_files: list[str],
    modified_files: list[str],
) -> str:
    """Ensure the summary has <read-files> and <modified-files> tags.

    If the summary already contains them, leave as-is.
    Otherwise, append them at the end.
    """
    if "<read-files>" in summary and "<modified-files>" in summary:
        return summary

    parts = [summary.rstrip()]

    if "<read-files>" not in summary:
        parts.append("\n\n<read-files>")
        for f in read_files:
            parts.append(f"\n{f}")
        parts.append("\n</read-files>")

    if "<modified-files>" not in summary:
        parts.append("\n\n<modified-files>")
        for f in modified_files:
            parts.append(f"\n{f}")
        parts.append("\n</modified-files>")

    return "".join(parts)


# ── Token Estimation Helpers ────────────────────────────────────────


def estimate_message_tokens(msg: dict[str, Any]) -> int:
    """Estimate token count for a single message.

    Uses the same logic as TokenEstimator.estimate_messages() to ensure
    consistency. Includes ~4 tokens overhead per message.
    """
    total = 4  # message overhead
    for text in _extract_message_texts(msg):
        total += estimate_tokens(text)
    return total


def estimate_messages_tokens(messages: list[dict[str, Any]]) -> int:
    """Estimate total token count for a list of messages."""
    return sum(estimate_message_tokens(msg) for msg in messages)


# ── Compaction Prompt Builders ──────────────────────────────────────


def build_compaction_prompt(
    messages_to_summarize: list[dict[str, Any]],
    turn_prefix_messages: list[dict[str, Any]] | None = None,
    instructions: str = "",
    is_split_turn: bool = False,
) -> str:
    """Build the user prompt for the summarization LLM call.

    For normal compaction, only messages_to_summarize is provided.
    For split turns, turn_prefix_messages contains the beginning
    of the oversized turn.
    """
    parts: list[str] = []

    if instructions:
        parts.append(f"Special instructions: {instructions}\n\n")

    if messages_to_summarize:
        parts.append("--- Complete Conversation Turns to Summarize ---\n")
        parts.append(serialize_conversation(messages_to_summarize))
        parts.append("\n")

    if turn_prefix_messages:
        parts.append("--- In-Progress Turn (Early Portion) ---\n")
        parts.append(
            "The following is the beginning of a task that is still in progress.\n"
            "Summarize what was discovered and done, but do NOT mark it as complete.\n\n"
        )
        parts.append(serialize_conversation(turn_prefix_messages))
        parts.append("\n")

    parts.append("Provide a structured summary using the exact format specified.")

    return "".join(parts)


# ── Compactor ───────────────────────────────────────────────────────


class Compactor:
    """Handles context compaction for a conversation (Pi-aligned)."""

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
        prev_read_files: list[str] | None = None,
        prev_modified_files: list[str] | None = None,
    ) -> CompactResult:
        """Compact a conversation by summarizing old messages.

        Algorithm:
        1. Find split index by walking backwards with token budget
        2. Adjust to turn boundary (never break tool pairs)
        3. Detect split turn (single turn exceeds budget)
        4. Extract file operations cumulatively
        5. Serialize and summarize via LLM

        Args:
            messages: The full conversation history (all messages).
            instructions: Optional custom instructions for summarization.
            prev_read_files: Files tracked from previous compaction.
            prev_modified_files: Modified files from previous compaction.

        Returns:
            A CompactResult with the summary and metadata.
        """
        if not self.config.enabled:
            return CompactResult(
                success=False,
                error="Compaction is disabled",
                original_count=len(messages),
            )

        if not messages:
            return CompactResult(
                success=False,
                error="No messages to compact",
                original_count=0,
            )

        tokens_before = estimate_messages_tokens(messages)

        # Step 1: Find split index by token budget (walk backwards)
        raw_split_idx = self._find_split_by_tokens(messages)

        if raw_split_idx <= 0:
            # Entire conversation fits within keep_recent_tokens — nothing to compact
            return CompactResult(
                success=False,
                error="Conversation too short to compact",
                original_count=len(messages),
                tokens_before=tokens_before,
            )

        # Step 2: Adjust to turn boundary
        split_idx = self._adjust_to_turn_boundary(messages, raw_split_idx)

        if split_idx <= 0:
            return CompactResult(
                success=False,
                error="No clean split point found",
                original_count=len(messages),
                tokens_before=tokens_before,
            )

        # Step 3: Detect split turn
        turn_start = self._find_turn_start(messages, split_idx)
        is_split = turn_start < split_idx

        # Partition messages
        if is_split:
            messages_to_summarize = messages[:turn_start]
            turn_prefix = messages[turn_start:split_idx]
        else:
            messages_to_summarize = messages[:split_idx]
            turn_prefix = None

        recent_messages = messages[split_idx:]

        if not messages_to_summarize and not turn_prefix:
            return CompactResult(
                success=False,
                error="Nothing to summarize",
                original_count=len(messages),
                tokens_before=tokens_before,
            )

        # Step 4: Extract file operations (cumulative)
        all_summarized = messages[:split_idx]
        read_files, modified_files = extract_file_ops(
            all_summarized,
            prev_read=prev_read_files,
            prev_modified=prev_modified_files,
        )

        # Step 5: Build prompt and call LLM
        try:
            prompt = build_compaction_prompt(
                messages_to_summarize=messages_to_summarize,
                turn_prefix_messages=turn_prefix,
                instructions=instructions,
                is_split_turn=is_split,
            )
            system_prompt = SPLIT_TURN_SYSTEM_PROMPT if is_split else COMPACTION_SYSTEM_PROMPT
            summary = self._call_llm_for_summary(prompt, system_prompt)
        except Exception as e:
            return CompactResult(
                success=False,
                error=f"LLM summarization failed: {e}",
                original_count=len(messages),
                tokens_before=tokens_before,
            )

        if not summary:
            return CompactResult(
                success=False,
                error="LLM returned empty summary",
                original_count=len(messages),
                tokens_before=tokens_before,
            )

        # Ensure file tags are present
        summary = append_file_tags(summary, read_files, modified_files)

        return CompactResult(
            success=True,
            summary=summary,
            original_count=len(messages),
            compacted_count=1 + len(recent_messages),
            tokens_before=tokens_before,
            first_kept_index=split_idx,
            read_files=read_files,
            modified_files=modified_files,
            is_split_turn=is_split,
            _recent_messages=recent_messages,
            _summary_messages=all_summarized,
        )

    def _find_split_by_tokens(
        self,
        messages: list[dict[str, Any]],
    ) -> int:
        """Walk backwards from newest message, accumulating tokens.

        Returns the index where the budget was exceeded (the first
        message that would go OVER budget). All messages from this
        index onward are "kept".
        """
        budget = self.config.keep_recent_tokens
        accumulated = 0
        split_idx = len(messages)

        for i in range(len(messages) - 1, -1, -1):
            msg_tokens = estimate_message_tokens(messages[i])
            if accumulated + msg_tokens > budget:
                # This message would exceed budget — it's the boundary
                split_idx = i + 1
                break
            accumulated += msg_tokens
            split_idx = i

        return split_idx

    def _adjust_to_turn_boundary(
        self,
        messages: list[dict[str, Any]],
        split_idx: int,
    ) -> int:
        """Adjust split point to a clean turn boundary.

        Walk backward from split_idx to find the nearest user message
        (start of a turn). Ensure we never split a tool_call + tool_result
        pair.

        Valid cut points:
        - user messages (preferred — clean turn start)
        - assistant messages without pending tool results

        Invalid cut points:
        - tool results (would orphan the tool call)
        - assistant messages whose tool results appear after the cut
        """
        if split_idx <= 0 or split_idx >= len(messages):
            return split_idx

        # First, ensure we're not cutting inside a tool pair
        split_idx = self._adjust_for_tool_pairs(messages, split_idx)

        if split_idx <= 0:
            return 0

        # Walk backward to find a user message (clean turn boundary)
        for idx in range(split_idx, 0, -1):
            if messages[idx].get("role") == "user":
                # Verify previous message isn't a tool result
                if idx > 0 and messages[idx - 1].get("role") != "tool":
                    return idx

        # No user message found — fall back to split_idx
        # (this can happen if conversation starts with assistant)
        return split_idx

    def _adjust_for_tool_pairs(
        self,
        messages: list[dict[str, Any]],
        idx: int,
    ) -> int:
        """Adjust split point forward if it would break a tool pair.

        If messages[idx] is:
        - A tool result: walk forward past all consecutive tool results
        - An assistant with tool_calls whose results follow: walk forward
        """
        while idx < len(messages):
            msg = messages[idx]

            if msg.get("role") == "tool":
                # Walk forward past all consecutive tool results
                while idx < len(messages) and messages[idx].get("role") == "tool":
                    idx += 1
                continue

            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                # Check if tool results follow
                next_idx = idx + 1
                if next_idx < len(messages) and messages[next_idx].get("role") == "tool":
                    # Include this assistant and all its tool results
                    idx += 1
                    while idx < len(messages) and messages[idx].get("role") == "tool":
                        idx += 1
                    continue

            # Clean boundary
            break

        return idx

    def _find_turn_start(
        self,
        messages: list[dict[str, Any]],
        split_idx: int,
    ) -> int:
        """Find the start of the turn containing split_idx.

        A turn starts at a user message. Walk backward from split_idx
        to find the most recent user message before it.
        """
        for i in range(split_idx - 1, -1, -1):
            if messages[i].get("role") == "user":
                return i
        return 0

    def _call_llm_for_summary(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        """Call the LLM to generate a summary."""
        if not self.client:
            raise RuntimeError("No OpenAI client configured for compaction")

        model = self.config.model or "gpt-4o-mini"
        sys_prompt = system_prompt or COMPACTION_SYSTEM_PROMPT

        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )

        return response.choices[0].message.content or ""
