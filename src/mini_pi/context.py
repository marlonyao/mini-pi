"""
Context management for mini-pi.

Handles pruning (trimming old tool results) and compaction (summarizing old conversation).
Inspired by Pi's layered context management strategy.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field


@dataclass
class PruningConfig:
    """Configuration for session pruning (trimming old tool results)."""

    enabled: bool = True
    keep_recent_turns: int = 3      # Keep tool results from the last N turns intact
    soft_trim_chars: int = 500      # Kept for config compatibility; recent tool results remain intact
    max_tool_result_chars: int = 2000  # Hard limit for any single tool result


def prune_messages(
    messages: list[dict],
    config: PruningConfig | None = None,
) -> list[dict]:
    """
    Prune old tool results from messages to reduce context size.

    This is a pure function that returns a new list — it does NOT modify
    the original messages or the on-disk session.

    Strategy:
    1. Find all tool result messages and group them into "turns"
    2. Tool results from the last N turns are kept intact
    3. Tool results older than N turns are replaced with a placeholder
    4. User and assistant messages are never modified

    Args:
        messages: The conversation messages in OpenAI format.
        config: Pruning configuration. Uses defaults if None.

    Returns:
        A new list of messages with old tool results pruned.
    """
    if config is None:
        config = PruningConfig()

    if not config.enabled:
        return list(messages)

    if not messages:
        return []

    # Find indices of all tool result messages
    tool_result_indices = [
        i for i, m in enumerate(messages) if m.get("role") == "tool"
    ]

    if not tool_result_indices:
        return list(messages)

    # Group tool results into turns.
    # A turn boundary is when we see a user message between two tool results.
    # Simple approach: count the number of tool calls/results; each = 1 turn.
    # We treat each tool result as its own "turn" for simplicity.
    total_tool_results = len(tool_result_indices)
    keep_count = config.keep_recent_turns

    # Determine which tool results to keep vs clear
    # Last keep_count tool results are "recent", rest are "old"
    recent_indices = set(tool_result_indices[-keep_count:]) if keep_count > 0 else set()
    old_indices = set(tool_result_indices[:-keep_count]) if keep_count < total_tool_results else set()

    # Build result list
    result = []
    for i, msg in enumerate(messages):
        if msg.get("role") == "tool":
            if i in old_indices:
                # Hard clear: replace with placeholder
                new_msg = copy.copy(msg)
                new_msg["content"] = "[tool output removed - older than recent turns]"
                result.append(new_msg)
            elif i in recent_indices:
                # Keep recent results intact so the next model turn can use the full tool output.
                result.append(copy.copy(msg))
            else:
                result.append(copy.copy(msg))
        else:
            # User/assistant messages pass through unchanged
            result.append(msg)

    return result
