"""
Context management for mini-pi.

Pruning: trims old tool results to reduce context size.
Does NOT touch recent tool results — only clears results from old turns.

Key principle: recent tool outputs must be kept INTACT so the LLM can use them.
Only old turn results get replaced with a short placeholder.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass


@dataclass
class PruningConfig:
    """Configuration for session pruning (trimming old tool results)."""

    enabled: bool = True
    keep_recent_turns: int = 3      # Keep tool results from the last N user turns
    max_tool_result_chars: int = 0  # 0 = no truncation for recent results


def _group_into_turns(messages: list[dict]) -> list[list[int]]:
    """Group tool result indices by user turn.

    A turn starts at each user message. Tool results between two user
    messages belong to the same turn.

    Returns list of groups, each group is a list of indices of tool results.
    """
    groups: list[list[int]] = []
    current_group: list[int] = []

    for i, msg in enumerate(messages):
        if msg.get("role") == "user":
            # New user turn starts — flush previous group
            if current_group:
                groups.append(current_group)
                current_group = []
        if msg.get("role") == "tool":
            current_group.append(i)

    # Flush last group
    if current_group:
        groups.append(current_group)

    return groups


def prune_messages(
    messages: list[dict],
    config: PruningConfig | None = None,
) -> list[dict]:
    """
    Prune old tool results from messages to reduce context size.

    Pure function — returns a new list, does NOT modify original.

    Strategy:
    1. Group tool results by user turn
    2. Tool results from the last N turns are kept INTACT (never truncated)
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

    # Group tool results by turn
    turn_groups = _group_into_turns(messages)

    if not turn_groups:
        return list(messages)

    # Determine which turns to keep
    keep_turns = config.keep_recent_turns
    recent_turn_count = min(keep_turns, len(turn_groups))

    # All indices in recent turns → keep intact
    recent_indices: set[int] = set()
    for group in turn_groups[-recent_turn_count:]:
        recent_indices.update(group)

    # All other tool result indices → replace with placeholder
    all_tool_indices: set[int] = set()
    for group in turn_groups:
        all_tool_indices.update(group)
    old_indices = all_tool_indices - recent_indices

    # Build result list
    result = []
    for i, msg in enumerate(messages):
        if i in old_indices and msg.get("role") == "tool":
            # Old tool result: replace content with placeholder
            new_msg = copy.copy(msg)
            new_msg["content"] = "[tool output removed]"
            result.append(new_msg)
        else:
            # Everything else passes through unchanged
            # Recent tool results are kept INTACT — no truncation
            result.append(copy.copy(msg) if i in recent_indices else msg)

    return result
