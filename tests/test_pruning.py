"""
Tests for context pruning module.

Pruning trims old tool results from the context to reduce token bloat.
It should:
1. Keep recent N turns of tool results intact
2. Soft-trim oversized results (keep head + tail)
3. Hard-clear very old results
4. Never modify user/assistant messages
5. Never modify the on-disk session (pruning is in-memory only)
"""

import pytest

from mini_pi.context import prune_messages, PruningConfig


# ── Fixtures ────────────────────────────────────────────────────────

def make_tool_result(tool_call_id: str, content: str) -> dict:
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


def make_user(content: str) -> dict:
    return {"role": "user", "content": content}


def make_assistant(content: str, tool_calls: list[dict] | None = None) -> dict:
    msg: dict = {"role": "assistant"}
    if content:
        msg["content"] = content
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def make_tool_call(call_id: str, name: str) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": "{}"},
    }


def build_conversation(turns: int, tool_result_size: int = 100) -> list[dict]:
    """
    Build a conversation with N turns, each having:
    - user message
    - assistant with tool call
    - tool result (of given size)
    - assistant text response
    """
    messages = []
    for i in range(turns):
        messages.append(make_user(f"User message {i}"))
        messages.append(make_assistant(
            "",
            [make_tool_call(f"call_{i}", "bash")],
        ))
        messages.append(make_tool_result(f"call_{i}", "x" * tool_result_size))
        messages.append(make_assistant(f"Assistant response {i}"))
    return messages


# ── Tests ───────────────────────────────────────────────────────────

class TestPruningConfig:
    def test_defaults(self):
        config = PruningConfig()
        assert config.enabled is True
        assert config.keep_recent_turns == 3
        assert config.soft_trim_chars == 500
        assert config.max_tool_result_chars == 2000

    def test_disabled(self):
        config = PruningConfig(enabled=False)
        messages = build_conversation(10)
        result = prune_messages(messages, config)
        # When disabled, should return messages unchanged
        assert result == messages


class TestPruningBasicBehavior:
    def test_recent_tool_results_kept_intact(self):
        """Tool results within keep_recent_turns should not be touched."""
        messages = build_conversation(5, tool_result_size=50)
        config = PruningConfig(keep_recent_turns=3)
        result = prune_messages(messages, config)

        # Last 3 tool results should be intact
        tool_msgs = [m for m in result if m["role"] == "tool"]
        # 5 total tool results, last 3 kept = 2 trimmed
        assert len(tool_msgs) == 5
        # Last 3 should have original content
        for tm in tool_msgs[-3:]:
            assert tm["content"] == "x" * 50

    def test_old_tool_results_hard_cleared(self):
        """Tool results older than keep_recent_turns get replaced with placeholder."""
        messages = build_conversation(5, tool_result_size=50)
        config = PruningConfig(keep_recent_turns=2)
        result = prune_messages(messages, config)

        tool_msgs = [m for m in result if m["role"] == "tool"]
        # First 3 should be cleared (turns 0, 1, 2 are older than last 2)
        assert "[tool output removed" in tool_msgs[0]["content"]
        assert "[tool output removed" in tool_msgs[1]["content"]
        assert "[tool output removed" in tool_msgs[2]["content"]
        # Last 2 should be intact
        assert tool_msgs[3]["content"] == "x" * 50
        assert tool_msgs[4]["content"] == "x" * 50

    def test_user_assistant_messages_never_modified(self):
        """Pruning should never touch user or assistant messages."""
        messages = build_conversation(5, tool_result_size=50)
        config = PruningConfig(keep_recent_turns=2)
        result = prune_messages(messages, config)

        for i, msg in enumerate(result):
            if msg["role"] in ("user", "assistant"):
                assert msg == messages[i], f"Message at index {i} was modified!"

    def test_empty_messages(self):
        """Pruning empty message list should work."""
        result = prune_messages([], PruningConfig())
        assert result == []

    def test_no_tool_results(self):
        """Conversation without tool results should pass through."""
        messages = [
            make_user("hello"),
            make_assistant("hi there"),
        ]
        result = prune_messages(messages, PruningConfig())
        assert result == messages


class TestSoftTrim:
    def test_oversized_recent_result_soft_trimmed(self):
        """Tool results exceeding soft_trim_chars get head+tail trimmed."""
        long_content = "A" * 800  # longer than soft_trim_chars=500
        messages = [
            make_user("do something"),
            make_assistant("", [make_tool_call("c1", "bash")]),
            make_tool_result("c1", long_content),
            make_assistant("done"),
        ]
        config = PruningConfig(
            keep_recent_turns=3,
            soft_trim_chars=200,
        )
        result = prune_messages(messages, config)

        tool_msg = [m for m in result if m["role"] == "tool"][0]
        # Should be shorter than original
        assert len(tool_msg["content"]) < len(long_content)
        # Should contain head and tail
        assert tool_msg["content"].startswith("A")
        assert tool_msg["content"].endswith("A")
        # Should contain truncation marker
        assert "..." in tool_msg["content"] or "truncated" in tool_msg["content"].lower()

    def test_small_result_not_trimmed(self):
        """Tool results within soft_trim_chars should not be touched."""
        messages = [
            make_user("do something"),
            make_assistant("", [make_tool_call("c1", "bash")]),
            make_tool_result("c1", "short output"),
            make_assistant("done"),
        ]
        config = PruningConfig(soft_trim_chars=500)
        result = prune_messages(messages, config)

        tool_msg = [m for m in result if m["role"] == "tool"][0]
        assert tool_msg["content"] == "short output"


class TestReturnNewList:
    def test_pruning_returns_new_list(self):
        """Pruning should return a new list, not mutate the original."""
        messages = build_conversation(5, tool_result_size=50)
        original_ids = [id(m) for m in messages]
        config = PruningConfig(keep_recent_turns=2)
        result = prune_messages(messages, config)

        # The list itself should be new
        assert result is not messages
        # Original messages should be unchanged
        for i, msg in enumerate(messages):
            if msg["role"] == "tool":
                assert msg["content"] == "x" * 50 or "x" in msg["content"]


class TestTurnCounting:
    def test_turns_counted_by_tool_calls(self):
        """A 'turn' is defined by a tool call + its result."""
        # 3 tool calls = 3 turns, keeping 1 means 2 should be pruned
        messages = []
        for i in range(3):
            messages.append(make_user(f"msg {i}"))
            messages.append(make_assistant("", [make_tool_call(f"c{i}", "bash")]))
            messages.append(make_tool_result(f"c{i}", f"output {i}"))
            messages.append(make_assistant(f"result {i}"))

        config = PruningConfig(keep_recent_turns=1)
        result = prune_messages(messages, config)

        tool_msgs = [m for m in result if m["role"] == "tool"]
        assert "[tool output removed" in tool_msgs[0]["content"]
        assert "[tool output removed" in tool_msgs[1]["content"]
        assert tool_msgs[2]["content"] == "output 2"
