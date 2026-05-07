"""
Session management for mini-pi.

Sessions are stored as JSONL files (one JSON object per line).
Each entry records a message with role, content, and optional tool call data.

Inspired by Pi's tree-structured sessions — this is a simplified linear version.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


class Session:
    """Manages conversation history with JSONL persistence."""

    def __init__(self, path: Path | None = None):
        self.path = path
        self.messages: list[dict[str, Any]] = []
        self.created_at = datetime.now().isoformat()
        self.token_usage = {"prompt": 0, "completion": 0, "total": 0}

        if path and path.exists():
            self._load()

    def _load(self) -> None:
        """Load session from JSONL file."""
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                entry = json.loads(line)
                if entry.get("type") == "message":
                    self.messages.append(entry["data"])
                elif entry.get("type") == "meta":
                    self.created_at = entry.get("created_at", self.created_at)

    def record_compaction(self, result: "CompactResult") -> None:
        """Record a compaction event in the session.

        This adds a meta entry to the JSONL so we know compaction happened,
        and replaces older messages with the summary + recent tail.
        """
        if not result.success:
            return

        # Store the summary as a special message
        summary_msg = {
            "role": "system",
            "content": f"[Compaction Summary]\n{result.summary}",
        }

        # Get recent messages from the result
        recent = getattr(result, "_recent_messages", [])

        # Replace messages: summary + recent
        self.messages = [summary_msg] + list(recent)

        # Add compaction meta entry
        self._compaction_count = getattr(self, "_compaction_count", 0) + 1

    def save(self) -> None:
        """Persist session to JSONL file."""
        if not self.path:
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = []

        # Meta entry
        meta = {
            "type": "meta",
            "created_at": self.created_at,
            "saved_at": datetime.now().isoformat(),
        }
        if hasattr(self, "_compaction_count") and self._compaction_count:
            meta["compaction_count"] = self._compaction_count
        lines.append(json.dumps(meta))

        # Compaction entry (if any)
        if hasattr(self, "_compaction_count") and self._compaction_count:
            lines.append(json.dumps({
                "type": "compaction",
                "count": self._compaction_count,
                "saved_at": datetime.now().isoformat(),
            }))

        # Message entries
        for msg in self.messages:
            lines.append(json.dumps({"type": "message", "data": msg}))

        self.path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def add(self, role: str, content: str | None = None, **kwargs: Any) -> dict:
        """Add a message and return it."""
        msg: dict[str, Any] = {"role": role}
        if content is not None:
            msg["content"] = content
        msg.update(kwargs)
        self.messages.append(msg)
        return msg

    def add_user(self, content: str) -> dict:
        return self.add("user", content)

    def add_assistant(self, content: str | None = None, **kwargs: Any) -> dict:
        return self.add("assistant", content, **kwargs)

    def add_tool_result(self, tool_call_id: str, content: str) -> dict:
        return self.add("tool", content, tool_call_id=tool_call_id)

    def get_openai_messages(self) -> list[dict[str, Any]]:
        """Get messages in OpenAI API format."""
        result = []
        for msg in self.messages:
            entry: dict[str, Any] = {"role": msg["role"]}
            if "content" in msg:
                entry["content"] = msg["content"]
            if "tool_calls" in msg:
                entry["tool_calls"] = msg["tool_calls"]
            if "tool_call_id" in msg:
                entry["tool_call_id"] = msg["tool_call_id"]
            # DeepSeek thinking mode: reasoning_content must be passed back
            if "reasoning_content" in msg:
                entry["reasoning_content"] = msg["reasoning_content"]
            if msg["role"] == "tool":
                entry["role"] = "tool"
            result.append(entry)
        return result

    def update_token_usage(self, usage: dict) -> None:
        """Accumulate token usage stats."""
        self.token_usage["prompt"] += usage.get("prompt_tokens", 0)
        self.token_usage["completion"] += usage.get("completion_tokens", 0)
        self.token_usage["total"] += usage.get("total_tokens", 0)


def create_session(session_dir: str, name: str | None = None) -> Session:
    """Create a new session with an auto-generated name."""
    session_dir_path = Path(session_dir)
    session_dir_path.mkdir(parents=True, exist_ok=True)

    if not name:
        name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    path = session_dir_path / f"{name}.jsonl"
    return Session(path)


def load_session(path: str) -> Session:
    """Load an existing session from file."""
    return Session(Path(path))


def list_sessions(session_dir: str) -> list[dict]:
    """List all sessions in the session directory."""
    session_path = Path(session_dir)
    if not session_path.exists():
        return []

    sessions = []
    for f in sorted(session_path.glob("*.jsonl"), reverse=True):
        stat = f.stat()
        sessions.append({
            "name": f.stem,
            "path": str(f),
            "size": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        })
    return sessions
