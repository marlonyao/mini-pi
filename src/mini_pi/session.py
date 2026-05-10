"""
Session management for mini-pi.

Sessions are stored as append-only JSONL files (one JSON object per line).
Each entry records a message with role, content, and optional tool call data.

The JSONL is append-only — new messages are always appended, never overwritten.
This preserves full conversation history even after compaction.

Entry types:
  - meta:       Session metadata (created_at, etc.)
  - message:    A single message (user/assistant/tool)
  - snapshot:   A compaction checkpoint (contains the compacted message list)

Loading reads from the last snapshot forward, so only the active messages
are in memory, while the full history remains on disk.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


class Session:
    """Manages conversation history with append-only JSONL persistence."""

    def __init__(self, path: Path | None = None):
        self.path = path
        self.messages: list[dict[str, Any]] = []
        self.created_at = datetime.now().isoformat()
        self.token_usage = {"prompt": 0, "completion": 0, "total": 0}

        # Track how many messages have been persisted to disk.
        # Only messages beyond this index need to be appended on save().
        self._persisted_count: int = 0

        if path and path.exists():
            self._load()

    def _load(self) -> None:
        """
        Load session from JSONL file.

        Reads the entire file to find the last snapshot and session metadata,
        then loads only messages after the last snapshot into self.messages.
        """
        last_snapshot_index = -1
        lines: list[str] = []

        for line_no, line in enumerate(self.path.read_text(encoding="utf-8").splitlines()):
            line = line.strip()
            if not line:
                continue
            lines.append(line)
            entry = json.loads(line)
            if entry.get("type") == "meta":
                self.created_at = entry.get("created_at", self.created_at)
                # Restore compaction state from meta
                if "compaction_count" in entry:
                    self._compaction_count = entry["compaction_count"]
            elif entry.get("type") == "snapshot":
                last_snapshot_index = len(lines) - 1

        # Load messages: from last snapshot forward
        start = last_snapshot_index + 1 if last_snapshot_index >= 0 else 0
        self.messages = []
        for line in lines[start:]:
            entry = json.loads(line)
            if entry.get("type") == "message":
                self.messages.append(entry["data"])

        # Also load snapshot messages if there was a snapshot
        if last_snapshot_index >= 0:
            snapshot_entry = json.loads(lines[last_snapshot_index])
            snapshot_messages = snapshot_entry.get("messages", [])
            self.messages = snapshot_messages + self.messages

            # Restore compaction summary
            if "summary" in snapshot_entry:
                self._last_compaction_summary = snapshot_entry["summary"]

        # All loaded messages are considered persisted
        self._persisted_count = len(self.messages)

    def record_compaction(self, result: "CompactResult") -> None:
        """
        Record a compaction event by appending a snapshot to the JSONL.

        The snapshot contains the compacted message list. The full history
        remains in the JSONL file before this snapshot entry — nothing is
        overwritten.

        After calling this, save() will append the snapshot to disk.
        """
        if not result.success:
            return

        # Store the summary as a system message
        summary_msg = {
            "role": "system",
            "content": f"[Compaction Summary]\n{result.summary}",
        }

        # Get recent messages from the result
        recent = getattr(result, "_recent_messages", [])

        # Replace in-memory messages: summary + recent
        self.messages = [summary_msg] + list(recent)

        # Track compaction count
        self._compaction_count = getattr(self, "_compaction_count", 0) + 1
        self._last_compaction_summary = result.summary

        # Write snapshot immediately (append-only)
        self._append_snapshot(result.summary)

    def save(self) -> None:
        """
        Persist new messages to JSONL file (append-only).

        Only appends messages that haven't been written to disk yet.
        Never overwrites existing content.
        """
        if not self.path:
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)

        is_new = not self.path.exists()

        with open(self.path, "a", encoding="utf-8") as f:
            if is_new:
                # Write meta entry for new sessions
                meta = {
                    "type": "meta",
                    "created_at": self.created_at,
                    "saved_at": datetime.now().isoformat(),
                }
                if hasattr(self, "_compaction_count") and self._compaction_count:
                    meta["compaction_count"] = self._compaction_count
                f.write(json.dumps(meta, ensure_ascii=False) + "\n")

            # Append only new messages (beyond _persisted_count)
            for msg in self.messages[self._persisted_count:]:
                f.write(json.dumps({"type": "message", "data": msg}, ensure_ascii=False) + "\n")

        self._persisted_count = len(self.messages)

    def _append_snapshot(self, summary: str) -> None:
        """
        Append a compaction snapshot to the JSONL file.

        The snapshot captures the current compacted message list.
        On next load(), messages before this snapshot are skipped.
        """
        if not self.path:
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Update meta with current compaction count
        meta_update = {
            "type": "meta",
            "created_at": self.created_at,
            "saved_at": datetime.now().isoformat(),
        }
        if hasattr(self, "_compaction_count") and self._compaction_count:
            meta_update["compaction_count"] = self._compaction_count

        snapshot = {
            "type": "snapshot",
            "messages": self.messages,
            "summary": summary,
            "saved_at": datetime.now().isoformat(),
        }
        if hasattr(self, "_compaction_count") and self._compaction_count:
            snapshot["compaction_count"] = self._compaction_count

        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(meta_update, ensure_ascii=False) + "\n")
            f.write(json.dumps(snapshot, ensure_ascii=False) + "\n")

        self._persisted_count = len(self.messages)

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


def fork_session(source: Session, session_dir: str, name: str | None = None) -> Session:
    """
    Create a fork of an existing session.

    Copies all messages from source into a new session file.
    The new session gets a new name and path.
    """
    new_session = create_session(session_dir, name)
    new_session.messages = [msg.copy() for msg in source.messages]
    new_session.token_usage = dict(source.token_usage)
    # Preserve compaction state
    if hasattr(source, "_compaction_count"):
        new_session._compaction_count = source._compaction_count
    if hasattr(source, "_last_compaction_summary"):
        new_session._last_compaction_summary = source._last_compaction_summary
    new_session.save()
    return new_session
