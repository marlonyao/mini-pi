"""
Session management for mini-pi.

Sessions are stored as append-only JSONL files (one JSON object per line).
Each entry records a message with role, content, and optional tool call data.

The JSONL is append-only — new messages are always appended, never overwritten.
This preserves full conversation history even after compaction.

Entry types:
  - meta:       Session metadata (created_at, etc.)
  - message:    A single message (user/assistant/tool)
  - compaction: A compaction entry (summary + first_kept_index + file tracking)

Loading reads from the last compaction entry forward, so only the active
messages are in memory, while the full history remains on disk.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class Session:
    """Manages conversation history with append-only JSONL persistence.

    After compaction, messages are trimmed in memory to the kept portion,
    but the full message list remains on disk. The compaction entry stores
    first_kept_index so that on reload, only the relevant messages are loaded.
    """

    def __init__(self, path: Path | None = None):
        self.path = path
        self.messages: list[dict[str, Any]] = []
        self.created_at = datetime.now().isoformat()
        self.token_usage: dict[str, int] = {"prompt": 0, "completion": 0, "total": 0}

        # Track how many messages have been persisted to disk.
        # Only messages beyond this index need to be appended on save().
        self._persisted_count: int = 0

        # Compaction state (restored from JSONL on load)
        self._compaction_count: int = 0
        self._last_compaction_summary: str = ""
        self._last_first_kept_index: int = 0
        self._last_read_files: list[str] = []
        self._last_modified_files: list[str] = []
        self._last_tokens_before: int = 0
        # Absolute offset: the index in the full JSONL message list where
        # self.messages starts. Used to convert relative first_kept_index
        # to an absolute index for the compaction entry.
        self._base_offset: int = 0

        if path and path.exists():
            self._load()

    # ── Properties for compaction state ────────────────────────────

    @property
    def last_compaction_summary(self) -> str:
        return self._last_compaction_summary

    @property
    def last_read_files(self) -> list[str]:
        return list(self._last_read_files)

    @property
    def last_modified_files(self) -> list[str]:
        return list(self._last_modified_files)

    @property
    def compaction_count(self) -> int:
        return self._compaction_count

    # ── Loading ────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load session from JSONL file.

        Reads the entire file to find the last compaction entry and
        session metadata. Then reconstructs messages:
        - If a compaction entry exists: only messages from first_kept_index onward
        - Otherwise: all messages
        """
        last_compaction_line_idx = -1
        lines: list[str] = []
        all_messages: list[dict[str, Any]] = []

        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            lines.append(line)
            entry = json.loads(line)

            if entry.get("type") == "meta":
                self.created_at = entry.get("created_at", self.created_at)
                # Note: do NOT restore _compaction_count from meta here.
                # The authoritative value comes from the compaction entry below.

            elif entry.get("type") == "compaction":
                last_compaction_line_idx = len(lines) - 1

            elif entry.get("type") == "message":
                all_messages.append(entry["data"])

            elif entry.get("type") == "snapshot":
                # Legacy: convert old snapshot format
                last_compaction_line_idx = len(lines) - 1

        if not lines:
            return

        if last_compaction_line_idx >= 0:
            compaction_entry = json.loads(lines[last_compaction_line_idx])

            if compaction_entry.get("type") == "snapshot":
                # Legacy snapshot migration
                self._last_compaction_summary = compaction_entry.get("summary", "")
                snapshot_messages = compaction_entry.get("messages", [])
                self._last_first_kept_index = 0
                self.messages = list(snapshot_messages)
            else:
                # New compaction entry
                first_kept = compaction_entry.get("first_kept_index", 0)
                self._last_compaction_summary = compaction_entry.get("summary", "")
                self._last_first_kept_index = first_kept
                self._last_read_files = compaction_entry.get("read_files", [])
                self._last_modified_files = compaction_entry.get("modified_files", [])
                self._last_tokens_before = compaction_entry.get("tokens_before", 0)
                self._compaction_count = compaction_entry.get(
                    "compaction_count", self._compaction_count
                )
                self._base_offset = first_kept
                self.messages = all_messages[first_kept:]
        else:
            self.messages = all_messages

        self._persisted_count = len(self.messages)

    # ── Compaction Recording ───────────────────────────────────────

    def record_compaction(self, result: "CompactResult") -> None:
        """Record a compaction event by appending a compaction entry to JSONL.

        The compaction entry stores the summary, first_kept_index (absolute
        in JSONL), and file tracking data. The full message history remains
        on disk — nothing is overwritten.

        After recording, self.messages is trimmed to the kept portion.
        """
        if not result.success:
            return

        # Convert relative first_kept_index to absolute JSONL index
        absolute_first_kept = self._base_offset + result.first_kept_index

        # Update compaction state
        self._compaction_count += 1
        self._last_compaction_summary = result.summary
        self._last_first_kept_index = absolute_first_kept
        self._last_read_files = result.read_files
        self._last_modified_files = result.modified_files
        self._last_tokens_before = result.tokens_before

        # Trim in-memory messages to the kept portion
        self.messages = self.messages[result.first_kept_index:]
        self._persisted_count = len(self.messages)

        # Update base offset
        self._base_offset = absolute_first_kept

        # Write compaction entry immediately (append-only)
        self._append_compaction_entry(result, absolute_first_kept)

    # ── Persistence ────────────────────────────────────────────────

    def save(self) -> None:
        """Persist new messages to JSONL file (append-only).

        Only appends messages that haven't been written to disk yet.
        Never overwrites existing content.
        """
        if not self.path:
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)

        is_new = not self.path.exists()

        with open(self.path, "a", encoding="utf-8") as f:
            if is_new:
                meta = {
                    "type": "meta",
                    "created_at": self.created_at,
                    "saved_at": datetime.now().isoformat(),
                }
                if self._compaction_count:
                    meta["compaction_count"] = self._compaction_count
                f.write(json.dumps(meta, ensure_ascii=False) + "\n")

            # Append only new messages (beyond _persisted_count)
            for msg in self.messages[self._persisted_count:]:
                f.write(json.dumps({"type": "message", "data": msg}, ensure_ascii=False) + "\n")

        self._persisted_count = len(self.messages)

    def _append_compaction_entry(self, result: "CompactResult", absolute_first_kept: int) -> None:
        """Append a compaction entry to the JSONL file.

        The entry captures the summary, split boundary, and file tracking.
        On next load(), messages before first_kept_index are skipped.
        first_kept_index is an ABSOLUTE index into the full JSONL message list.
        """
        if not self.path:
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)

        entry: dict[str, Any] = {
            "type": "compaction",
            "summary": result.summary,
            "first_kept_index": absolute_first_kept,
            "tokens_before": result.tokens_before,
            "read_files": result.read_files,
            "modified_files": result.modified_files,
            "compaction_count": self._compaction_count,
            "saved_at": datetime.now().isoformat(),
        }

        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # ── Message Operations ─────────────────────────────────────────

    def add(self, role: str, content: str | None = None, **kwargs: Any) -> dict:
        """Add a message and return it."""
        msg: dict[str, Any] = {"role": role}
        if content is not None:
            msg["content"] = content
        msg.update(kwargs)
        self.messages.append(msg)
        return msg

    def add_user(self, content: str) -> dict:
        """Add a user message."""
        return self.add("user", content)

    def add_assistant(self, content: str | None = None, **kwargs: Any) -> dict:
        """Add an assistant message."""
        return self.add("assistant", content, **kwargs)

    def add_tool_result(self, tool_call_id: str, content: str) -> dict:
        """Add a tool result message."""
        return self.add("tool", content, tool_call_id=tool_call_id)

    def get_openai_messages(self) -> list[dict[str, Any]]:
        """Get messages in OpenAI API format.

        If there's a compaction summary, it's prepended as a system message.
        """
        result: list[dict[str, Any]] = []

        # Inject compaction summary if present
        if self._last_compaction_summary:
            result.append({
                "role": "system",
                "content": (
                    "[Previous conversation summary]\n\n"
                    f"{self._last_compaction_summary}\n\n"
                    "[End of summary. Continue from here.]"
                ),
            })

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
            result.append(entry)

        return result

    def update_token_usage(self, usage: dict) -> None:
        """Accumulate token usage stats."""
        self.token_usage["prompt"] += usage.get("prompt_tokens", 0)
        self.token_usage["completion"] += usage.get("completion_tokens", 0)
        self.token_usage["total"] += usage.get("total_tokens", 0)


# ── Session Factory Functions ───────────────────────────────────────


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
    """Create a fork of an existing session.

    Copies all messages from source into a new session file.
    The new session gets a new name and path.
    """
    new_session = create_session(session_dir, name)
    new_session.messages = [msg.copy() for msg in source.messages]
    new_session.token_usage = dict(source.token_usage)
    # Preserve compaction state
    new_session._compaction_count = source._compaction_count
    new_session._last_compaction_summary = source._last_compaction_summary
    new_session._last_first_kept_index = source._last_first_kept_index
    new_session._last_read_files = list(source._last_read_files)
    new_session._last_modified_files = list(source._last_modified_files)
    new_session._last_tokens_before = source._last_tokens_before
    new_session._base_offset = source._base_offset
    new_session.save()
    return new_session
