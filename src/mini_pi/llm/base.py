"""
Base LLM interface — the anti-corruption layer.

All LLM providers implement this interface. The agent loop only depends
on LLMBase, never on OpenAI or any specific SDK.

Key principle: chat() returns a complete ChatResponse.
Streaming is an internal implementation detail.
Real-time output is handled via optional callbacks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolCall:
    """A tool call from the LLM response."""

    id: str
    name: str
    arguments: str  # JSON string

    def to_openai_dict(self) -> dict[str, Any]:
        """Convert to OpenAI tool_call format for message history."""
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.arguments,
            },
        }


@dataclass
class Usage:
    """Token usage from the LLM response."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class ChatResponse:
    """
    Complete response from the LLM.

    This is what chat() returns — always fully assembled,
    never a partial/streaming result.
    """

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    reasoning_content: str | None = None  # DeepSeek thinking mode
    usage: Usage | None = None

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class LLMBase(ABC):
    """
    Abstract base for LLM providers.

    Usage:
        llm = OpenAILLM(api_key="...", base_url="...", model="deepseek-v4-flash")
        response = llm.chat(
            messages=[...],
            tools=[...],
            on_text=lambda s: print(s, end="", flush=True),
        )
        print(response.content)
    """

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        on_text: Callable[[str], None] | None = None,
        on_reasoning: Callable[[str], None] | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """
        Send messages and return a complete response.

        Args:
            messages: OpenAI-format message list.
            tools: OpenAI-format tool definitions.
            on_text: Callback for real-time text output (streaming).
            on_reasoning: Callback for thinking content (DeepSeek).
            **kwargs: Additional provider-specific options.

        Returns:
            A ChatResponse with content, tool_calls, reasoning, and usage.
        """
        ...
