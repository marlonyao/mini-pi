"""
Fake LLM for testing.

Returns predetermined responses, tracks all calls.
No API key needed, no network, fully deterministic.
"""

from __future__ import annotations

from typing import Any, Callable

from .base import LLMBase, ChatResponse, ToolCall, Usage


class FakeLLM(LLMBase):
    """
    Deterministic fake LLM for testing.

    Usage:
        fake = FakeLLM(responses=[
            ChatResponse(content="Hello!"),
            ChatResponse(tool_calls=[
                ToolCall(id="c1", name="bash", arguments='{"command":"ls"}'),
            ]),
        ])
        r1 = fake.chat([{"role": "user", "content": "hi"}])
        assert r1.content == "Hello!"

        r2 = fake.chat([{"role": "user", "content": "run ls"}])
        assert r2.has_tool_calls
    """

    def __init__(
        self,
        responses: list[ChatResponse] | None = None,
        *,
        default_response: ChatResponse | None = None,
    ):
        """
        Args:
            responses: Sequence of responses to return in order.
            default_response: Fallback when responses are exhausted.
                             Defaults to ChatResponse(content="ok").
        """
        self.responses = list(responses) if responses else []
        self.default_response = default_response or ChatResponse(content="ok")
        self._call_index = 0
        self.calls: list[dict[str, Any]] = []

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        on_text: Callable[[str], None] | None = None,
        on_reasoning: Callable[[str], None] | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Return the next predetermined response."""
        # Record the call
        self.calls.append({
            "messages": messages,
            "tools": tools,
            "kwargs": kwargs,
        })

        # Pick response
        if self._call_index < len(self.responses):
            response = self.responses[self._call_index]
        else:
            response = self.default_response
        self._call_index += 1

        # Simulate callbacks
        if response.content and on_text:
            on_text(response.content)
        if response.reasoning_content and on_reasoning:
            on_reasoning(response.reasoning_content)

        return response

    @property
    def call_count(self) -> int:
        """Number of times chat() was called."""
        return len(self.calls)

    @property
    def last_call(self) -> dict[str, Any]:
        """The most recent call's arguments."""
        if not self.calls:
            raise IndexError("No calls made yet")
        return self.calls[-1]

    @property
    def last_messages(self) -> list[dict[str, Any]]:
        """Messages from the most recent call."""
        return self.last_call["messages"]
