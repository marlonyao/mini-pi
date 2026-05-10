"""
OpenAI-compatible LLM implementation.

Supports any provider that uses the OpenAI chat completions API format:
- OpenAI, DeepSeek, Moonshot, etc.
- Streaming with real-time callbacks
- DeepSeek reasoning_content (thinking mode)
- Non-streaming fallback
"""

from __future__ import annotations

from typing import Any, Callable

from openai import OpenAI

from .base import LLMBase, ChatResponse, ToolCall, Usage


class OpenAILLM(LLMBase):
    """LLM provider using the OpenAI SDK (compatible with DeepSeek, etc.)."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o",
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

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
        Call the LLM. Tries streaming first, falls back to sync.
        """
        try:
            return self._chat_streaming(messages, tools, on_text, on_reasoning, **kwargs)
        except Exception:
            # Provider doesn't support streaming
            return self._chat_sync(messages, tools, **kwargs)

    def _chat_streaming(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        on_text: Callable[[str], None] | None,
        on_reasoning: Callable[[str], None] | None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Streaming implementation — assembles deltas into a complete response."""
        # Build call kwargs — kwargs may override defaults
        call_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "temperature": 0,
            "stream": True,
        }
        call_kwargs.update(kwargs)  # kwargs overrides defaults

        stream = self.client.chat.completions.create(**call_kwargs)

        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls_map: dict[int, dict] = {}
        usage = None

        for chunk in stream:
            # Usage-only chunk (some providers send this separately)
            if not chunk.choices and hasattr(chunk, "usage") and chunk.usage:
                usage = chunk.usage
                continue
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            if delta is None:
                continue

            # Reasoning content (DeepSeek thinking mode)
            reasoning_delta = getattr(delta, "reasoning_content", None) or ""
            if reasoning_delta:
                reasoning_parts.append(reasoning_delta)
                if on_reasoning:
                    on_reasoning(reasoning_delta)

            # Text content
            text_delta = delta.content or ""
            if text_delta:
                text_parts.append(text_delta)
                if on_text:
                    on_text(text_delta)

            # Tool calls — collect incrementally
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_calls_map:
                        tool_calls_map[idx] = {
                            "id": "",
                            "function": {"name": "", "arguments": ""},
                        }
                    tc = tool_calls_map[idx]
                    if tc_delta.id:
                        tc["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            tc["function"]["name"] += tc_delta.function.name
                        if tc_delta.function.arguments:
                            tc["function"]["arguments"] += tc_delta.function.arguments

            # Usage from last chunk
            if hasattr(chunk.choices[0], "usage") and chunk.choices[0].usage:
                usage = chunk.choices[0].usage

        # Assemble final response
        content = "".join(text_parts)
        reasoning_content = "".join(reasoning_parts) if reasoning_parts else None

        tool_calls = [
            ToolCall(
                id=tc["id"],
                name=tc["function"]["name"],
                arguments=tc["function"]["arguments"],
            )
            for tc in (tool_calls_map[i] for i in sorted(tool_calls_map.keys()))
        ]

        usage_obj = None
        if usage:
            usage_obj = Usage(
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                total_tokens=getattr(usage, "total_tokens", 0) or 0,
            )

        return ChatResponse(
            content=content,
            tool_calls=tool_calls,
            reasoning_content=reasoning_content,
            usage=usage_obj,
        )

    def _chat_sync(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Non-streaming fallback."""
        call_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "temperature": 0,
        }
        call_kwargs.update(kwargs)

        response = self.client.chat.completions.create(**call_kwargs)

        msg = response.choices[0].message
        content = msg.content or ""
        reasoning_content = getattr(msg, "reasoning_content", None) or None

        tool_calls = []
        if msg.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=tc.function.arguments,
                )
                for tc in msg.tool_calls
            ]

        usage_obj = None
        if response.usage:
            usage_obj = Usage(
                prompt_tokens=response.usage.prompt_tokens or 0,
                completion_tokens=response.usage.completion_tokens or 0,
                total_tokens=response.usage.total_tokens or 0,
            )

        return ChatResponse(
            content=content,
            tool_calls=tool_calls,
            reasoning_content=reasoning_content,
            usage=usage_obj,
        )
