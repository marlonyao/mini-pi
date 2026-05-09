"""
LLM abstraction layer for mini-pi.

Provides a unified interface for different LLM providers.
Key design:
  - Base class returns complete ChatResponse (not streams)
  - Implementations handle streaming internally
  - Real-time output via callbacks (on_text, on_reasoning)
  - Supports DeepSeek reasoning_content, tool_calls, usage tracking
"""

from .base import LLMBase, ChatResponse, ToolCall, Usage
from .openai_llm import OpenAILLM
from .fake import FakeLLM

__all__ = [
    "LLMBase",
    "ChatResponse",
    "ToolCall",
    "Usage",
    "OpenAILLM",
    "FakeLLM",
]
