"""
Tests for the LLM abstraction layer.

Tests:
- LLMBase interface contract
- OpenAILLM streaming/sync assembly
- FakeLLM for deterministic testing
- ChatResponse / ToolCall / Usage dataclasses
"""

import pytest

from mini_pi.llm import LLMBase, ChatResponse, ToolCall, Usage, FakeLLM


# ── Dataclass tests ────────────────────────────────────────────────

class TestToolCall:
    def test_to_openai_dict(self):
        tc = ToolCall(id="call_1", name="bash", arguments='{"command":"ls"}')
        d = tc.to_openai_dict()
        assert d["id"] == "call_1"
        assert d["type"] == "function"
        assert d["function"]["name"] == "bash"
        assert d["function"]["arguments"] == '{"command":"ls"}'

    def test_equality(self):
        tc1 = ToolCall(id="c1", name="read", arguments='{}')
        tc2 = ToolCall(id="c1", name="read", arguments='{}')
        assert tc1 == tc2


class TestUsage:
    def test_to_dict(self):
        u = Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        d = u.to_dict()
        assert d == {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}

    def test_defaults(self):
        u = Usage()
        assert u.prompt_tokens == 0
        assert u.completion_tokens == 0


class TestChatResponse:
    def test_has_tool_calls_true(self):
        r = ChatResponse(tool_calls=[ToolCall(id="c1", name="bash", arguments="{}")])
        assert r.has_tool_calls is True

    def test_has_tool_calls_false(self):
        r = ChatResponse(content="hello")
        assert r.has_tool_calls is False

    def test_empty_response(self):
        r = ChatResponse()
        assert r.content == ""
        assert r.tool_calls == []
        assert r.reasoning_content is None
        assert r.usage is None
        assert r.has_tool_calls is False

    def test_with_reasoning(self):
        r = ChatResponse(
            content="The answer is 42",
            reasoning_content="Let me think about this...",
        )
        assert r.reasoning_content == "Let me think about this..."


# ── FakeLLM tests ──────────────────────────────────────────────────

class TestFakeLLM:
    def test_returns_responses_in_order(self):
        fake = FakeLLM(responses=[
            ChatResponse(content="first"),
            ChatResponse(content="second"),
            ChatResponse(content="third"),
        ])

        r1 = fake.chat([{"role": "user", "content": "hi"}])
        assert r1.content == "first"

        r2 = fake.chat([{"role": "user", "content": "hello"}])
        assert r2.content == "second"

        r3 = fake.chat([{"role": "user", "content": "hey"}])
        assert r3.content == "third"

    def test_returns_default_after_exhausted(self):
        fake = FakeLLM(
            responses=[ChatResponse(content="only one")],
            default_response=ChatResponse(content="default"),
        )

        fake.chat([{"role": "user", "content": "hi"}])
        r2 = fake.chat([{"role": "user", "content": "hi again"}])
        assert r2.content == "default"

    def test_default_default_response(self):
        fake = FakeLLM()
        r = fake.chat([{"role": "user", "content": "hi"}])
        assert r.content == "ok"

    def test_tracks_calls(self):
        fake = FakeLLM()
        fake.chat([{"role": "user", "content": "first"}])
        fake.chat([{"role": "user", "content": "second"}])

        assert fake.call_count == 2
        assert fake.last_messages[-1]["content"] == "second"

    def test_tracks_tools_kwarg(self):
        fake = FakeLLM()
        tools = [{"type": "function", "function": {"name": "bash"}}]
        fake.chat([{"role": "user", "content": "run"}], tools=tools)

        assert fake.last_call["tools"] == tools

    def test_tool_call_response(self):
        fake = FakeLLM(responses=[
            ChatResponse(
                tool_calls=[
                    ToolCall(id="c1", name="bash", arguments='{"command":"ls"}'),
                ],
            ),
        ])

        r = fake.chat([{"role": "user", "content": "run ls"}])
        assert r.has_tool_calls
        assert r.tool_calls[0].name == "bash"
        assert r.tool_calls[0].to_openai_dict()["type"] == "function"

    def test_reasoning_content_response(self):
        fake = FakeLLM(responses=[
            ChatResponse(
                content="42",
                reasoning_content="thinking...",
            ),
        ])

        r = fake.chat([{"role": "user", "content": "what is 6*7?"}])
        assert r.content == "42"
        assert r.reasoning_content == "thinking..."

    def test_usage_response(self):
        fake = FakeLLM(responses=[
            ChatResponse(
                content="hi",
                usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            ),
        ])

        r = fake.chat([{"role": "user", "content": "hello"}])
        assert r.usage is not None
        assert r.usage.total_tokens == 15

    def test_on_text_callback(self):
        collected = []
        fake = FakeLLM(responses=[ChatResponse(content="hello")])
        r = fake.chat(
            [{"role": "user", "content": "hi"}],
            on_text=lambda s: collected.append(s),
        )

        assert r.content == "hello"
        assert collected == ["hello"]

    def test_on_reasoning_callback(self):
        collected = []
        fake = FakeLLM(responses=[
            ChatResponse(content="answer", reasoning_content="thinking"),
        ])
        r = fake.chat(
            [{"role": "user", "content": "why?"}],
            on_reasoning=lambda s: collected.append(s),
        )

        assert collected == ["thinking"]

    def test_no_callbacks_no_crash(self):
        fake = FakeLLM(responses=[ChatResponse(content="ok")])
        r = fake.chat([{"role": "user", "content": "hi"}])
        assert r.content == "ok"

    def test_empty_responses_list(self):
        fake = FakeLLM(responses=[])
        r = fake.chat([{"role": "user", "content": "hi"}])
        assert r.content == "ok"  # default

    def test_last_call_raises_when_no_calls(self):
        fake = FakeLLM()
        with pytest.raises(IndexError):
            _ = fake.last_call


# ── Agent + FakeLLM integration ────────────────────────────────────

class TestAgentWithFakeLLM:
    """Test the agent loop using FakeLLM — no real API calls."""

    def test_simple_conversation(self, tmp_path):
        """Agent returns text response without tool calls."""
        from mini_pi.agent import Agent
        from mini_pi.config import Config
        from mini_pi.session import Session

        fake = FakeLLM(responses=[
            ChatResponse(content="Hello! How can I help?"),
        ])
        config = Config(api_key="test", session_dir=str(tmp_path))
        session = Session(tmp_path / "test.jsonl")
        agent = Agent(config, session, llm=fake)

        result = agent.chat("hi")
        assert result == "Hello! How can I help?"
        assert fake.call_count == 1

    def test_tool_call_then_response(self, tmp_path):
        """Agent executes tool call, then gets final response."""
        from mini_pi.agent import Agent
        from mini_pi.config import Config
        from mini_pi.session import Session

        fake = FakeLLM(responses=[
            # First call: tool call
            ChatResponse(
                tool_calls=[
                    ToolCall(id="c1", name="bash", arguments='{"command":"echo hi"}'),
                ],
            ),
            # Second call: final text
            ChatResponse(content="The output was 'hi'"),
        ])
        config = Config(api_key="test", session_dir=str(tmp_path))
        session = Session(tmp_path / "test.jsonl")
        agent = Agent(config, session, llm=fake)

        result = agent.chat("run echo hi")
        assert result == "The output was 'hi'"
        assert fake.call_count == 2

        # Verify session recorded both turns
        msgs = session.get_openai_messages()
        assert len(msgs) == 4  # user + assistant(tool_call) + tool_result + assistant(text)

    def test_multiple_tool_calls(self, tmp_path):
        """Agent handles multiple sequential tool calls."""
        from mini_pi.agent import Agent
        from mini_pi.config import Config
        from mini_pi.session import Session

        fake = FakeLLM(responses=[
            ChatResponse(
                tool_calls=[
                    ToolCall(id="c1", name="bash", arguments='{"command":"ls"}'),
                ],
            ),
            ChatResponse(
                tool_calls=[
                    ToolCall(id="c2", name="bash", arguments='{"command":"pwd"}'),
                ],
            ),
            ChatResponse(content="Files listed and directory shown."),
        ])
        config = Config(api_key="test", session_dir=str(tmp_path))
        session = Session(tmp_path / "test.jsonl")
        agent = Agent(config, session, llm=fake)

        result = agent.chat("list files and show dir")
        assert result == "Files listed and directory shown."
        assert fake.call_count == 3

    def test_reasoning_content_preserved(self, tmp_path):
        """DeepSeek-style reasoning_content is stored in session."""
        from mini_pi.agent import Agent
        from mini_pi.config import Config
        from mini_pi.session import Session

        fake = FakeLLM(responses=[
            ChatResponse(
                content="The answer is 42.",
                reasoning_content="Let me calculate 6*7...",
            ),
        ])
        config = Config(api_key="test", session_dir=str(tmp_path))
        session = Session(tmp_path / "test.jsonl")
        agent = Agent(config, session, llm=fake)

        result = agent.chat("what is 6*7?")
        assert result == "The answer is 42."

        # Reasoning should be stored in session
        msgs = session.get_openai_messages()
        assistant_msgs = [m for m in msgs if m["role"] == "assistant"]
        assert assistant_msgs[0].get("reasoning_content") == "Let me calculate 6*7..."

    def test_usage_tracked(self, tmp_path):
        """Token usage is tracked in session."""
        from mini_pi.agent import Agent
        from mini_pi.config import Config
        from mini_pi.session import Session

        fake = FakeLLM(responses=[
            ChatResponse(
                content="done",
                usage=Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
            ),
        ])
        config = Config(api_key="test", session_dir=str(tmp_path))
        session = Session(tmp_path / "test.jsonl")
        agent = Agent(config, session, llm=fake)

        agent.chat("do something")
        assert session.token_usage["total"] == 150
