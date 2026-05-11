"""
Tests for mid-execution steering.

Covers:
- Agent.steer() queues messages
- Steering messages are consumed in agent loop
- Multiple steering messages are processed in order
- Steering messages appear as user messages in session
"""

import pytest

from mini_pi.agent import Agent
from mini_pi.config import Config
from mini_pi.llm import FakeLLM, ChatResponse
from mini_pi.session import Session


# ── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def agent(tmp_path):
    """Create an agent with a fake LLM for testing."""
    config = Config(api_key="test-key")
    session = Session(tmp_path / "test.jsonl")
    llm = FakeLLM()
    return Agent(config, session, llm=llm)


# ── Tests ───────────────────────────────────────────────────────────

class TestSteeringQueue:
    def test_steer_adds_to_queue(self, agent):
        assert len(agent._steering_messages) == 0
        agent.steer("check error handling")
        assert len(agent._steering_messages) == 1
        assert agent._steering_messages[0] == "check error handling"

    def test_steer_multiple(self, agent):
        agent.steer("first")
        agent.steer("second")
        agent.steer("third")
        assert len(agent._steering_messages) == 3
        assert agent._steering_messages == ["first", "second", "third"]

    def test_steer_empty_queue(self, agent):
        assert agent._steering_messages == []


class TestSteeringInAgentLoop:
    def test_steering_consumed_during_loop(self, tmp_path):
        """Steering messages are consumed during the agent loop."""
        config = Config(api_key="test-key")
        session = Session(tmp_path / "test.jsonl")

        llm = FakeLLM(responses=[
            ChatResponse(content="Done after steering."),
        ])

        agent = Agent(config, session, llm=llm)

        # Queue a steering message before chat
        agent.steer("focus on edge cases")

        # Chat should consume the steering message
        result = agent.chat("do something")

        # The steering message should have been consumed
        assert len(agent._steering_messages) == 0

        # Session should contain the steering message as a user message
        user_msgs = [m for m in session.messages if m.get("role") == "user"]
        assert any("focus on edge cases" in m.get("content", "") for m in user_msgs)

    def test_steering_appears_in_order(self, tmp_path):
        """Multiple steering messages are processed in FIFO order."""
        config = Config(api_key="test-key")
        session = Session(tmp_path / "test.jsonl")

        llm = FakeLLM(responses=[
            ChatResponse(content="Done."),
        ])

        agent = Agent(config, session, llm=llm)
        agent.steer("first steer")
        agent.steer("second steer")

        agent.chat("go")

        user_msgs = [m["content"] for m in session.messages if m.get("role") == "user"]
        # Original message first, then steering in order
        assert user_msgs[0] == "go"
        assert user_msgs[1] == "first steer"
        assert user_msgs[2] == "second steer"

    def test_steer_from_thread_is_thread_safe(self, tmp_path):
        """Steering from a background thread is safe."""
        import threading

        config = Config(api_key="test-key")
        session = Session(tmp_path / "test.jsonl")

        llm = FakeLLM(responses=[ChatResponse(content="Done.")])
        agent = Agent(config, session, llm=llm)

        # Simulate background thread calling steer
        results = []

        def bg_steer():
            agent.steer("from background")
            results.append(True)

        t = threading.Thread(target=bg_steer)
        t.start()
        t.join()

        assert len(agent._steering_messages) == 1
        assert agent._steering_messages[0] == "from background"
