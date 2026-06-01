"""Tests for the tools layer: safe calculator + the graph plan<->tools loop (offline)."""

import asyncio

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

import agentkit.core.graph.nodes as nodes
from agentkit.core.graph import build_graph
from agentkit.retrieval.store import RetrievedChunk
from agentkit.tools.calc import CalcError, safe_eval

# --- calculator --------------------------------------------------------------


def test_safe_eval_arithmetic():
    assert safe_eval("2 + 3 * (4 - 1)") == 11
    assert safe_eval("2 ** 10") == 1024
    assert safe_eval("-7 % 3") == 2


def test_safe_eval_rejects_non_arithmetic():
    for bad in ["__import__('os')", "open('x')", "a + 1", "1; 2"]:
        with pytest.raises(CalcError):
            safe_eval(bad)


# --- scripted LLM + graph helpers --------------------------------------------


class ScriptedChat(BaseChatModel):
    """Returns a fixed sequence of messages; supports bind_tools (no-op)."""

    responses: list = []
    cursor: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        msg = self.responses[self.cursor]
        self.cursor += 1
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def bind_tools(self, tools, **kwargs):
        return self


def _patch(monkeypatch, llm, tools_confident=True):
    monkeypatch.setattr(nodes, "get_llm", lambda *a, **k: llm)
    chunk = RetrievedChunk(
        id=1, source="overview.md", chunk_index=0,
        content="agentkit info", vscore=0.9 if tools_confident else 0.05, lscore=0.5, rrf=0.03,
    )

    class _R:
        def search(self, query, *, k=5):
            return [chunk]

    monkeypatch.setattr(nodes, "get_retriever", lambda: _R())


def _tool_call(name, args, cid):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": cid}])


def _run(tools, llm, monkeypatch, thread):
    _patch(monkeypatch, llm)
    graph = build_graph(MemorySaver(), tools=tools)
    return asyncio.run(
        graph.ainvoke(
            {"messages": [HumanMessage("compute something")]},
            {"configurable": {"thread_id": thread}},
        )
    )


# --- tool loop ---------------------------------------------------------------


def test_tool_is_called_then_synthesized(monkeypatch):
    @tool
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    llm = ScriptedChat(
        responses=[
            AIMessage(content="question"),            # classify
            _tool_call("add", {"a": 2, "b": 3}, "c1"),  # plan -> call tool
            AIMessage(content=""),                    # plan -> no more tools
            AIMessage(content="The sum is 5 [1]."),   # synthesize
        ]
    )
    result = _run([add], llm, monkeypatch, "t-add")

    tool_msgs = [m for m in result["messages"] if getattr(m, "type", None) == "tool"]
    assert any("5" in str(m.content) for m in tool_msgs)
    assert result["messages"][-1].content == "The sum is 5 [1]."
    assert result["citations"] == [{"marker": 1, "source": "overview.md", "chunk_index": 0}]
    assert result["tool_iterations"] == 1


def test_tool_error_is_survived(monkeypatch):
    @tool
    def boom(x: int) -> int:
        """Always fails."""
        raise ValueError("kaboom")

    llm = ScriptedChat(
        responses=[
            AIMessage(content="question"),
            _tool_call("boom", {"x": 1}, "c1"),
            AIMessage(content=""),
            AIMessage(content="Recovered despite the tool error [1]."),
        ]
    )
    result = _run([boom], llm, monkeypatch, "t-boom")

    tool_msgs = [m for m in result["messages"] if getattr(m, "type", None) == "tool"]
    assert any("kaboom" in str(m.content) for m in tool_msgs)  # error captured, not raised
    assert "Recovered" in result["messages"][-1].content


def test_iteration_cap_stops_loop(monkeypatch):
    seen = {"n": 0}

    @tool
    def inc(x: int) -> int:
        """Increment x."""
        seen["n"] += 1
        return x + 1

    # The model keeps asking for the tool; the cap (3) must stop the loop.
    llm = ScriptedChat(
        responses=[
            AIMessage(content="question"),
            _tool_call("inc", {"x": 1}, "c1"),
            _tool_call("inc", {"x": 2}, "c2"),
            _tool_call("inc", {"x": 3}, "c3"),
            _tool_call("inc", {"x": 4}, "c4"),
            AIMessage(content="Final answer after cap [1]."),
        ]
    )
    result = _run([inc], llm, monkeypatch, "t-cap")

    assert result["tool_iterations"] == 3
    assert seen["n"] == 3  # tool executed exactly max_tool_iterations times
    assert "Final answer" in result["messages"][-1].content
