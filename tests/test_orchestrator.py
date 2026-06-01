"""Tests for the orchestrator interface and the LangGraph engine (offline)."""

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

import agentkit.core.graph.nodes as nodes
from agentkit.core.graph import build_graph
from agentkit.core.orchestrator import (
    ClaudeAgentSDKOrchestrator,
    LangGraphOrchestrator,
    Orchestrator,
    get_orchestrator,
)
from agentkit.eval.runner import calculator
from agentkit.eval.stub_llm import StubChat
from agentkit.retrieval.store import RetrievedChunk
from tests.test_tools import ScriptedChat, _tool_call


def _confident_retriever(monkeypatch):
    chunk = RetrievedChunk(
        id=1, source="deployment.md", chunk_index=0,
        content="served by uvicorn on port 8000", vscore=0.9, lscore=0.5, rrf=0.03,
    )

    class _R:
        def search(self, query, *, k=5):
            return [chunk]

    monkeypatch.setattr(nodes, "get_retriever", lambda: _R())


# --- interface / selection ---------------------------------------------------


def test_both_engines_implement_interface():
    assert issubclass(LangGraphOrchestrator, Orchestrator)
    assert issubclass(ClaudeAgentSDKOrchestrator, Orchestrator)


def test_get_orchestrator_selects_engine():
    assert isinstance(get_orchestrator("langgraph", graph=None), LangGraphOrchestrator)
    assert isinstance(get_orchestrator("claude_agent_sdk"), ClaudeAgentSDKOrchestrator)


async def test_sdk_orchestrator_resume_unsupported():
    with pytest.raises(NotImplementedError):
        await ClaudeAgentSDKOrchestrator().resume("t", {"action": "approve"})


# --- LangGraph engine behaviour ---------------------------------------------


async def test_langgraph_run_produces_traced_result(monkeypatch):
    monkeypatch.setattr(nodes, "get_llm", lambda *a, **k: StubChat())
    _confident_retriever(monkeypatch)
    graph = build_graph(MemorySaver(), tools=[calculator])
    orch = LangGraphOrchestrator(graph)

    rr = await orch.run("What serves the agentkit application and on what port?")

    assert rr.status == "done"
    assert "8000" in rr.response
    assert rr.intent == "question"
    # end-to-end trace covers the nodes that ran
    nodes_traced = [s["node"] for s in rr.trace]
    assert {"classify", "retrieve", "plan", "synthesize", "validate"} <= set(nodes_traced)
    assert rr.latency_ms >= 0.0


async def test_langgraph_run_and_resume_hitl(monkeypatch):
    saved: list[str] = []

    @tool
    def save_note(text: str) -> str:
        """Persist a note (side effect)."""
        saved.append(text)
        return "saved"

    llm = ScriptedChat(
        responses=[
            AIMessage(content="question"),
            _tool_call("save_note", {"text": "hello"}, "c1"),
            AIMessage(content=""),
            AIMessage(content="Saved it [1]."),
        ]
    )
    monkeypatch.setattr(nodes, "get_llm", lambda *a, **k: llm)
    _confident_retriever(monkeypatch)
    graph = build_graph(MemorySaver(), tools=[save_note])
    orch = LangGraphOrchestrator(graph)

    paused = await orch.run("save a note", thread_id="o-hitl")
    assert paused.status == "awaiting_review"
    assert paused.interrupt and paused.interrupt["type"] == "tool_approval"
    assert saved == []

    done = await orch.resume("o-hitl", {"action": "approve"})
    assert done.status == "done"
    assert saved == ["hello"]
