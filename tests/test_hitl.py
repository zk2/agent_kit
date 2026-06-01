"""Human-in-the-loop tests: pause at review, resume with approve/edit/reject (offline)."""

import asyncio

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from agentkit.core.graph import build_graph
from tests.test_tools import ScriptedChat, _patch, _tool_call


def _build(monkeypatch, responses):
    saved: list[str] = []

    @tool
    def save_note(text: str) -> str:
        """Persist a note (side effect)."""
        saved.append(text)
        return "saved"

    _patch(monkeypatch, ScriptedChat(responses=responses))
    graph = build_graph(MemorySaver(), tools=[save_note])
    return graph, saved


def _cfg(tid) -> RunnableConfig:
    return {"configurable": {"thread_id": tid}}


def _start(graph, cfg):
    return asyncio.run(graph.ainvoke({"messages": [HumanMessage("save a note")]}, cfg))


def _resume(graph, cfg, decision):
    return asyncio.run(graph.ainvoke(Command(resume=decision), cfg))


def test_pauses_for_review_without_executing(monkeypatch):
    graph, saved = _build(
        monkeypatch,
        [
            AIMessage(content="question"),
            _tool_call("save_note", {"text": "hello"}, "c1"),
            AIMessage(content=""),
            AIMessage(content="Done [1]."),
        ],
    )
    cfg = _cfg("h-pause")

    _start(graph, cfg)

    snapshot = graph.get_state(cfg)
    assert snapshot.next  # a node is pending -> paused
    assert [i for task in snapshot.tasks for i in task.interrupts]  # has an interrupt
    assert saved == []  # side-effecting tool has NOT run


def test_approve_resumes_and_executes(monkeypatch):
    graph, saved = _build(
        monkeypatch,
        [
            AIMessage(content="question"),
            _tool_call("save_note", {"text": "hello"}, "c1"),
            AIMessage(content=""),
            AIMessage(content="Saved it [1]."),
        ],
    )
    cfg = _cfg("h-approve")

    _start(graph, cfg)
    res = _resume(graph, cfg, {"action": "approve"})

    assert saved == ["hello"]  # executed only after approval
    assert "Saved it" in res["messages"][-1].content
    assert not graph.get_state(cfg).next  # completed


def test_reject_skips_tool(monkeypatch):
    graph, saved = _build(
        monkeypatch,
        [
            AIMessage(content="question"),
            _tool_call("save_note", {"text": "hello"}, "c1"),
            AIMessage(content="I did not save anything [1]."),  # synthesize (tools skipped)
        ],
    )
    cfg = _cfg("h-reject")

    _start(graph, cfg)
    res = _resume(graph, cfg, {"action": "reject"})

    assert saved == []  # never executed
    assert "did not save" in res["messages"][-1].content


def test_edit_changes_arguments(monkeypatch):
    graph, saved = _build(
        monkeypatch,
        [
            AIMessage(content="question"),
            _tool_call("save_note", {"text": "hello"}, "c1"),
            AIMessage(content=""),
            AIMessage(content="Saved the edited note [1]."),
        ],
    )
    cfg = _cfg("h-edit")

    _start(graph, cfg)
    res = _resume(
        graph,
        cfg,
        {
            "action": "edit",
            "tool_calls": [{"name": "save_note", "args": {"text": "edited"}, "id": "c1"}],
        },
    )

    assert saved == ["edited"]  # ran with operator-corrected args
    assert "edited" in res["messages"][-1].content
