"""Graph behavior tests with fake LLM/retriever and an in-memory checkpointer (offline)."""

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

import agentkit.core.graph.nodes as nodes
from agentkit.core.graph import build_graph
from agentkit.retrieval.store import RetrievedChunk


def _patch_llm(monkeypatch, *contents: str):
    fake = GenericFakeChatModel(messages=iter([AIMessage(content=c) for c in contents]))
    monkeypatch.setattr(nodes, "get_llm", lambda *a, **k: fake)


class _FakeRetriever:
    def __init__(self, chunks):
        self._chunks = chunks

    def search(self, query, *, k=5):
        return self._chunks[:k]


def _patch_retriever(monkeypatch, chunks):
    monkeypatch.setattr(nodes, "get_retriever", lambda: _FakeRetriever(chunks))


def _chunk(content, vscore):
    return RetrievedChunk(
        id=1,
        source="overview.md",
        chunk_index=0,
        content=content,
        vscore=vscore,
        lscore=0.5,
        rrf=0.03,
    )


def test_chitchat_skips_retrieval(monkeypatch):
    _patch_llm(monkeypatch, "chitchat", "Hello there!")
    # retriever must NOT be called on the chitchat path
    monkeypatch.setattr(nodes, "get_retriever", lambda: (_ for _ in ()).throw(AssertionError()))
    graph = build_graph(MemorySaver())

    result = graph.invoke({"messages": [HumanMessage("hi")]}, {"configurable": {"thread_id": "c"}})

    assert result["intent"] == "chitchat"
    assert result["messages"][-1].content == "Hello there!"


def test_rag_answer_with_citations(monkeypatch):
    _patch_llm(monkeypatch, "question", "agentkit uses pgvector for retrieval [1].")
    _patch_retriever(monkeypatch, [_chunk("agentkit uses pgvector", vscore=0.9)])
    graph = build_graph(MemorySaver())

    result = graph.invoke(
        {"messages": [HumanMessage("what does agentkit use?")]},
        {"configurable": {"thread_id": "q1"}},
    )

    assert result["intent"] == "question"
    assert "pgvector" in result["messages"][-1].content
    assert result["citations"] == [{"marker": 1, "source": "overview.md", "chunk_index": 0}]
    assert "error" not in result  # grounded


def test_no_answer_below_threshold(monkeypatch):
    # Only classify calls the LLM; the no-answer node is canned.
    _patch_llm(monkeypatch, "question")
    _patch_retriever(monkeypatch, [_chunk("unrelated content", vscore=0.05)])
    graph = build_graph(MemorySaver())

    result = graph.invoke(
        {"messages": [HumanMessage("what is the capital of mars?")]},
        {"configurable": {"thread_id": "q2"}},
    )

    assert result["citations"] == []
    assert "don't have enough information" in result["messages"][-1].content
    assert "error" not in result  # no-answer is not flagged as ungrounded
