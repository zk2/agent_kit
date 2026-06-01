"""Manual integration check for Stage 2: real Postgres + pgvector, no API key needed.

Proves the RAG DoD end to end:
  - ingest sample docs -> chunk -> embed -> store in pgvector
  - in-base question  -> answer WITH citations
  - out-of-base question -> no-answer fallback (no citations)

The LLM is faked (synth output is canned) so no API key is required; the retrieval, hybrid
search and no-answer decision are exercised for real against Postgres.

Run with Postgres up:
    DATABASE_URL=postgresql://agentkit:agentkit@localhost:5432/agentkit \
        python tests/manual_rag_check.py
"""

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

import agentkit.core.graph.nodes as nodes
from agentkit.core.graph import build_graph
from agentkit.retrieval.ingest import ingest_path
from agentkit.retrieval.store import get_retriever


def _patch_llm() -> None:
    # invoke 1: classify -> "question", synth -> grounded answer citing [1]
    # invoke 2: classify -> "question"  (no-answer path makes no synth call)
    fake = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(content="question"),
                AIMessage(content="agentkit stores chunks in pgvector [1]."),
                AIMessage(content="question"),
            ]
        )
    )
    nodes.get_llm = lambda *a, **k: fake  # type: ignore[assignment]


def main() -> None:
    get_retriever().clear()
    ingest_path("data/docs", reset=True)
    _patch_llm()

    graph = build_graph(MemorySaver())

    # --- in-base question -> grounded answer with citations ---
    r1 = graph.invoke(
        {"messages": [HumanMessage("Where does agentkit store retrieval chunks?")]},
        {"configurable": {"thread_id": "rag-1"}},
    )
    print(f"in-base : score={r1['retrieval_score']:.3f} citations={r1['citations']}")
    assert r1["citations"], "expected citations on an in-base question"
    assert "pgvector" in r1["messages"][-1].content
    assert "error" not in r1

    # --- out-of-base question -> no-answer fallback ---
    r2 = graph.invoke(
        {"messages": [HumanMessage("What were quarterly avocado prices in Lisbon markets?")]},
        {"configurable": {"thread_id": "rag-2"}},
    )
    print(f"out-base: score={r2['retrieval_score']:.3f} citations={r2['citations']}")
    assert not r2["citations"], "expected no citations on an out-of-base question"
    assert "don't have enough information" in r2["messages"][-1].content

    print("OK: grounded answer on in-base, no-answer fallback on out-of-base")


if __name__ == "__main__":
    main()
