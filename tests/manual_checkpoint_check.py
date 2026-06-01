"""Manual integration check: real Postgres checkpointer + fake LLM, no API key needed.

Proves Stage 1 DoD: a run's state is written to Postgres and is recoverable after the pool is
closed and reopened (simulating a process restart).

Run with Postgres up:
    DATABASE_URL=postgresql://agentkit:agentkit@localhost:5432/agentkit \
        python tests/manual_checkpoint_check.py
"""

import asyncio
from uuid import uuid4

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage

import agentkit.core.graph.nodes as nodes
from agentkit.core.checkpoint import open_checkpointer
from agentkit.core.graph import build_graph


def _patch_fake_llm() -> None:
    # chitchat path keeps this a pure checkpoint test (no DB-backed retriever).
    fake = GenericFakeChatModel(
        messages=iter([AIMessage(content="chitchat"), AIMessage(content="Hi there!")])
    )
    nodes.get_llm = lambda *a, **k: fake  # type: ignore[assignment]


async def main() -> None:
    _patch_fake_llm()
    config = {"configurable": {"thread_id": f"manual-{uuid4().hex[:8]}"}}

    # --- "process 1": run the graph, persist to Postgres ---
    async with open_checkpointer() as saver:
        graph = build_graph(saver)
        result = await graph.ainvoke({"messages": [HumanMessage("hello")]}, config)
        assert result["intent"] == "chitchat", result
        assert result["messages"][-1].content == "Hi there!"
        print("process 1: ran graph, intent =", result["intent"])

    # --- "process 2": brand-new pool, read the state back ---
    async with open_checkpointer() as saver:
        graph = build_graph(saver)
        snapshot = await graph.aget_state(config)
        assert snapshot.values["intent"] == "chitchat", snapshot.values
        n = len(snapshot.values["messages"])
        assert n == 2
        print("process 2: recovered state after restart, messages =", n)

    print("OK: state persisted and recovered across pools")


if __name__ == "__main__":
    asyncio.run(main())
