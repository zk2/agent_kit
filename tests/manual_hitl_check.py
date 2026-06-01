"""Manual integration check for Stage 4: human-in-the-loop with real Postgres, no API key needed.

Proves the HITL DoD including durability:
  - the run pauses at review and the side-effecting tool does NOT run
  - the checkpoint survives closing and reopening the connection pool (process restart)
  - after /resume approve, the run continues from the interrupt and the tool runs exactly once

The LLM is faked; the tool performs a real Postgres write so we can confirm it ran only after
approval.

Run with Postgres up:
    DATABASE_URL=postgresql://agentkit:agentkit@localhost:5432/agentkit \
        python tests/manual_hitl_check.py
"""

import asyncio
from uuid import uuid4

import psycopg
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.types import Command

import agentkit.core.graph.nodes as nodes
from agentkit.config import settings
from agentkit.core.checkpoint import open_checkpointer
from agentkit.core.graph import build_graph
from agentkit.retrieval.store import RetrievedChunk


class ScriptedChat(BaseChatModel):
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


def _reset_notes() -> None:
    with psycopg.connect(settings.database_url, autocommit=True) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS notes (id bigserial PRIMARY KEY, text text)")
        conn.execute("TRUNCATE notes")


def _note_count() -> int:
    with psycopg.connect(settings.database_url, autocommit=True) as conn:
        return conn.execute("SELECT count(*) FROM notes").fetchone()[0]


@tool
def save_note(text: str) -> str:
    """Persist a note (real side effect)."""
    with psycopg.connect(settings.database_url, autocommit=True) as conn:
        conn.execute("INSERT INTO notes (text) VALUES (%s)", (text,))
    return "saved"


def _patch_fakes() -> None:
    llm = ScriptedChat(
        responses=[
            AIMessage(content="question"),
            AIMessage(
                content="",
                tool_calls=[{"name": "save_note", "args": {"text": "hi"}, "id": "c1"}],
            ),
            AIMessage(content=""),
            AIMessage(content="Saved your note [1]."),
        ]
    )
    nodes.get_llm = lambda *a, **k: llm  # type: ignore[assignment]

    chunk = RetrievedChunk(
        id=1, source="overview.md", chunk_index=0, content="info", vscore=0.9, lscore=0.5, rrf=0.03
    )

    class _R:
        def search(self, query, *, k=5):
            return [chunk]

    nodes.get_retriever = lambda: _R()  # type: ignore[assignment]


async def main() -> None:
    _reset_notes()
    _patch_fakes()
    cfg = {"configurable": {"thread_id": f"hitl-{uuid4().hex[:8]}"}}

    # --- process 1: run until it pauses for review ---
    async with open_checkpointer() as saver:
        graph = build_graph(saver, tools=[save_note])
        res = await graph.ainvoke({"messages": [HumanMessage("save a note")]}, cfg)
        assert "__interrupt__" in res, "expected the run to pause at review"
        assert _note_count() == 0, "tool must NOT run before approval"
        print("process 1: paused at review, notes =", _note_count())

    # --- process 2: fresh pool; state must have survived ---
    async with open_checkpointer() as saver:
        graph = build_graph(saver, tools=[save_note])
        snap = await graph.aget_state(cfg)
        assert snap.next, "interrupt state did not survive the restart"
        print("process 2: recovered paused run, next =", snap.next)

        res = await graph.ainvoke(Command(resume={"action": "approve"}), cfg)
        assert _note_count() == 1, "tool must run exactly once after approval"
        assert "Saved your note" in res["messages"][-1].content
        print("process 2: resumed and completed, notes =", _note_count())

    print("OK: paused, survived restart, resumed, side effect applied once")


if __name__ == "__main__":
    asyncio.run(main())
