"""Manual check for Stage 6: end-to-end decision trace via the orchestrator (real Postgres).

Runs a question through the LangGraph engine and prints the per-node trace (decision + latency)
that travels with the run state, proving every decision is observable end to end. No API key:
uses the deterministic stub model and the real retriever.

Run with Postgres up:
    DATABASE_URL=postgresql://agentkit:agentkit@localhost:5432/agentkit \
        python tests/manual_obs_check.py
"""

import asyncio
import json

import agentkit.core.graph.nodes as nodes
from agentkit.core.checkpoint import open_checkpointer
from agentkit.core.graph import build_graph
from agentkit.core.orchestrator import LangGraphOrchestrator
from agentkit.eval.runner import calculator
from agentkit.eval.stub_llm import StubChat
from agentkit.retrieval.ingest import ingest_path

_CORPUS = "agentkit/eval/corpus"


async def main() -> None:
    ingest_path(_CORPUS, reset=True)
    nodes.get_llm = lambda *a, **k: StubChat()  # type: ignore[assignment]

    async with open_checkpointer() as saver:
        graph = build_graph(saver, tools=[calculator])
        orch = LangGraphOrchestrator(graph)

        rr = await orch.run("How does agentkit combine vector and full text search?")

        print("status :", rr.status)
        print("intent :", rr.intent)
        print(f"latency: {rr.latency_ms} ms   cost: ${rr.cost_usd}")
        print("trace  :")
        for span in rr.trace:
            print("  ", json.dumps(span, default=str))

        traced = [s["node"] for s in rr.trace]
        assert traced == ["classify", "retrieve", "plan", "synthesize", "validate"], traced
        assert rr.status == "done"
        assert rr.latency_ms >= 0.0

    print("\nOK: every node decision traced end-to-end through the orchestrator")


if __name__ == "__main__":
    asyncio.run(main())
