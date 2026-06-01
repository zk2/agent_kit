"""Orchestrator abstraction.

A single interface (`Orchestrator`) lets the platform run the same RAG+tools scenario on
different engines, so the right one can be chosen per client. Two implementations:

  - LangGraphOrchestrator   : the in-house graph (state, checkpointing, HITL, tracing). Default.
  - ClaudeAgentSDKOrchestrator: the same scenario driven by the Claude Agent SDK's agent loop.

Both return a uniform `RunResult`, so callers (the API) don't depend on the engine.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal
from uuid import uuid4

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from agentkit.config import settings

RunStatus = Literal["running", "awaiting_review", "done", "failed"]


@dataclass
class RunResult:
    thread_id: str
    status: RunStatus
    response: str = ""
    intent: str | None = None
    citations: list[dict] = field(default_factory=list)
    interrupt: dict | None = None
    trace: list[dict] = field(default_factory=list)
    latency_ms: float = 0.0
    cost_usd: float = 0.0


class NotAwaitingReview(Exception):
    """Raised when resume is called on a run that is not paused."""


class Orchestrator(ABC):
    name: str

    @abstractmethod
    async def run(self, message: str, thread_id: str | None = None) -> RunResult: ...

    async def resume(self, thread_id: str, decision: dict) -> RunResult:
        raise NotImplementedError(f"{self.name} does not support human-in-the-loop resume")


def _aggregate_trace(trace: list[dict]) -> tuple[float, float]:
    latency = round(sum(s.get("duration_ms", 0.0) for s in trace), 2)
    cost = round(sum(s.get("cost_usd", 0.0) for s in trace), 6)
    return latency, cost


class LangGraphOrchestrator(Orchestrator):
    name = "langgraph"

    def __init__(self, graph) -> None:
        self._graph = graph

    async def _result(self, thread_id: str, config: dict) -> RunResult:
        # Read status from the checkpointed snapshot - robust across pause/resume and restarts.
        snapshot = await self._graph.aget_state(config)
        values = snapshot.values
        trace = values.get("trace", [])
        latency, cost = _aggregate_trace(trace)

        interrupts = [i for task in snapshot.tasks for i in task.interrupts]
        if interrupts:
            return RunResult(
                thread_id=thread_id,
                status="awaiting_review",
                intent=values.get("intent"),
                interrupt=interrupts[0].value,
                trace=trace,
                latency_ms=latency,
                cost_usd=cost,
            )

        messages = values.get("messages", [])
        answer = str(messages[-1].content) if messages else ""
        if values.get("error") == "empty answer":
            status = "failed"
        elif snapshot.next:
            status = "running"
        else:
            status = "done"
        return RunResult(
            thread_id=thread_id,
            status=status,
            response=answer,
            intent=values.get("intent"),
            citations=values.get("citations", []),
            trace=trace,
            latency_ms=latency,
            cost_usd=cost,
        )

    async def run(self, message: str, thread_id: str | None = None) -> RunResult:
        tid = thread_id or uuid4().hex
        config = {"configurable": {"thread_id": tid}}
        await self._graph.ainvoke({"messages": [HumanMessage(message)]}, config)
        return await self._result(tid, config)

    async def resume(self, thread_id: str, decision: dict) -> RunResult:
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = await self._graph.aget_state(config)
        if not snapshot.values:
            raise LookupError(thread_id)
        if not snapshot.next:
            raise NotAwaitingReview(thread_id)
        await self._graph.ainvoke(Command(resume=decision), config)
        return await self._result(thread_id, config)


_SDK_SYSTEM_PROMPT = (
    "You are agentkit. Answer questions about the knowledge base. Use kb_search to retrieve "
    "context and cite it; use calculator for arithmetic. If kb_search returns nothing relevant, "
    "say you don't have enough information rather than guessing."
)


class ClaudeAgentSDKOrchestrator(Orchestrator):
    """The same scenario driven by the Claude Agent SDK.

    Requires the optional `claude-agent-sdk` dependency and an Anthropic API key (and the Claude
    Code runtime the SDK drives). Imports are lazy so this module loads without the dependency.
    """

    name = "claude_agent_sdk"

    def __init__(self, retriever_factory=None) -> None:
        self._retriever_factory = retriever_factory

    def _retriever(self):
        if self._retriever_factory is not None:
            return self._retriever_factory()
        from agentkit.retrieval.store import get_retriever

        return get_retriever()

    def _build_tools(self, sdk):
        from agentkit.tools.calc import CalcError, safe_eval

        retriever = self._retriever

        @sdk.tool("kb_search", "Search the knowledge base", {"query": str})
        async def kb_search(args):
            chunks = retriever().search(args["query"], k=settings.retrieval_top_k)
            text = "\n\n".join(f"(source: {c.source}#{c.chunk_index}) {c.content}" for c in chunks)
            return {"content": [{"type": "text", "text": text or "no results"}]}

        @sdk.tool("calculator", "Evaluate arithmetic", {"expression": str})
        async def calculator(args):
            try:
                value = str(safe_eval(args["expression"]))
            except CalcError as exc:
                value = f"error: {exc}"
            return {"content": [{"type": "text", "text": value}]}

        return [kb_search, calculator]

    async def run(self, message: str, thread_id: str | None = None) -> RunResult:
        try:
            import claude_agent_sdk as sdk  # type: ignore[import-not-found]  # optional dep
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "claude-agent-sdk is not installed; `pip install '.[agent-sdk]'`"
            ) from exc

        server = sdk.create_sdk_mcp_server(
            name="agentkit", version="0.1.0", tools=self._build_tools(sdk)
        )
        options = sdk.ClaudeAgentOptions(
            mcp_servers={"agentkit": server},
            allowed_tools=["mcp__agentkit__kb_search", "mcp__agentkit__calculator"],
            system_prompt=_SDK_SYSTEM_PROMPT,
            max_turns=settings.max_tool_iterations + 2,
        )

        parts: list[str] = []
        async for msg in sdk.query(prompt=message, options=options):
            if isinstance(msg, sdk.AssistantMessage):
                for block in msg.content:
                    if isinstance(block, sdk.TextBlock):
                        parts.append(block.text)

        return RunResult(
            thread_id=thread_id or uuid4().hex,
            status="done",
            response="".join(parts).strip(),
        )


def get_orchestrator(name: str | None = None, *, graph=None) -> Orchestrator:
    name = name or settings.orchestrator
    if name == "claude_agent_sdk":
        return ClaudeAgentSDKOrchestrator()
    return LangGraphOrchestrator(graph)
