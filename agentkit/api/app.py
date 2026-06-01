"""FastAPI application.

Endpoints:
  POST /chat              -> run the agent for a new or existing thread
  POST /runs/{id}/resume  -> resume a run paused for human review (approve/edit/reject)
  GET  /runs/{id}         -> inspect a run's state (status, history, trace, latency, cost)
  GET  /healthz           -> liveness

A single Orchestrator (LangGraph by default) drives runs; the API does not depend on the engine.
The compiled graph and orchestrator are created once at startup and shared across requests.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException

import agentkit.core.graph.nodes as graph_nodes
from agentkit.api.schemas import (
    ChatRequest,
    ChatResponse,
    Citation,
    MessageView,
    ResumeRequest,
    RunView,
)
from agentkit.config import settings
from agentkit.core.checkpoint import open_checkpointer
from agentkit.core.graph import build_graph
from agentkit.core.orchestrator import LangGraphOrchestrator, NotAwaitingReview, RunResult
from agentkit.tools.client import load_mcp_tools

logger = logging.getLogger(__name__)


def _maybe_use_stub_llm() -> None:
    """For manual testing/demos: serve the API with a deterministic model and no API key."""
    if settings.app_llm == "stub":
        from agentkit.eval.stub_llm import StubChat

        stub = StubChat()
        graph_nodes.get_llm = lambda *a, **k: stub
        logger.warning("APP_LLM=stub: using deterministic stub model (no real LLM calls)")

_ROLE_BY_TYPE = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Tools are optional: if the MCP server can't be loaded, the agent still runs without them.
    try:
        tools = await load_mcp_tools()
        logger.info("loaded %d MCP tool(s): %s", len(tools), [t.name for t in tools])
    except Exception:
        logger.exception("failed to load MCP tools; continuing without tools")
        tools = []

    _maybe_use_stub_llm()

    async with open_checkpointer() as checkpointer:
        app.state.graph = build_graph(checkpointer, tools=tools)
        # The API uses the LangGraph engine (state/checkpointing/HITL/tracing). The orchestrator
        # interface allows swapping engines without touching the endpoints.
        app.state.orchestrator = LangGraphOrchestrator(app.state.graph)
        yield


app = FastAPI(title="agentkit", version="0.1.0", lifespan=lifespan)


def _serialize(messages: list) -> list[MessageView]:
    views = []
    for m in messages:
        role = _ROLE_BY_TYPE.get(getattr(m, "type", ""), getattr(m, "type", "unknown"))
        content = m.content if isinstance(m.content, str) else str(m.content)
        views.append(MessageView(role=role, content=content))
    return views


def _to_response(rr: RunResult) -> ChatResponse:
    return ChatResponse(
        thread_id=rr.thread_id,
        status=rr.status,
        intent=rr.intent,
        response=rr.response,
        citations=[Citation(**c) for c in rr.citations],
        interrupt=rr.interrupt,
        latency_ms=rr.latency_ms,
        cost_usd=rr.cost_usd,
    )


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    rr = await app.state.orchestrator.run(req.message, req.thread_id)
    return _to_response(rr)


@app.post("/runs/{thread_id}/resume", response_model=ChatResponse)
async def resume_run(thread_id: str, req: ResumeRequest) -> ChatResponse:
    decision: dict[str, Any] = {"action": req.action}
    if req.tool_calls is not None:
        decision["tool_calls"] = req.tool_calls
    try:
        rr = await app.state.orchestrator.resume(thread_id, decision)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="run not found") from exc
    except NotAwaitingReview as exc:
        raise HTTPException(status_code=409, detail="run is not awaiting review") from exc
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    return _to_response(rr)


@app.get("/runs/{thread_id}", response_model=RunView)
async def get_run(thread_id: str) -> RunView:
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = await app.state.graph.aget_state(config)

    if not snapshot.values:
        raise HTTPException(status_code=404, detail="run not found")

    interrupts = [i for task in snapshot.tasks for i in task.interrupts]
    if interrupts:
        status = "awaiting_review"
    elif snapshot.next:
        status = "running"
    elif snapshot.values.get("error") == "empty answer":
        status = "failed"
    else:
        status = "done"

    trace = snapshot.values.get("trace", [])
    latency = round(sum(s.get("duration_ms", 0.0) for s in trace), 2)
    cost = round(sum(s.get("cost_usd", 0.0) for s in trace), 6)

    return RunView(
        thread_id=thread_id,
        status=status,
        intent=snapshot.values.get("intent"),
        messages=_serialize(snapshot.values.get("messages", [])),
        citations=[Citation(**c) for c in snapshot.values.get("citations", [])],
        retrieval_score=snapshot.values.get("retrieval_score"),
        interrupt=interrupts[0].value if interrupts else None,
        trace=trace,
        latency_ms=latency,
        cost_usd=cost,
        next=list(snapshot.next),
    )
