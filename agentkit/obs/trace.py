"""Decision tracing.

`traced` wraps a graph node so every execution records a span - node name, duration, the
decision it made (intent, retrieval score, tool calls, review decision, citations, error) and
token cost. Spans are appended to `state["trace"]` (reduced with +), so the trace is checkpointed
and can be inspected end-to-end via the API. Each span is also emitted as a structured log line.

If LangSmith env vars (LANGCHAIN_TRACING_V2) are set, LangChain also exports full traces there;
this layer adds an in-band, dependency-free trace that travels with the run state.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable
from typing import Any, TypeVar

from agentkit.config import settings
from agentkit.obs.cost import cost_usd

logger = logging.getLogger("agentkit.obs")

# Preserve the node's state type through the wrapper so LangGraph sees the right node signature.
StateT = TypeVar("StateT")

_DECISION_KEYS = ("intent", "retrieval_score", "review_decision", "error")


def _token_usage(messages: list) -> tuple[int, int]:
    inp = out = 0
    for m in messages:
        usage = getattr(m, "usage_metadata", None) or {}
        inp += usage.get("input_tokens", 0)
        out += usage.get("output_tokens", 0)
    return inp, out


def _make_span(name: str, duration_ms: float, update: dict) -> dict:
    span: dict = {"node": name, "duration_ms": duration_ms}

    for key in _DECISION_KEYS:
        if key in update:
            span[key] = update[key]

    messages = update.get("messages", []) or []
    tool_calls = [tc["name"] for m in messages for tc in (getattr(m, "tool_calls", None) or [])]
    if tool_calls:
        span["tool_calls"] = tool_calls
    if "citations" in update:
        span["citations"] = len(update["citations"])

    inp, out = _token_usage(messages)
    if inp or out:
        span["input_tokens"] = inp
        span["output_tokens"] = out
        span["cost_usd"] = cost_usd(settings.anthropic_model, inp, out)

    return span


def _finish(name: str, start: float, update: dict) -> dict:
    update = update or {}
    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    span = _make_span(name, duration_ms, update)
    logger.info("span %s", json.dumps(span, default=str))
    return {**update, "trace": [span]}


def traced(name: str, fn: Callable[[StateT], Any]):
    """Wrap a node (sync or async) to record a span, preserving its state type.

    The wrapper keeps a `state: StateT` parameter so LangGraph's add_node sees a valid node
    signature (it expects a parameter named `state`). GraphInterrupt (HITL) propagates untimed:
    the span is recorded only once the node resolves.
    """
    if asyncio.iscoroutinefunction(fn):

        async def awrapper(state: StateT) -> dict:
            start = time.perf_counter()
            update = await fn(state)
            return _finish(name, start, update)

        awrapper.__name__ = name
        return awrapper

    def wrapper(state: StateT) -> dict:
        start = time.perf_counter()
        update = fn(state)
        return _finish(name, start, update)

    wrapper.__name__ = name
    return wrapper
