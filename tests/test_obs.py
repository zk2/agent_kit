"""Tests for observability: cost accounting and the tracing wrapper (offline)."""

from langchain_core.messages import AIMessage

from agentkit.obs.cost import cost_usd
from agentkit.obs.trace import traced


def test_cost_usd():
    # sonnet: $3/M in, $15/M out
    assert cost_usd("claude-sonnet-4-6", 1_000_000, 1_000_000) == 18.0
    assert cost_usd("unknown-model", 1000, 1000) == 0.0


def test_traced_records_decision_and_duration():
    fn = lambda state: {"intent": "question", "retrieval_score": 0.7}  # noqa: E731
    out = traced("classify", fn)({})

    assert out["intent"] == "question"  # original update preserved
    span = out["trace"][0]
    assert span["node"] == "classify"
    assert span["intent"] == "question"
    assert span["retrieval_score"] == 0.7
    assert span["duration_ms"] >= 0.0


def test_traced_records_tool_calls():
    msg = AIMessage(content="", tool_calls=[{"name": "calculator", "args": {}, "id": "c1"}])
    out = traced("plan", lambda s: {"messages": [msg]})({})
    assert out["trace"][0]["tool_calls"] == ["calculator"]


def test_traced_records_tokens_and_cost():
    msg = AIMessage(
        content="hi",
        usage_metadata={"input_tokens": 1_000_000, "output_tokens": 1_000_000, "total_tokens": 0},
    )
    span = traced("synthesize", lambda s: {"messages": [msg]})({})["trace"][0]
    assert span["input_tokens"] == 1_000_000
    assert span["cost_usd"] > 0.0
