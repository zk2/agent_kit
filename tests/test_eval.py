"""Offline tests for the eval harness: stub model + scoring/aggregation/gating (no DB)."""

from langchain_core.messages import HumanMessage

from agentkit.core.graph.nodes import _CLASSIFY_PROMPT, _PLAN_PROMPT, _SYNTH_PROMPT
from agentkit.eval.runner import CaseResult, Observed, aggregate, calculator, gate, score_case
from agentkit.eval.stub_llm import StubChat

# --- stub model --------------------------------------------------------------


def _classify(stub, text):
    return stub.invoke([_CLASSIFY_PROMPT, HumanMessage(text)]).content


def test_stub_classify():
    stub = StubChat()
    assert _classify(stub, "Hello there, thanks!") == "chitchat"
    assert _classify(stub, "What is the port?") == "question"


def test_stub_plan_calls_calculator_on_arithmetic():
    bound = StubChat().bind_tools([calculator])
    ai = bound.invoke([_PLAN_PROMPT, HumanMessage("what is 12 * 12?")])
    assert ai.tool_calls and ai.tool_calls[0]["name"] == "calculator"
    assert ai.tool_calls[0]["args"]["expression"] == "12 * 12"


def test_stub_plan_no_tool_without_arithmetic():
    bound = StubChat().bind_tools([calculator])
    ai = bound.invoke([_PLAN_PROMPT, HumanMessage("how does retrieval work?")])
    assert not ai.tool_calls


def test_stub_synth_echoes_context_with_citation():
    user = HumanMessage(
        "Context:\n[1] (source: deployment.md#0) listens on port 8000\n\n"
        "Tool results:\n(none)\n\nQuestion: which port?\n\nAnswer ..."
    )
    out = StubChat().invoke([_SYNTH_PROMPT, user]).content
    assert "8000" in out and "[1]" in out


# --- scoring -----------------------------------------------------------------


def _obs(
    *,
    intent: str | None = "question",
    retrieved_sources: list[str] | None = None,
    cited_sources: list[str] | None = None,
    answer: str = "listens on port 8000 [1]",
    invoked_tools: list[str] | None = None,
    no_answer: bool = False,
) -> Observed:
    return Observed(
        intent=intent,
        retrieved_sources=retrieved_sources if retrieved_sources is not None else ["deployment.md"],
        cited_sources=cited_sources if cited_sources is not None else ["deployment.md"],
        answer=answer,
        invoked_tools=invoked_tools if invoked_tools is not None else [],
        no_answer=no_answer,
    )


def test_score_case_all_pass():
    case = {
        "id": "c", "type": "rag", "expected_intent": "question",
        "expected_source": "deployment.md", "must_cite": True,
        "must_contain": ["8000"], "must_not_contain": ["mysql"],
    }
    res = score_case(case, _obs())
    assert res.passed
    assert res.checks == {"intent": True, "no_answer": True, "retrieval": True,
                          "grounding": True, "content": True}


def test_score_case_detects_bad_grounding():
    case = {"id": "c", "type": "rag", "must_cite": True}
    # cites a source that was not retrieved -> ungrounded
    res = score_case(case, _obs(cited_sources=["other.md"], retrieved_sources=["deployment.md"]))
    assert res.checks["grounding"] is False
    assert not res.passed


def test_aggregate_and_gate():
    results = [
        CaseResult("a", {"retrieval": True, "content": True}),
        CaseResult("b", {"retrieval": False, "content": True}),
    ]
    metrics = aggregate(results)
    assert metrics["retrieval_recall"] == 0.5
    assert metrics["answer_correctness"] == 1.0
    assert metrics["pass_rate"] == 0.5

    # gate: pass_rate < 1.0 fails even with no baseline
    ok, _ = gate(metrics, {})
    assert ok is False

    # regression: metric below baseline
    ok, regr = gate({"retrieval_recall": 0.5, "pass_rate": 1.0}, {"retrieval_recall": 0.9})
    assert ok is False and regr and regr[0][0] == "retrieval_recall"

    # clean pass
    ok, regr = gate({"retrieval_recall": 0.9, "pass_rate": 1.0}, {"retrieval_recall": 0.9})
    assert ok is True and not regr
