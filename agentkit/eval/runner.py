"""Eval runner: run the golden dataset through the agent, score it, and gate releases.

Usage:
    python -m agentkit.eval.runner                    # run + gate against baseline.json
    python -m agentkit.eval.runner --update-baseline  # record current metrics as the baseline

Requires Postgres (DATABASE_URL). By default it uses the deterministic StubChat so it needs no
API key; set EVAL_LLM=real to evaluate against the real model.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

import agentkit.core.graph.nodes as nodes
from agentkit.core.graph import build_graph
from agentkit.core.graph.nodes import _NO_ANSWER_TEXT
from agentkit.eval.stub_llm import StubChat
from agentkit.retrieval.ingest import ingest_path
from agentkit.tools.calc import CalcError, safe_eval

_DIR = Path(__file__).parent
_CORPUS = _DIR / "corpus"
_DATASET = _DIR / "dataset.json"
_BASELINE = _DIR / "baseline.json"

# Map per-case check keys to reported metric names.
_METRICS = {
    "intent": "intent_accuracy",
    "retrieval": "retrieval_recall",
    "grounding": "citation_grounding",
    "no_answer": "no_answer_accuracy",
    "tool": "tool_use_correctness",
    "content": "answer_correctness",
}


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression."""
    try:
        return str(safe_eval(expression))
    except CalcError as exc:
        return f"error: {exc}"


@dataclass
class Observed:
    intent: str | None
    retrieved_sources: list[str]
    cited_sources: list[str]
    answer: str
    invoked_tools: list[str]
    no_answer: bool


@dataclass
class CaseResult:
    id: str
    checks: dict[str, bool] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(self.checks.values())


def observe(result: dict) -> Observed:
    messages = result.get("messages", [])
    answer = str(messages[-1].content) if messages else ""
    invoked = [
        tc["name"]
        for m in messages
        for tc in (getattr(m, "tool_calls", None) or [])
    ]
    return Observed(
        intent=result.get("intent"),
        retrieved_sources=[c["source"] for c in result.get("chunks", [])],
        cited_sources=[c["source"] for c in result.get("citations", [])],
        answer=answer,
        invoked_tools=invoked,
        no_answer=answer.strip() == _NO_ANSWER_TEXT,
    )


def score_case(case: dict, obs: Observed) -> CaseResult:
    checks: dict[str, bool] = {}

    if "expected_intent" in case:
        checks["intent"] = obs.intent == case["expected_intent"]

    if case["type"] in ("rag", "no_answer", "tool"):
        checks["no_answer"] = obs.no_answer == case.get("expect_no_answer", False)

    if case.get("expected_source") and not case.get("expect_no_answer"):
        checks["retrieval"] = case["expected_source"] in obs.retrieved_sources

    if case.get("must_cite"):
        checks["grounding"] = bool(obs.cited_sources) and all(
            s in obs.retrieved_sources for s in obs.cited_sources
        )

    if case.get("expected_tool"):
        checks["tool"] = case["expected_tool"] in obs.invoked_tools

    must = case.get("must_contain", [])
    must_not = case.get("must_not_contain", [])
    if must or must_not:
        a = obs.answer.lower()
        checks["content"] = all(s.lower() in a for s in must) and all(
            s.lower() not in a for s in must_not
        )

    return CaseResult(id=case["id"], checks=checks)


def aggregate(results: list[CaseResult]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, name in _METRICS.items():
        vals = [r.checks[key] for r in results if key in r.checks]
        if vals:
            metrics[name] = round(sum(vals) / len(vals), 4)
    metrics["pass_rate"] = round(sum(r.passed for r in results) / len(results), 4)
    return metrics


def gate(metrics: dict[str, float], baseline: dict[str, float], eps: float = 1e-6):
    regressions = [
        (k, base, metrics.get(k, 0.0))
        for k, base in baseline.items()
        if metrics.get(k, 0.0) < base - eps
    ]
    ok = not regressions and metrics.get("pass_rate", 0.0) >= 1.0 - eps
    return ok, regressions


def run_eval() -> tuple[list[CaseResult], dict[str, float]]:
    ingest_path(str(_CORPUS), reset=True)

    if os.getenv("EVAL_LLM") != "real":
        stub = StubChat()
        nodes.get_llm = lambda *a, **k: stub  # type: ignore[assignment]

    graph = build_graph(MemorySaver(), tools=[calculator])
    cases = json.loads(_DATASET.read_text())

    async def _invoke(case):
        return await graph.ainvoke(
            {"messages": [HumanMessage(case["question"])]},
            {"configurable": {"thread_id": case["id"]}},
        )

    results: list[CaseResult] = []
    for case in cases:
        out = asyncio.run(_invoke(case))  # ainvoke: tools nodes may be async (MCP)
        results.append(score_case(case, observe(out)))
    return results, aggregate(results)


def _print(results: list[CaseResult], metrics: dict[str, float]) -> None:
    print("\nper-case:")
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        detail = ", ".join(f"{k}={'ok' if v else 'X'}" for k, v in r.checks.items())
        print(f"  [{status}] {r.id:28s} {detail}")
    print("\nmetrics:")
    for k, v in sorted(metrics.items()):
        print(f"  {k:22s} {v:.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the agent eval harness")
    parser.add_argument("--update-baseline", action="store_true", help="save metrics as baseline")
    args = parser.parse_args()

    results, metrics = run_eval()
    _print(results, metrics)

    if args.update_baseline:
        _BASELINE.write_text(json.dumps(metrics, indent=2) + "\n")
        print(f"\nbaseline updated -> {_BASELINE}")
        return 0

    baseline = json.loads(_BASELINE.read_text()) if _BASELINE.exists() else {}
    ok, regressions = gate(metrics, baseline)
    if regressions:
        print("\nREGRESSIONS:")
        for k, base, cur in regressions:
            print(f"  {k}: {base:.4f} -> {cur:.4f}")
    if not ok and not regressions:
        print("\nGATE FAILED: not all cases passed (pass_rate < 1.0)")
    print("\nGATE:", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
