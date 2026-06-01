"""Graph assembly. `build_graph` is the single entry point for the API and for tests."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agentkit.core.graph.nodes import (
    classify,
    make_plan,
    make_tools_node,
    no_answer,
    respond,
    retrieve,
    review,
    route_after_plan,
    route_after_retrieve,
    route_after_review,
    route_intent,
    synthesize,
    validate,
)
from agentkit.core.state import State
from agentkit.obs.trace import traced

__all__ = ["build_graph"]


def build_graph(checkpointer=None, tools: list | None = None):
    """Compile the agent graph.

    `checkpointer` is injected so the API can use Postgres while tests use an in-memory saver.
    `tools` enables the plan<->tools loop; without tools the question path is retrieve ->
    synthesize.

        classify -> (chitchat) respond -> END
                 -> retrieve -> (confident) -> [plan <-> tools ->] synthesize -> validate -> END
                            -> (else)        no_answer -> validate -> END
    """
    tools = tools or []

    builder = StateGraph(State)
    builder.add_node("classify", traced("classify", classify))
    builder.add_node("respond", traced("respond", respond))
    builder.add_node("retrieve", traced("retrieve", retrieve))
    builder.add_node("synthesize", traced("synthesize", synthesize))
    builder.add_node("no_answer", traced("no_answer", no_answer))
    builder.add_node("validate", traced("validate", validate))

    builder.add_edge(START, "classify")
    builder.add_conditional_edges(
        "classify", route_intent, {"respond": "respond", "retrieve": "retrieve"}
    )
    builder.add_edge("respond", END)

    # Confident retrieval enters the tool loop when tools exist, else goes straight to synthesize.
    confident_target = "plan" if tools else "synthesize"
    builder.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {"answer": confident_target, "no_answer": "no_answer"},
    )

    if tools:
        builder.add_node("plan", traced("plan", make_plan(tools)))
        builder.add_node("review", traced("review", review))
        builder.add_node("tools", traced("tools", make_tools_node(tools)))
        builder.add_conditional_edges(
            "plan",
            route_after_plan,
            {"review": "review", "tools": "tools", "synthesize": "synthesize"},
        )
        builder.add_conditional_edges(
            "review", route_after_review, {"tools": "tools", "synthesize": "synthesize"}
        )
        builder.add_edge("tools", "plan")

    builder.add_edge("synthesize", "validate")
    builder.add_edge("no_answer", "validate")
    builder.add_edge("validate", END)

    return builder.compile(checkpointer=checkpointer)
