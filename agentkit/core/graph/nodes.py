r"""Graph nodes.

Flow (Stage 3):

    classify -> chitchat -------------------------------------------> respond -> END
            \-> question -> retrieve -> (confident?) -> no -> no_answer -> validate -> END
                                           \- yes -> plan <-> tools -> synthesize -> validate -> END

The plan<->tools loop only exists when tools are provided to `build_graph`; otherwise the question
path goes straight retrieve -> synthesize (Stage 2 behaviour). Nodes read/write the typed `State`
and return only the keys they change. The LLM and retriever are reached through cached factories
so tests can swap in fakes.
"""

from __future__ import annotations

import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt

from agentkit.config import settings
from agentkit.core.llm import get_llm
from agentkit.core.state import State
from agentkit.retrieval.store import chunk_to_dict, get_retriever

_CITATION = re.compile(r"\[(\d+)\]")

_CLASSIFY_PROMPT = SystemMessage(
    "Classify the user's latest message into exactly one lowercase word: "
    "`question` (asks for information), `chitchat` (greeting/smalltalk), or `other`. "
    "Reply with only that single word."
)

_RESPOND_PROMPT = SystemMessage(
    "You are agentkit, a concise, friendly assistant. Reply briefly to the user's smalltalk."
)

_PLAN_PROMPT = SystemMessage(
    "You decide whether external tools are needed to answer the question. Call a tool only when "
    "it improves correctness (arithmetic, structured lookups, fresh data). If the retrieved "
    "context already suffices, respond without calling any tool - your reply ends the loop."
)

_SYNTH_PROMPT = SystemMessage(
    "You answer strictly from the provided context and tool results. Cite the context sources "
    "you use with bracketed numbers like [1], [2]. If they don't contain the answer, say you "
    "don't know rather than guessing. Be concise."
)

_NO_ANSWER_TEXT = (
    "I don't have enough information in the knowledge base to answer that confidently."
)


def _text(message) -> str:
    """A message's content as a plain string (LangChain content can be str or a list of blocks)."""
    content = message.content
    return content if isinstance(content, str) else str(content)


def _last_user_text(state: State) -> str:
    for m in reversed(state.get("messages", [])):
        if getattr(m, "type", None) == "human":
            return _text(m)
    return ""


def _format_context(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"[{i + 1}] (source: {c['source']}#{c['chunk_index']}) {c['content']}"
        for i, c in enumerate(chunks)
    )


def _tool_results(state: State) -> list[str]:
    return [str(m.content) for m in state.get("messages", []) if getattr(m, "type", None) == "tool"]


def _extract_citations(answer: str, chunks: list[dict]) -> list[dict]:
    """Map [n] markers in the answer back to the chunk sources they reference."""
    seen: dict[int, dict] = {}
    for marker in _CITATION.findall(answer):
        i = int(marker) - 1
        if 0 <= i < len(chunks) and i not in seen:
            c = chunks[i]
            seen[i] = {"marker": i + 1, "source": c["source"], "chunk_index": c["chunk_index"]}
    return list(seen.values())


# --- nodes -------------------------------------------------------------------


def classify(state: State) -> dict:
    llm = get_llm()
    result = llm.invoke([_CLASSIFY_PROMPT, *state["messages"]])
    text = _text(result).strip()
    intent = text.lower().split()[0] if text else "other"
    return {"intent": intent}


def respond(state: State) -> dict:
    llm = get_llm()
    result = llm.invoke([_RESPOND_PROMPT, *state["messages"]])
    return {"messages": [result]}


def retrieve(state: State) -> dict:
    query = _last_user_text(state)
    chunks = [chunk_to_dict(c) for c in get_retriever().search(query, k=settings.retrieval_top_k)]
    score = max((c["vscore"] for c in chunks), default=0.0)
    return {"query": query, "chunks": chunks, "retrieval_score": score}


def make_plan(tools: list):
    """Plan node: let the model decide whether to call a tool, given the retrieved context."""

    def plan(state: State) -> dict:
        llm = get_llm()
        if tools:
            try:
                llm = llm.bind_tools(tools)
            except NotImplementedError:
                pass  # model without tool support -> degrade to no tools
        context = _format_context(state.get("chunks", []))
        system = SystemMessage(
            _text(_PLAN_PROMPT) + (f"\n\nContext:\n{context}" if context else "")
        )
        ai = llm.invoke([system, *state["messages"]])
        return {"messages": [ai]}

    return plan


def make_tools_node(tools: list):
    """Execute requested tool calls and count the iteration (anti-looping).

    Async so that async-only tools (e.g. MCP tools) execute correctly; sync tools are handled by
    ToolNode transparently. The graph is driven with ainvoke everywhere tools are enabled.
    """
    # handle_tool_errors=True: a raised tool returns a ToolMessage with the error instead of
    # crashing the run, so the model can retry or stop gracefully.
    tool_node = ToolNode(tools, handle_tool_errors=True)

    async def run_tools(state: State) -> dict:
        out = await tool_node.ainvoke(state)
        out["tool_iterations"] = state.get("tool_iterations", 0) + 1
        return out

    return run_tools


def _review_tools() -> set[str]:
    return {t.strip() for t in settings.hitl_review_tools.split(",") if t.strip()}


def _pending_tool_calls(state: State) -> list[dict]:
    last = state["messages"][-1]
    return list(getattr(last, "tool_calls", None) or [])


def review(state: State) -> dict:
    """Human-in-the-loop gate before side-effecting tools.

    Pauses the run via interrupt(); the checkpointer persists state until an operator resumes
    with a decision. On resume, interrupt() returns that decision instead of pausing again.
    """
    pending = _pending_tool_calls(state)
    decision = interrupt(
        {
            "type": "tool_approval",
            "pending_tool_calls": pending,
            "instructions": "resume with {'action': 'approve' | 'reject' | 'edit', "
            "'tool_calls': [...] (for edit)}",
        }
    )
    action = (decision or {}).get("action", "approve")

    if action == "reject":
        return {"review_decision": "reject"}

    if action == "edit":
        last = state["messages"][-1]
        edited = AIMessage(
            content=_text(last),
            tool_calls=decision.get("tool_calls", getattr(last, "tool_calls", [])),
            id=last.id,  # same id -> add_messages replaces the original message
        )
        return {"review_decision": "approve", "messages": [edited]}

    return {"review_decision": "approve"}


def synthesize(state: State) -> dict:
    chunks = state.get("chunks", [])
    context = _format_context(chunks)
    tool_block = "\n".join(f"- {t}" for t in _tool_results(state)) or "(none)"
    user = HumanMessage(
        f"Context:\n{context}\n\nTool results:\n{tool_block}\n\n"
        f"Question: {state.get('query', '')}\n\n"
        "Answer using only the context and tool results, and cite sources with [n]."
    )
    result = get_llm().invoke([_SYNTH_PROMPT, user])
    citations = _extract_citations(_text(result), chunks)
    return {"messages": [result], "citations": citations}


def no_answer(state: State) -> dict:
    return {"messages": [AIMessage(_NO_ANSWER_TEXT)], "citations": []}


def validate(state: State) -> dict:
    messages = state.get("messages", [])
    answer = messages[-1].content if messages else ""
    if not str(answer).strip():
        return {"error": "empty answer"}
    # Grounding check only applies when we actually answered from context (synthesize path).
    answered_from_context = state.get("retrieval_score", 0.0) >= settings.retrieval_min_score
    if answered_from_context and state.get("chunks") and not state.get("citations"):
        return {"error": "ungrounded: answer cites no sources"}
    return {}


# --- routing -----------------------------------------------------------------


def route_intent(state: State) -> str:
    return "respond" if state.get("intent") == "chitchat" else "retrieve"


def route_after_retrieve(state: State) -> str:
    if state.get("retrieval_score", 0.0) < settings.retrieval_min_score:
        return "no_answer"
    return "answer"


def route_after_plan(state: State) -> str:
    calls = getattr(state["messages"][-1], "tool_calls", None) or []
    if calls and state.get("tool_iterations", 0) < settings.max_tool_iterations:
        if any(c["name"] in _review_tools() for c in calls):
            return "review"  # risky tool -> human approval first
        return "tools"
    return "synthesize"


def route_after_review(state: State) -> str:
    return "synthesize" if state.get("review_decision") == "reject" else "tools"
