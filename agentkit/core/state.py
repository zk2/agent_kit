"""Typed graph state.

Short-lived execution state lives here and is checkpointed by LangGraph. Persistent business
state belongs in the application DB (out of scope for Stage 1). Fields are added per roadmap
stage; for now we keep the conversation, the classified intent, and an error slot.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class State(TypedDict, total=False):
    # Conversation history. `add_messages` appends and de-dupes by id across turns.
    messages: Annotated[list[AnyMessage], add_messages]

    # Output of the classify node: e.g. "question" | "chitchat" | "other".
    intent: str

    # Retrieval (Stage 2).
    query: str  # text used for retrieval (the user's question)
    chunks: list[dict]  # retrieved chunks, serialized for checkpointing
    retrieval_score: float  # top cosine similarity, drives the no-answer decision
    citations: list[dict]  # sources actually cited in the answer

    # Tools (Stage 3).
    tool_iterations: int  # number of plan<->tools loops taken, capped to avoid looping

    # Human-in-the-loop (Stage 4).
    review_decision: str  # last operator decision: "approve" | "reject"

    # Observability (Stage 6): one span per node execution, accumulated across the run.
    trace: Annotated[list[dict], operator.add]

    # Populated when a node fails or flags a quality issue in a recoverable way.
    error: str
