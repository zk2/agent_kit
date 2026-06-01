"""Request/response models for the API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    # Continue an existing conversation by passing its thread_id; omit to start a new one.
    thread_id: str | None = None


class ResumeRequest(BaseModel):
    action: Literal["approve", "edit", "reject"] = "approve"
    # For "edit": the corrected tool_calls to run instead of the proposed ones.
    tool_calls: list[dict] | None = None


class Citation(BaseModel):
    marker: int
    source: str
    chunk_index: int


RunStatus = Literal["running", "awaiting_review", "done", "failed"]


class ChatResponse(BaseModel):
    thread_id: str
    status: RunStatus
    intent: str | None = None
    response: str = ""
    citations: list[Citation] = Field(default_factory=list)
    # Present when status == "awaiting_review": the payload the operator must act on.
    interrupt: dict[str, Any] | None = None
    latency_ms: float = 0.0
    cost_usd: float = 0.0


class MessageView(BaseModel):
    role: str
    content: str


class RunView(BaseModel):
    thread_id: str
    status: RunStatus
    intent: str | None = None
    messages: list[MessageView]
    citations: list[Citation] = Field(default_factory=list)
    retrieval_score: float | None = None
    interrupt: dict[str, Any] | None = None
    trace: list[dict[str, Any]] = Field(default_factory=list)
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    next: list[str]
