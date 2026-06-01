"""Application settings, loaded from environment / .env."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-6"

    database_url: str = "postgresql://agentkit:agentkit@localhost:5432/agentkit"

    # LLM defaults
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1024

    # Retrieval
    embedding_provider: str = "hashing"  # "hashing" (offline) | "fastembed" (semantic)
    embedding_dim: int = 384
    retrieval_top_k: int = 5
    retrieval_min_score: float = 0.25  # min top cosine similarity before no-answer

    # Tools
    max_tool_iterations: int = 3  # cap on plan<->tools loops (anti-looping)

    # Human-in-the-loop: comma-separated tool names that require approval before execution.
    hitl_review_tools: str = "save_note"

    # Orchestration engine: "langgraph" | "claude_agent_sdk"
    orchestrator: str = "langgraph"

    # LLM backend for the running app: "real" (Anthropic) | "stub" (deterministic, no API key).
    # "stub" lets you exercise the whole HTTP API offline for manual testing/demos.
    app_llm: str = "real"


settings = Settings()
