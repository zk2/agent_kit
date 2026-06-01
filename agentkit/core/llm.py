"""Thin wrapper around the Anthropic chat model.

Keeping construction behind a cached factory lets the rest of the code depend on a single
place for model config, retries and cost knobs - and lets tests swap in a fake model.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_anthropic import ChatAnthropic

from agentkit.config import settings


@lru_cache(maxsize=4)
def get_llm(model: str | None = None) -> ChatAnthropic:
    return ChatAnthropic(  # type: ignore[call-arg]  # aliased kwargs verified at runtime
        model_name=model or settings.anthropic_model,
        api_key=settings.anthropic_api_key,
        temperature=settings.llm_temperature,
        max_tokens_to_sample=settings.llm_max_tokens,
        max_retries=3,
        timeout=60,
    )
