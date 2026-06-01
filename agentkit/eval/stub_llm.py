"""A deterministic stand-in for the chat model, used by the eval harness.

It dispatches on each node's system prompt and produces rule-based output:
  - classify : greeting -> "chitchat", otherwise "question"
  - plan     : emit a calculator tool call when the question contains an arithmetic expression
               and the calculator tool is bound; stop once a tool result is present
  - synthesize: echo the retrieved context (which carries [n] citation markers and the source
               text), so answer quality tracks REAL retrieval - break retrieval and the metrics drop
  - respond  : a fixed chitchat reply

This keeps evals hermetic (no API key) while still exercising the real graph, retrieval and
routing. Swap back to the real model by not overriding `get_llm` (EVAL_LLM=real).
"""

from __future__ import annotations

import re

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

_ARITH = re.compile(r"\d+(?:\s*[-+*/]\s*\d+)+")
_GREETINGS = ("hello", "hi ", "hey", "thanks", "thank you", "good morning")


def _last_human(messages) -> str:
    for m in reversed(messages):
        if getattr(m, "type", None) == "human":
            return m.content if isinstance(m.content, str) else str(m.content)
    return ""


class StubChat(BaseChatModel):
    bound_tool_names: list[str] = []

    @property
    def _llm_type(self) -> str:
        return "stub"

    def bind_tools(self, tools, **kwargs):
        return self.model_copy(update={"bound_tool_names": [t.name for t in tools]})

    # --- per-node behaviours ---

    def _classify(self, messages) -> AIMessage:
        text = _last_human(messages).lower()
        greet = any(g in text for g in _GREETINGS)
        return AIMessage(content="chitchat" if greet and "?" not in text else "question")

    def _plan(self, messages) -> AIMessage:
        if any(getattr(m, "type", None) == "tool" for m in messages):
            return AIMessage(content="")  # tool already ran -> stop the loop
        user = _last_human(messages)
        low = user.lower()
        if "save_note" in self.bound_tool_names and ("save" in low or "note" in low):
            return AIMessage(
                content="",
                tool_calls=[{"name": "save_note", "args": {"text": user}, "id": "note-1"}],
            )
        match = _ARITH.search(user)
        if match and "calculator" in self.bound_tool_names:
            return AIMessage(
                content="",
                tool_calls=[
                    {"name": "calculator", "args": {"expression": match.group(0).strip()},
                     "id": "calc-1"}
                ],
            )
        return AIMessage(content="")

    def _synthesize(self, messages) -> AIMessage:
        user = _last_human(messages)
        context = ""
        if "Context:" in user:
            context = user.split("Context:", 1)[1].split("Question:")[0].strip()
        return AIMessage(content=f"Based on the retrieved material: {context} [1]")

    def _respond(self) -> AIMessage:
        return AIMessage(content="Hi! How can I help you today?")

    # --- BaseChatModel hook ---

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        first = messages[0] if messages else None
        system = first.content if getattr(first, "type", None) == "system" else ""
        if "Classify the user" in system:
            msg = self._classify(messages)
        elif "decide whether external tools" in system:
            msg = self._plan(messages)
        elif "answer strictly from the provided context" in system:
            msg = self._synthesize(messages)
        elif "friendly assistant" in system:
            msg = self._respond()
        else:
            msg = AIMessage(content="ok")
        return ChatResult(generations=[ChatGeneration(message=msg)])
