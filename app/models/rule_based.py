"""Deterministic assistant model: regex intent detection + AST pre-validation.

All user-facing copy lives here and only here, so swapping in a real LLM or
localizing to Spanish/Portuguese never touches the service layer.
"""

import asyncio
import re
from collections.abc import AsyncIterator

from app.domain.actions import DirectResponse, NextAction, ToolCall
from app.domain.messages import Message
from app.domain.tools import ToolResult
from app.tools.calculator import validate_expression

# Must match CalculatorTool.spec.name, the same way an LLM prompt names available tools.
CALCULATOR_TOOL_NAME = "calculator"

_EXPRESSION_CANDIDATE = re.compile(r"[\d\s.+\-*/()]+")
_CALC_INTENT = re.compile(
    r"\b(calc|calculate|calculation|compute|evaluate|what\s+is|what's|whats|how\s+much\s+is)\b",
    re.IGNORECASE,
)
_GREETING = re.compile(r"\b(hi|hello|hey|hola|oi|olá)\b", re.IGNORECASE)

GREETING_REPLY = "Hi! I can chat and do arithmetic. Try asking: what is (10 - 4) * 2?"
FALLBACK_REPLY = (
    "I'm a simple assistant that is great at arithmetic. Ask me something like: what is 2 + 2?"
)


class RuleBasedAssistantModel:
    def __init__(self, chunk_delay_seconds: float = 0.02) -> None:
        # Pacing makes streaming visible in the UI; tests inject 0.
        self.chunk_delay_seconds = chunk_delay_seconds

    async def plan_next_action(self, messages: list[Message]) -> NextAction:
        candidate = unvalidated_calculator_intent(messages[-1].content)
        if candidate is None or validate_expression(candidate) is not None:
            return DirectResponse()
        return ToolCall(tool_name=CALCULATOR_TOOL_NAME, tool_input=candidate)

    async def stream_response(
        self, messages: list[Message], tool_result: ToolResult | None = None
    ) -> AsyncIterator[str]:
        text = self._compose(messages[-1].content, tool_result)
        for chunk in re.findall(r"\S+\s*", text):
            if self.chunk_delay_seconds > 0:
                await asyncio.sleep(self.chunk_delay_seconds)
            yield chunk

    def _compose(self, user_text: str, tool_result: ToolResult | None) -> str:
        if tool_result is not None:
            if not tool_result.ok:
                return f"I couldn't compute that: {tool_result.error}."
            expression = _extract_expression(user_text)
            if expression is None:
                return f"The result is {tool_result.content}."
            return f"{expression} = {tool_result.content}"
        if _GREETING.search(user_text):
            return GREETING_REPLY
        return FALLBACK_REPLY


def _arithmetic_candidate(text: str) -> str | None:
    """Longest arithmetic-looking substring with at least a digit and an operator."""
    candidates = [candidate.strip() for candidate in _EXPRESSION_CANDIDATE.findall(text)]
    best = max(candidates, key=len, default="")
    if not any(ch.isdigit() for ch in best) or not any(op in best for op in "+-*/"):
        return None
    return best


def unvalidated_calculator_intent(text: str) -> str | None:
    """The message's arithmetic candidate when it reads as a calculator request.

    Deliberately unvalidated: this planner pre-validates the candidate, while
    the ollama adapter's backstop hands it straight to the tool, which refuses
    bad expressions visibly.
    """
    best = _arithmetic_candidate(text)
    if best is None:
        return None
    whole_message = text.strip().rstrip("?.!").strip()
    if _CALC_INTENT.search(text) or whole_message == best:
        return best
    return None


def _extract_expression(text: str) -> str | None:
    """Longest arithmetic-looking substring, only if the calculator would accept it."""
    best = _arithmetic_candidate(text)
    if best is None or validate_expression(best) is not None:
        return None
    return best
