"""Optional local LLM adapter over Ollama (recommended model: qwen2.5:7b).

Demo realism only: non-deterministic, never used by tests or CI. The
deterministic RuleBasedAssistantModel stays the default everywhere. The LLM
only classifies the message into a calculator call, a weather_lookup call, or
a direct reply; every proposed plan is validated here (calculator expressions
with the same AST whitelist the tool uses) and anything invalid falls back to
a direct reply. The tools do all the actual work: the LLM never computes.
"""

import json
import re
from collections.abc import AsyncIterator

import httpx

from app.domain.actions import DirectResponse, NextAction, ToolCall
from app.domain.messages import Message
from app.domain.tools import ToolResult
from app.tools.calculator import validate_expression

# Must match the registered tools' spec.name.
CALCULATOR_TOOL_NAME = "calculator"
WEATHER_TOOL_NAME = "weather_lookup"
MAX_CITY_LENGTH = 80
# Enough context to resolve follow-ups ("and in Lisbon?") without unbounded prompts.
PLANNER_CONTEXT_MESSAGES = 6

_PLANNER_PROMPT = (
    "You classify the user's LAST message and pick one action. "
    "Earlier messages are only context for resolving references. "
    "Reply with JSON only, no other text, in exactly one of these shapes:\n"
    '- Arithmetic to compute: {"action": "calculator", "expression": "<expression only>"}\n'
    '- Current weather somewhere: {"action": "weather_lookup", "location": "<place>"}\n'
    '- Anything else: {"action": "direct"}\n'
    "For calculator, copy only the math expression (e.g. 15 * 23); never compute it yourself."
)

_RESPONDER_PROMPT = (
    "You are a friendly, concise assistant. Always answer in the language the user wrote in."
)


class OllamaAssistantModel:
    """AssistantModel adapter speaking Ollama's /api/chat HTTP API."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        model_name: str = "qwen2.5:7b",
        request_timeout_seconds: float = 120.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model_name = model_name
        # Long read timeout: a 7B model can pause noticeably between tokens on first load.
        self._timeout = httpx.Timeout(request_timeout_seconds, connect=5.0)
        self._transport = transport  # injectable for offline tests

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=self._timeout, transport=self._transport)

    async def plan_next_action(self, messages: list[Message]) -> NextAction:
        try:
            async with self._client() as client:
                response = await client.post(
                    f"{self._base_url}/api/chat",
                    json={
                        "model": self._model_name,
                        "messages": self._planner_messages(messages),
                        "stream": False,
                        "format": "json",
                        "options": {"temperature": 0},
                    },
                )
                response.raise_for_status()
                raw_plan = response.json()["message"]["content"]
        except Exception:
            # Unreachable or misbehaving server must never block the reply.
            return DirectResponse()
        return parse_plan(raw_plan)

    async def stream_response(
        self, messages: list[Message], tool_result: ToolResult | None = None
    ) -> AsyncIterator[str]:
        payload = {
            "model": self._model_name,
            "messages": self._responder_messages(messages, tool_result),
            "stream": True,
        }
        emitted = False
        try:
            async with (
                self._client() as client,
                client.stream("POST", f"{self._base_url}/api/chat", json=payload) as response,
            ):
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    chunk = json.loads(line).get("message", {}).get("content", "")
                    if chunk:
                        emitted = True
                        yield chunk
        except httpx.HTTPError:
            if emitted:
                # Mid-reply failure surfaces through the in-band error event.
                raise
            yield (
                f"The local model at {self._base_url} is not responding. "
                f"Start it with: ollama run {self._model_name}"
            )

    def _planner_messages(self, messages: list[Message]) -> list[dict[str, str]]:
        chat = [{"role": "system", "content": _PLANNER_PROMPT}]
        chat.extend(
            {"role": m.role, "content": m.content}
            for m in messages[-PLANNER_CONTEXT_MESSAGES:]
        )
        return chat

    def _responder_messages(
        self, messages: list[Message], tool_result: ToolResult | None
    ) -> list[dict[str, str]]:
        chat = [{"role": "system", "content": _RESPONDER_PROMPT}]
        chat.extend({"role": m.role, "content": m.content} for m in messages)
        if tool_result is not None:
            if tool_result.ok:
                note = f"Tool result: {tool_result.content}. Answer using it."
            else:
                note = f"The tool failed ({tool_result.error}). Briefly say so."
            chat.append({"role": "system", "content": note})
        return chat


def parse_plan(raw_plan: str) -> NextAction:
    """Validate the LLM's tool plan; anything unexpected becomes a direct response.

    The LLM only proposes. For calculator, the extracted expression is validated
    with the calculator's own AST whitelist before we emit a ToolCall, so the
    tool stays the only thing that actually computes.
    """
    match = re.search(r"\{.*\}", raw_plan, re.S)
    if match is None:
        return DirectResponse()
    try:
        plan = json.loads(match.group())
    except json.JSONDecodeError:
        return DirectResponse()
    if not isinstance(plan, dict):
        return DirectResponse()
    if plan.get("action") == CALCULATOR_TOOL_NAME:
        return _calculator_call(plan.get("expression"))
    if plan.get("action") == WEATHER_TOOL_NAME:
        # Accept "location" (current shape) or "city" (older shape).
        return _weather_call(plan.get("location") or plan.get("city"))
    return DirectResponse()


def _calculator_call(expression: object) -> NextAction:
    if not isinstance(expression, str):
        return DirectResponse()
    expression = expression.strip()
    # Same validator the deterministic planner uses; the tool still recomputes.
    if validate_expression(expression) is not None:
        return DirectResponse()
    return ToolCall(tool_name=CALCULATOR_TOOL_NAME, tool_input=expression)


def _weather_call(location: object) -> NextAction:
    if not isinstance(location, str):
        return DirectResponse()
    location = location.strip()
    if not location or len(location) > MAX_CITY_LENGTH:
        return DirectResponse()
    return ToolCall(tool_name=WEATHER_TOOL_NAME, tool_input=location)
