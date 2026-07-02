"""Optional local LLM adapter over Ollama (recommended model: qwen2.5:7b).

Demo realism only: non-deterministic, never used by tests or CI. The
deterministic RuleBasedAssistantModel stays the default everywhere. The LLM
only classifies the message into a calculator call, a weather_lookup call, or
a direct reply. Three nets keep it from improvising math: every calculator
plan is executed against the tool, which re-validates and refuses bad
expressions with a visible reason; unusable planner output (malformed or
truncated JSON) falls through a deterministic backstop that still routes
plainly-arithmetic messages to the calculator; and the responder prompt
forbids computing without a tool result. The tools do all the actual work:
the LLM never computes.
"""

import json
import re
from collections.abc import AsyncIterator, Sequence

import httpx

from app.domain.actions import DirectResponse, NextAction, ToolCall
from app.domain.messages import Message
from app.domain.tools import ToolResult, ToolSpec
from app.models.rule_based import unvalidated_calculator_intent
from app.tools.calculator import CalculatorTool
from app.tools.weather import WeatherLookupTool

# Must match the registered tools' spec.name.
CALCULATOR_TOOL_NAME = "calculator"
WEATHER_TOOL_NAME = "weather_lookup"
MAX_CITY_LENGTH = 80
# Enough context to resolve follow-ups ("and in Lisbon?") without unbounded prompts.
PLANNER_CONTEXT_MESSAGES = 6

# The JSON shape per plannable tool; the adapter can only validate these two,
# so specs with any other name never reach the menu (fail closed).
_PLAN_SHAPES = {
    CALCULATOR_TOOL_NAME: '{"action": "calculator", "expression": "<expression only>"}',
    WEATHER_TOOL_NAME: '{"action": "weather_lookup", "location": "<place>"}',
}

# Standalone construction plans over the real tools' own specs; create_app
# passes the actually registered specs instead, keeping the registry the truth.
_DEFAULT_TOOL_SPECS = (CalculatorTool.spec, WeatherLookupTool.spec)


def _planner_prompt(tool_specs: Sequence[ToolSpec]) -> str:
    lines = [
        f"- {spec.description.rstrip('.')}: {_PLAN_SHAPES[spec.name]}"
        for spec in tool_specs
        if spec.name in _PLAN_SHAPES
    ]
    lines.append('- Anything else: {"action": "direct"}')
    prompt = (
        "You classify the user's LAST message and pick one action. "
        "Earlier messages are only context for resolving references. "
        "Reply with JSON only, no other text, in exactly one of these shapes:\n"
        + "\n".join(lines)
    )
    if any(spec.name == CALCULATOR_TOOL_NAME for spec in tool_specs):
        prompt += (
            "\nFor calculator, copy only the math expression (e.g. 15 * 23); "
            "never compute it yourself."
        )
    return prompt

_RESPONDER_PROMPT = (
    "You are a friendly, concise assistant. Always answer in the language the user wrote in. "
    "Never do arithmetic yourself: without a calculator tool result, say the calculation "
    "could not be run."
)


class OllamaAssistantModel:
    """AssistantModel adapter speaking Ollama's /api/chat HTTP API."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        model_name: str = "qwen2.5:7b",
        request_timeout_seconds: float = 120.0,
        transport: httpx.AsyncBaseTransport | None = None,
        tool_specs: Sequence[ToolSpec] | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model_name = model_name
        # Long read timeout: a 7B model can pause noticeably between tokens on first load.
        self._timeout = httpx.Timeout(request_timeout_seconds, connect=5.0)
        self._transport = transport  # injectable for offline tests
        self._planner_prompt = _planner_prompt(
            tool_specs if tool_specs is not None else _DEFAULT_TOOL_SPECS
        )

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
            raw_plan = ""
        action = parse_plan(raw_plan)
        if isinstance(action, DirectResponse):
            return _calculator_backstop(messages[-1].content)
        return action

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
        chat = [{"role": "system", "content": self._planner_prompt}]
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
    """Check the LLM's plan shape; malformed plans become a direct response.

    The LLM only proposes. A well-formed calculator plan is emitted as a
    ToolCall even when the expression looks doomed: the tool re-validates and
    refuses it, and that failed ToolResult reaches the responder, which then
    explains the refusal instead of improvising math.
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


def _calculator_backstop(text: str) -> NextAction:
    """Deterministic net under the LLM planner.

    Truncated or malformed planner JSON must not downgrade arithmetic into the
    LLM improvising math: when the message plainly reads as a calculation (the
    same gate the rule-based model uses), route the raw candidate to the tool,
    which computes it or refuses it visibly.
    """
    candidate = unvalidated_calculator_intent(text)
    if candidate is None:
        return DirectResponse()
    return ToolCall(tool_name=CALCULATOR_TOOL_NAME, tool_input=candidate)


def _calculator_call(expression: object) -> NextAction:
    if not isinstance(expression, str) or not expression.strip():
        return DirectResponse()
    # No content pre-check on purpose: the tool validates and refuses, and the
    # failure flows to the responder, which must explain rather than compute.
    return ToolCall(tool_name=CALCULATOR_TOOL_NAME, tool_input=expression.strip())


def _weather_call(location: object) -> NextAction:
    if not isinstance(location, str):
        return DirectResponse()
    location = location.strip()
    if not location or len(location) > MAX_CITY_LENGTH:
        return DirectResponse()
    return ToolCall(tool_name=WEATHER_TOOL_NAME, tool_input=location)
