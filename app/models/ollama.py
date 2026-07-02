"""Optional local LLM adapter over Ollama (recommended model: qwen2.5:7b).

Demo realism only: non-deterministic, never used by tests or CI. The
deterministic RuleBasedAssistantModel stays the default everywhere. The LLM
is only trusted to decide whether a weather_lookup tool call is useful; its
plan is validated here and anything invalid falls back to a direct reply.
"""

import json
import re
from collections.abc import AsyncIterator

import httpx

from app.domain.actions import DirectResponse, NextAction, ToolCall
from app.domain.messages import Message
from app.domain.tools import ToolResult

WEATHER_TOOL_NAME = "weather_lookup"
MAX_CITY_LENGTH = 80

_PLANNER_PROMPT = (
    "You route chat messages. Decide if the user's LAST message is a simple question "
    "about the current weather in a specific city.\n"
    'If it is, reply exactly: {"action": "weather_lookup", "city": "<city name>"}\n'
    'Otherwise reply exactly: {"action": "direct"}\n'
    "Reply with that JSON only, no other text."
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
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model_name = model_name
        # Long read timeout: a 7B model can pause noticeably between tokens on first load.
        self._timeout = httpx.Timeout(request_timeout_seconds, connect=5.0)

    async def plan_next_action(self, messages: list[Message]) -> NextAction:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{self._base_url}/api/chat",
                    json={
                        "model": self._model_name,
                        "messages": [
                            {"role": "system", "content": _PLANNER_PROMPT},
                            {"role": "user", "content": messages[-1].content},
                        ],
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
                httpx.AsyncClient(timeout=self._timeout) as client,
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

    def _responder_messages(
        self, messages: list[Message], tool_result: ToolResult | None
    ) -> list[dict[str, str]]:
        chat = [{"role": "system", "content": _RESPONDER_PROMPT}]
        chat.extend({"role": m.role, "content": m.content} for m in messages)
        if tool_result is not None:
            if tool_result.ok:
                note = f"Weather tool result: {tool_result.content}. Answer using it."
            else:
                note = f"The weather tool failed ({tool_result.error}). Briefly say so."
            chat.append({"role": "system", "content": note})
        return chat


def parse_plan(raw_plan: str) -> NextAction:
    """Validate the LLM's tool plan; anything unexpected becomes a direct response."""
    match = re.search(r"\{.*\}", raw_plan, re.S)
    if match is None:
        return DirectResponse()
    try:
        plan = json.loads(match.group())
    except json.JSONDecodeError:
        return DirectResponse()
    if not isinstance(plan, dict) or plan.get("action") != WEATHER_TOOL_NAME:
        return DirectResponse()
    city = plan.get("city")
    if not isinstance(city, str):
        return DirectResponse()
    city = city.strip()
    if not city or len(city) > MAX_CITY_LENGTH:
        return DirectResponse()
    return ToolCall(tool_name=WEATHER_TOOL_NAME, tool_input=city)
