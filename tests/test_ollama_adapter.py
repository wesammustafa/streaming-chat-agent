"""Offline IO tests for the Ollama adapter over httpx.MockTransport, no server.

Plan validation and model selection live in test_model_selection.py; this file
covers the HTTP behavior itself: streaming assembly, request composition, and
the two error paths (before and after the first emitted token).
"""

import json
from typing import Any

import httpx
import pytest

from app.domain.actions import ToolCall
from app.domain.messages import Message
from app.domain.tools import ToolResult, ToolSpec
from app.models.ollama import PLANNER_CONTEXT_MESSAGES, OllamaAssistantModel
from app.tools.calculator import CalculatorTool
from app.tools.weather import WeatherLookupTool


def user(text: str) -> list[Message]:
    return [Message(role="user", content=text)]


async def collect(chunks):
    return [chunk async for chunk in chunks]


def streaming_model(lines):
    body = "".join(json.dumps(line) + "\n" for line in lines).encode()
    return OllamaAssistantModel(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=body))
    )


def capturing_transport(payloads):
    def handler(request):
        payloads.append(json.loads(request.content))
        return httpx.Response(200, content=b'{"message": {"content": "ok"}}\n')

    return httpx.MockTransport(handler)


def capturing_model(payloads):
    return OllamaAssistantModel(transport=capturing_transport(payloads))


async def test_stream_response_assembles_chunks_in_order():
    model = streaming_model(
        [
            {"message": {"content": "Hola"}},
            {"message": {"content": ", "}},
            {"message": {"content": "mundo"}},
            {"done": True},
        ]
    )
    assert await collect(model.stream_response(user("hola"))) == ["Hola", ", ", "mundo"]


async def test_stream_response_skips_blank_and_contentless_lines():
    model = streaming_model([{"message": {"content": ""}}, {}, {"message": {"content": "solo"}}])
    assert await collect(model.stream_response(user("hi"))) == ["solo"]


async def test_plan_round_trips_through_the_transport():
    def handler(request):
        plan = '{"action": "calculator", "expression": "2+2"}'
        return httpx.Response(200, json={"message": {"content": plan}})

    model = OllamaAssistantModel(transport=httpx.MockTransport(handler))
    action = await model.plan_next_action(user("what is 2+2?"))
    assert action == ToolCall(tool_name="calculator", tool_input="2+2")


async def test_http_error_before_any_output_yields_setup_guidance():
    model = OllamaAssistantModel(
        transport=httpx.MockTransport(lambda request: httpx.Response(500))
    )
    text = "".join(await collect(model.stream_response(user("hi"))))
    assert "not responding" in text
    assert "ollama run" in text


class DroppingStream(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b'{"message": {"content": "partial "}}\n'
        raise httpx.ReadError("connection dropped mid-stream")


async def test_http_error_after_partial_output_reraises():
    # Once tokens went out, failures must surface (in-band error upstream), not
    # get papered over with the setup-guidance copy.
    model = OllamaAssistantModel(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=DroppingStream()))
    )
    chunks = []
    with pytest.raises(httpx.HTTPError):
        async for chunk in model.stream_response(user("hi")):
            chunks.append(chunk)
    assert chunks == ["partial "]


async def test_planner_sends_recent_history_for_follow_ups():
    payloads: list[dict[str, Any]] = []
    model = capturing_model(payloads)
    history = [
        Message(role="user", content="what's the weather in Madrid?"),
        Message(role="assistant", content="Madrid: 31°C, sunny."),
        Message(role="user", content="and in Lisbon?"),
    ]
    await model.plan_next_action(history)
    sent = payloads[0]["messages"]
    assert sent[0]["role"] == "system"
    assert sent[1:] == [{"role": m.role, "content": m.content} for m in history]


async def test_planner_context_window_is_capped():
    payloads: list[dict[str, Any]] = []
    model = capturing_model(payloads)
    history = [Message(role="user", content=f"message {i}") for i in range(10)]
    await model.plan_next_action(history)
    sent = payloads[0]["messages"]
    assert len(sent) == 1 + PLANNER_CONTEXT_MESSAGES
    assert sent[1]["content"] == "message 4"
    assert sent[-1]["content"] == "message 9"


async def test_planner_menu_is_built_from_registered_tool_specs():
    payloads: list[dict[str, Any]] = []
    specs = [ToolSpec(name="weather_lookup", description="Answers rooftop weather riddles")]
    model = OllamaAssistantModel(transport=capturing_transport(payloads), tool_specs=specs)
    await model.plan_next_action(user("hi"))
    system = payloads[0]["messages"][0]["content"]
    assert "Answers rooftop weather riddles" in system
    assert "calculator" not in system  # unregistered tools drop off the menu


async def test_planner_menu_omits_specs_it_cannot_validate():
    payloads: list[dict[str, Any]] = []
    specs = [
        ToolSpec(name="gif_search", description="Finds a gif"),
        CalculatorTool.spec,
    ]
    model = OllamaAssistantModel(transport=capturing_transport(payloads), tool_specs=specs)
    await model.plan_next_action(user("hi"))
    system = payloads[0]["messages"][0]["content"]
    assert "gif_search" not in system
    assert CalculatorTool.spec.description.rstrip(".") in system
    assert '"action": "direct"' in system


async def test_planner_menu_defaults_to_the_real_tool_specs():
    payloads: list[dict[str, Any]] = []
    model = capturing_model(payloads)
    await model.plan_next_action(user("hi"))
    system = payloads[0]["messages"][0]["content"]
    assert CalculatorTool.spec.description.rstrip(".") in system
    assert WeatherLookupTool.spec.description.rstrip(".") in system


async def test_responder_prepends_system_prompt_and_maps_history():
    payloads: list[dict[str, Any]] = []
    model = capturing_model(payloads)
    history = [
        Message(role="user", content="hola"),
        Message(role="assistant", content="¡hola!"),
        Message(role="user", content="¿qué tal?"),
    ]
    await collect(model.stream_response(history))
    messages = payloads[0]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1:] == [{"role": m.role, "content": m.content} for m in history]
    assert payloads[0]["stream"] is True


async def test_responder_appends_tool_success_note():
    payloads: list[dict[str, Any]] = []
    model = capturing_model(payloads)
    result = ToolResult.succeeded("Madrid: 31°C, sunny")
    await collect(model.stream_response(user("weather in Madrid?"), result))
    note = payloads[0]["messages"][-1]
    assert note["role"] == "system"
    assert "Madrid: 31°C, sunny" in note["content"]


async def test_responder_appends_tool_failure_note():
    payloads: list[dict[str, Any]] = []
    model = capturing_model(payloads)
    result = ToolResult.failed("division by zero")
    await collect(model.stream_response(user("1 / 0"), result))
    note = payloads[0]["messages"][-1]
    assert note["role"] == "system"
    assert "division by zero" in note["content"]
