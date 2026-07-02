import json

import httpx
import pytest

from app.domain.actions import DirectResponse
from app.domain.events import TERMINAL_EVENT_TYPES
from app.main import create_app
from app.models.rule_based import RuleBasedAssistantModel


class ExplodingModel:
    async def plan_next_action(self, messages):
        return DirectResponse()

    async def stream_response(self, messages, tool_result=None):
        yield "partial "
        raise RuntimeError("boom")


class CapturingModel:
    def __init__(self):
        self.seen_history_sizes = []

    async def plan_next_action(self, messages):
        self.seen_history_sizes.append(len(messages))
        return DirectResponse()

    async def stream_response(self, messages, tool_result=None):
        yield "ok"


def fast_app(**kwargs):
    kwargs.setdefault("model", RuleBasedAssistantModel(chunk_delay_seconds=0))
    return create_app(**kwargs)


async def post_stream(app, payload):
    transport = httpx.ASGITransport(app=app)
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://test") as client,
        client.stream("POST", "/api/chat/stream", json=payload) as response,
    ):
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/x-ndjson")
        lines = [line async for line in response.aiter_lines() if line.strip()]
    return [json.loads(line) for line in lines]


def types_of(events):
    return [event["type"] for event in events]


def text_of(events):
    return "".join(e["text"] for e in events if e["type"] == "text_delta")


async def test_non_tool_stream_has_no_tool_events():
    events = await post_stream(fast_app(), {"message": "hello!"})
    types = types_of(events)
    assert types[0] == "message_start"
    assert types[-1] == "message_done"
    assert not [t for t in types if t.startswith("tool_")]
    assert text_of(events)


async def test_tool_stream_orders_tool_events_before_text():
    events = await post_stream(fast_app(), {"message": "what is 2 + 2?"})
    types = types_of(events)
    assert types[:3] == ["message_start", "tool_start", "tool_result"]
    assert types[-1] == "message_done"
    assert "4" in text_of(events)


async def test_division_by_zero_streams_tool_error_then_completes():
    events = await post_stream(fast_app(), {"message": "1 / 0"})
    types = types_of(events)
    assert types[:3] == ["message_start", "tool_start", "tool_error"]
    assert types[-1] == "message_done"
    assert "division by zero" in text_of(events)


@pytest.mark.parametrize("message", ["hello!", "what is 2 + 2?", "1 / 0", "tell me a story"])
async def test_every_stream_ends_in_exactly_one_terminal_event(message):
    events = await post_stream(fast_app(), {"message": message})
    terminal_positions = [i for i, t in enumerate(types_of(events)) if t in TERMINAL_EVENT_TYPES]
    assert terminal_positions == [len(events) - 1]


BAD_REQUESTS = [
    {"message": ""},
    {"message": "   "},
    {"message": "x" * 4001},
    {"message": "hi", "conversation_id": "c" * 65},
    {},
]


@pytest.mark.parametrize("payload", BAD_REQUESTS)
async def test_invalid_requests_get_400_before_streaming(payload):
    transport = httpx.ASGITransport(app=fast_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/chat/stream", json=payload)
    assert response.status_code == 400


async def test_mid_stream_failure_surfaces_in_band_error():
    events = await post_stream(create_app(model=ExplodingModel()), {"message": "hi"})
    types = types_of(events)
    assert types[-1] == "error"
    assert [t for t in types if t in TERMINAL_EVENT_TYPES] == ["error"]


async def test_provided_conversation_id_is_echoed_in_message_start():
    events = await post_stream(fast_app(), {"message": "hi", "conversation_id": "my-convo"})
    assert events[0] == {"type": "message_start", "conversation_id": "my-convo"}


async def test_missing_conversation_id_is_generated():
    events = await post_stream(fast_app(), {"message": "hi"})
    generated = events[0]["conversation_id"]
    assert generated
    assert len(generated) <= 64


async def test_same_conversation_id_accumulates_history():
    model = CapturingModel()
    app = create_app(model=model)
    await post_stream(app, {"message": "one", "conversation_id": "abc"})
    await post_stream(app, {"message": "two", "conversation_id": "abc"})
    assert model.seen_history_sizes == [1, 3]
