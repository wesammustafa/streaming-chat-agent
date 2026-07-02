import asyncio

import pytest

from app.domain.actions import DirectResponse, ToolCall
from app.domain.messages import Message
from app.domain.tools import ToolResult, ToolSpec
from app.services.assistant import AssistantService
from app.services.conversation_store import InMemoryConversationStore
from app.services.tool_registry import ToolRegistry


class ScriptedModel:
    """Second AssistantModel implementation: proves the protocol seam works."""

    def __init__(self, action, chunks=("Hello ", "there")):
        self._action = action
        self._chunks = chunks
        # "unset" distinguishes "never called" from a genuine None tool result.
        self.seen_tool_result: ToolResult | str | None = "unset"
        self.seen_messages: list[Message] | None = None

    async def plan_next_action(self, messages):
        self.seen_messages = messages
        return self._action

    async def stream_response(self, messages, tool_result=None):
        self.seen_tool_result = tool_result
        for chunk in self._chunks:
            yield chunk


class ExplodingStreamModel:
    async def plan_next_action(self, messages):
        return DirectResponse()

    async def stream_response(self, messages, tool_result=None):
        yield "partial "
        raise RuntimeError("model died mid-stream")


class SucceedingTool:
    spec = ToolSpec(name="fake", description="always succeeds")

    async def run(self, tool_input):
        return ToolResult.succeeded("42")


class RaisingTool:
    spec = ToolSpec(name="fake", description="always raises")

    async def run(self, tool_input):
        raise RuntimeError("boom")


class NeverResolvingTool:
    spec = ToolSpec(name="fake", description="hangs forever")

    async def run(self, tool_input):
        await asyncio.Event().wait()


CALL_FAKE = ToolCall(tool_name="fake", tool_input="anything")


def make_service(model, tools=(), timeout=1.0, store=None):
    return AssistantService(
        model=model,
        store=store or InMemoryConversationStore(),
        registry=ToolRegistry(tools),
        tool_timeout_seconds=timeout,
    )


async def events_of(service, conversation_id="c1", text="hi"):
    return [event async for event in service.stream_reply(conversation_id, text)]


def types_of(events):
    return [event.type for event in events]


async def test_direct_path_event_sequence():
    model = ScriptedModel(DirectResponse())
    events = await events_of(make_service(model))
    assert types_of(events) == ["message_start", "text_delta", "text_delta", "message_done"]
    assert events[0].conversation_id == "c1"
    assert model.seen_tool_result is None


async def test_tool_success_event_sequence_and_payload():
    model = ScriptedModel(CALL_FAKE)
    events = await events_of(make_service(model, tools=[SucceedingTool()]))
    assert types_of(events) == [
        "message_start",
        "tool_start",
        "tool_result",
        "text_delta",
        "text_delta",
        "message_done",
    ]
    assert events[2].tool_name == "fake"
    assert events[2].result == "42"
    assert model.seen_tool_result == ToolResult.succeeded("42")


async def test_raising_tool_becomes_tool_error_and_reply_still_completes():
    model = ScriptedModel(CALL_FAKE)
    events = await events_of(make_service(model, tools=[RaisingTool()]))
    assert types_of(events) == [
        "message_start",
        "tool_start",
        "tool_error",
        "text_delta",
        "text_delta",
        "message_done",
    ]
    assert "failed unexpectedly" in events[2].error
    assert model.seen_tool_result == ToolResult.failed("fake failed unexpectedly")


async def test_hanging_tool_times_out_via_injected_timeout():
    model = ScriptedModel(CALL_FAKE)
    events = await events_of(make_service(model, tools=[NeverResolvingTool()], timeout=0.01))
    assert "tool_error" in types_of(events)
    assert "timed out" in events[2].error


async def test_unknown_tool_name_becomes_tool_error():
    model = ScriptedModel(ToolCall(tool_name="nope", tool_input="x"))
    events = await events_of(make_service(model, tools=[]))
    assert "tool_error" in types_of(events)
    assert "unknown tool" in events[2].error


async def test_completed_reply_is_persisted():
    store = InMemoryConversationStore()
    await events_of(make_service(ScriptedModel(DirectResponse()), store=store), text="question")
    history = store.get("c1")
    assert [(m.role, m.content) for m in history] == [
        ("user", "question"),
        ("assistant", "Hello there"),
    ]


async def test_assistant_message_not_persisted_when_stream_fails():
    store = InMemoryConversationStore()
    with pytest.raises(RuntimeError):
        await events_of(make_service(ExplodingStreamModel(), store=store))
    assert [m.role for m in store.get("c1")] == ["user"]


async def test_conversations_do_not_leak_into_each_other():
    store = InMemoryConversationStore()
    service = make_service(ScriptedModel(DirectResponse()), store=store)
    await events_of(service, conversation_id="a", text="for a")
    await events_of(service, conversation_id="b", text="for b")
    assert [m.content for m in store.get("a")] == ["for a", "Hello there"]
    assert [m.content for m in store.get("b")] == ["for b", "Hello there"]


async def test_model_sees_full_history_on_later_turns():
    store = InMemoryConversationStore()
    model = ScriptedModel(DirectResponse())
    service = make_service(model, store=store)
    await events_of(service, text="first")
    await events_of(service, text="second")
    assert model.seen_messages is not None
    assert [m.content for m in model.seen_messages] == ["first", "Hello there", "second"]


async def test_user_message_is_stored_before_planning():
    model = ScriptedModel(DirectResponse())
    await events_of(make_service(model), text="just sent")
    assert model.seen_messages is not None
    assert model.seen_messages[-1] == Message(role="user", content="just sent")


class YieldingModel:
    """Suspends mid-reply, so an unserialized concurrent request would interleave."""

    async def plan_next_action(self, messages):
        await asyncio.sleep(0)
        return DirectResponse()

    async def stream_response(self, messages, tool_result=None):
        yield "part one "
        await asyncio.sleep(0)
        yield "part two"


async def test_same_conversation_replies_do_not_interleave_history():
    store = InMemoryConversationStore()
    service = make_service(YieldingModel(), store=store)
    await asyncio.gather(
        events_of(service, conversation_id="c1", text="first"),
        events_of(service, conversation_id="c1", text="second"),
    )
    assert [(m.role, m.content) for m in store.get("c1")] == [
        ("user", "first"),
        ("assistant", "part one part two"),
        ("user", "second"),
        ("assistant", "part one part two"),
    ]


async def test_different_conversations_are_not_serialized():
    gate = asyncio.Event()

    class GatedModel:
        async def plan_next_action(self, messages):
            return DirectResponse()

        async def stream_response(self, messages, tool_result=None):
            if messages[-1].content == "blocked":
                await gate.wait()
            yield "done"

    service = make_service(GatedModel())
    blocked = asyncio.create_task(events_of(service, conversation_id="a", text="blocked"))
    # A global lock would deadlock conversation "b" behind "a" and trip the timeout.
    events = await asyncio.wait_for(
        events_of(service, conversation_id="b", text="free"), timeout=1
    )
    assert types_of(events)[-1] == "message_done"
    gate.set()
    await blocked
