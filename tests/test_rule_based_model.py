import pytest

from app.domain.actions import DirectResponse, ToolCall
from app.domain.messages import Message
from app.domain.tools import ToolResult
from app.models.rule_based import FALLBACK_REPLY, GREETING_REPLY, RuleBasedAssistantModel


def user(text: str) -> list[Message]:
    return [Message(role="user", content=text)]


async def collect(chunks) -> str:
    return "".join([chunk async for chunk in chunks])


@pytest.fixture
def model() -> RuleBasedAssistantModel:
    return RuleBasedAssistantModel(chunk_delay_seconds=0)


TOOL_CASES = [
    ("what is 2 + 2?", "2 + 2"),
    ("What's 10/4", "10/4"),
    ("calculate (1 + 2) * 3", "(1 + 2) * 3"),
    ("how much is 10 - 4", "10 - 4"),
    ("2+2", "2+2"),
    ("1 / 0", "1 / 0"),
]


@pytest.mark.parametrize(("text", "expected_input"), TOOL_CASES)
async def test_arithmetic_requests_become_tool_calls(model, text, expected_input):
    action = await model.plan_next_action(user(text))
    assert action == ToolCall(tool_name="calculator", tool_input=expected_input)


NEAR_MISSES = [
    "I ate 2 apples",
    "route 66 - the song",
    "calculate route 66",
    "what is love",
    "I have 3 + 4 questions about your service",
    "meet me at 7",
    "hello there",
]


@pytest.mark.parametrize("text", NEAR_MISSES)
async def test_near_misses_stay_direct_responses(model, text):
    action = await model.plan_next_action(user(text))
    assert isinstance(action, DirectResponse)


async def test_greeting_gets_greeting_copy(model):
    assert await collect(model.stream_response(user("hello!"))) == GREETING_REPLY


async def test_non_arithmetic_gets_fallback_copy(model):
    assert await collect(model.stream_response(user("tell me a story"))) == FALLBACK_REPLY


async def test_tool_success_response_includes_expression_and_result(model):
    response = model.stream_response(user("what is 2 + 2?"), ToolResult.succeeded("4"))
    assert await collect(response) == "2 + 2 = 4"


async def test_tool_error_response_surfaces_reason(model):
    response = model.stream_response(user("1 / 0"), ToolResult.failed("division by zero"))
    assert "division by zero" in await collect(response)


async def test_response_streams_word_by_word(model):
    chunks = [chunk async for chunk in model.stream_response(user("hi"))]
    assert len(chunks) > 1
    assert "".join(chunks) == GREETING_REPLY
