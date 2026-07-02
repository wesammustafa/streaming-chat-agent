"""Config selection and Ollama adapter safety, all offline: no Ollama required."""

import inspect

import pytest

import app.services.assistant
from app.domain.actions import DirectResponse, ToolCall
from app.domain.messages import Message
from app.main import model_from_env, tools_from_env
from app.models.ollama import OllamaAssistantModel, parse_plan
from app.models.rule_based import RuleBasedAssistantModel

UNREACHABLE = "http://127.0.0.1:1"


def test_default_model_is_deterministic(monkeypatch):
    monkeypatch.delenv("ASSISTANT_MODEL", raising=False)
    assert isinstance(model_from_env(tools_from_env()), RuleBasedAssistantModel)


def test_explicit_deterministic_selection(monkeypatch):
    monkeypatch.setenv("ASSISTANT_MODEL", "deterministic")
    assert isinstance(model_from_env(tools_from_env()), RuleBasedAssistantModel)


def test_ollama_selection_reads_env(monkeypatch):
    monkeypatch.setenv("ASSISTANT_MODEL", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
    monkeypatch.setenv("OLLAMA_URL", "http://127.0.0.1:12345/")
    model = model_from_env(tools_from_env())
    assert isinstance(model, OllamaAssistantModel)


def test_unknown_model_choice_fails_fast(monkeypatch):
    monkeypatch.setenv("ASSISTANT_MODEL", "gpt9000")
    with pytest.raises(ValueError, match="ASSISTANT_MODEL"):
        model_from_env(tools_from_env())


def test_both_models_satisfy_the_protocol_shape():
    for model in (RuleBasedAssistantModel(chunk_delay_seconds=0), OllamaAssistantModel()):
        assert callable(model.plan_next_action)
        assert callable(model.stream_response)


PLAN_CASES = [
    # weather: current "location" shape and older "city" shape both route
    ('{"action": "weather_lookup", "location": "Madrid"}', ToolCall("weather_lookup", "Madrid")),
    ('{"action": "weather_lookup", "city": "Madrid"}', ToolCall("weather_lookup", "Madrid")),
    (
        '```json\n{"action": "weather_lookup", "city": "Lisbon"}\n```',
        ToolCall("weather_lookup", "Lisbon"),
    ),
    ('{"action": "weather_lookup"}', DirectResponse()),
    ('{"action": "weather_lookup", "location": ""}', DirectResponse()),
    ('{"action": "weather_lookup", "location": 42}', DirectResponse()),
    ('{"action": "weather_lookup", "location": "' + "x" * 200 + '"}', DirectResponse()),
    # calculator: valid expressions route, invalid ones fall back to direct
    ('{"action": "calculator", "expression": "15 * 23"}', ToolCall("calculator", "15 * 23")),
    (
        '{"action": "calculator", "expression": "(10 - 4) * 2"}',
        ToolCall("calculator", "(10 - 4) * 2"),
    ),
    ('```json\n{"action": "calculator", "expression": "2+2"}\n```', ToolCall("calculator", "2+2")),
    ('{"action": "calculator", "expression": "2 +"}', DirectResponse()),  # syntax error
    ('{"action": "calculator", "expression": "2 ** 3"}', DirectResponse()),  # disallowed op
    ('{"action": "calculator", "expression": "__import__"}', DirectResponse()),  # unsafe name
    ('{"action": "calculator", "expression": "' + "9" * 200 + '"}', DirectResponse()),  # too long
    ('{"action": "calculator", "expression": ""}', DirectResponse()),  # empty
    ('{"action": "calculator"}', DirectResponse()),  # missing expression
    ('{"action": "calculator", "expression": 42}', DirectResponse()),  # non-string
    # non-tool and malformed plans
    ('{"action": "direct"}', DirectResponse()),
    ("no json here", DirectResponse()),
    ('{"action": "rm -rf", "city": "Madrid"}', DirectResponse()),
    ("[1, 2, 3]", DirectResponse()),
]


@pytest.mark.parametrize(("raw_plan", "expected"), PLAN_CASES)
def test_llm_plans_are_validated_before_execution(raw_plan, expected):
    assert parse_plan(raw_plan) == expected


def test_service_stays_model_agnostic():
    # Routing lives in the model adapter; the orchestrator must not branch on intent.
    source = inspect.getsource(app.services.assistant).lower()
    for token in ("calculator", "weather", "expression"):
        assert token not in source, f"orchestrator leaked intent-specific token: {token}"


async def test_unreachable_server_planning_falls_back_to_direct():
    model = OllamaAssistantModel(base_url=UNREACHABLE, request_timeout_seconds=2)
    action = await model.plan_next_action([Message(role="user", content="weather in Madrid?")])
    assert action == DirectResponse()


async def test_unreachable_server_reply_yields_guidance_instead_of_crashing():
    model = OllamaAssistantModel(base_url=UNREACHABLE, request_timeout_seconds=2)
    chunks = [c async for c in model.stream_response([Message(role="user", content="hi")])]
    text = "".join(chunks)
    assert "not responding" in text
    assert "ollama run" in text
