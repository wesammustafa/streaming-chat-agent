"""Config selection and Ollama adapter safety, all offline: no Ollama required."""

import pytest

from app.domain.actions import DirectResponse, ToolCall
from app.domain.messages import Message
from app.main import model_from_env
from app.models.ollama import OllamaAssistantModel, parse_plan
from app.models.rule_based import RuleBasedAssistantModel

UNREACHABLE = "http://127.0.0.1:1"


def test_default_model_is_deterministic(monkeypatch):
    monkeypatch.delenv("ASSISTANT_MODEL", raising=False)
    assert isinstance(model_from_env(), RuleBasedAssistantModel)


def test_explicit_deterministic_selection(monkeypatch):
    monkeypatch.setenv("ASSISTANT_MODEL", "deterministic")
    assert isinstance(model_from_env(), RuleBasedAssistantModel)


def test_ollama_selection_reads_env(monkeypatch):
    monkeypatch.setenv("ASSISTANT_MODEL", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
    monkeypatch.setenv("OLLAMA_URL", "http://127.0.0.1:12345/")
    model = model_from_env()
    assert isinstance(model, OllamaAssistantModel)


def test_unknown_model_choice_fails_fast(monkeypatch):
    monkeypatch.setenv("ASSISTANT_MODEL", "gpt9000")
    with pytest.raises(ValueError, match="ASSISTANT_MODEL"):
        model_from_env()


def test_both_models_satisfy_the_protocol_shape():
    for model in (RuleBasedAssistantModel(chunk_delay_seconds=0), OllamaAssistantModel()):
        assert callable(model.plan_next_action)
        assert callable(model.stream_response)


PLAN_CASES = [
    ('{"action": "weather_lookup", "city": "Madrid"}', ToolCall("weather_lookup", "Madrid")),
    (
        '```json\n{"action": "weather_lookup", "city": "Lisbon"}\n```',
        ToolCall("weather_lookup", "Lisbon"),
    ),
    ('{"action": "direct"}', DirectResponse()),
    ("no json here", DirectResponse()),
    ('{"action": "weather_lookup"}', DirectResponse()),
    ('{"action": "weather_lookup", "city": ""}', DirectResponse()),
    ('{"action": "weather_lookup", "city": 42}', DirectResponse()),
    ('{"action": "rm -rf", "city": "Madrid"}', DirectResponse()),
    ('{"action": "weather_lookup", "city": "' + "x" * 200 + '"}', DirectResponse()),
    ("[1, 2, 3]", DirectResponse()),
]


@pytest.mark.parametrize(("raw_plan", "expected"), PLAN_CASES)
def test_llm_plans_are_validated_before_execution(raw_plan, expected):
    assert parse_plan(raw_plan) == expected


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
