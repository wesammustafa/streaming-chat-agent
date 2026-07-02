import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.chat import build_chat_router
from app.domain.tools import Tool
from app.models.protocol import AssistantModel
from app.models.rule_based import RuleBasedAssistantModel
from app.services.assistant import AssistantService
from app.services.conversation_store import ConversationStore, InMemoryConversationStore
from app.services.tool_registry import ToolRegistry
from app.tools.calculator import CalculatorTool
from app.tools.weather import WeatherLookupTool
from app.tools.weather_live import LiveWeatherTool


def create_app(
    model: AssistantModel | None = None,
    store: ConversationStore | None = None,
    tools: list[Tool] | None = None,
) -> FastAPI:
    """Composition root: production defaults, injectable fakes for tests."""
    active_tools = tools if tools is not None else tools_from_env()
    service = AssistantService(
        model=model if model is not None else model_from_env(active_tools),
        store=store if store is not None else InMemoryConversationStore(),
        registry=ToolRegistry(active_tools),
    )
    app = FastAPI(title="Streaming Chat Agent")
    app.include_router(build_chat_router(service))
    app.add_exception_handler(RequestValidationError, _validation_error_to_400)
    # Mounted last so /api routes win; html=True serves index.html at /.
    app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True))
    return app


def model_from_env(tools: list[Tool]) -> AssistantModel:
    """ASSISTANT_MODEL selects the adapter; deterministic is the default and CI target."""
    choice = os.environ.get("ASSISTANT_MODEL", "deterministic").strip().lower()
    if choice == "deterministic":
        return RuleBasedAssistantModel()
    if choice == "ollama":
        from app.models.ollama import OllamaAssistantModel

        return OllamaAssistantModel(
            base_url=os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434"),
            model_name=os.environ.get("OLLAMA_MODEL", "qwen2.5:7b"),
            # The registered tools drive the planner menu; the registry stays the truth.
            tool_specs=[tool.spec for tool in tools],
        )
    raise ValueError(f"unknown ASSISTANT_MODEL {choice!r}: use 'deterministic' or 'ollama'")


def tools_from_env() -> list[Tool]:
    """WEATHER_SOURCE selects the weather adapter; the fixture is the default and CI target."""
    source = os.environ.get("WEATHER_SOURCE", "fixture").strip().lower()
    if source == "fixture":
        weather: Tool = WeatherLookupTool()
    elif source == "live":
        weather = LiveWeatherTool()
    else:
        raise ValueError(f"unknown WEATHER_SOURCE {source!r}: use 'fixture' or 'live'")
    return [CalculatorTool(), weather]


def _validation_error_to_400(request: Request, exc: Exception) -> JSONResponse:
    # The API contract is 400 for bad input; FastAPI defaults to 422.
    return JSONResponse(status_code=400, content={"detail": "invalid request"})


app = create_app()
