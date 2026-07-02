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


def create_app(
    model: AssistantModel | None = None,
    store: ConversationStore | None = None,
    tools: list[Tool] | None = None,
) -> FastAPI:
    """Composition root: production defaults, injectable fakes for tests."""
    service = AssistantService(
        model=model if model is not None else RuleBasedAssistantModel(),
        store=store if store is not None else InMemoryConversationStore(),
        registry=ToolRegistry(
            tools if tools is not None else [CalculatorTool(), WeatherLookupTool()]
        ),
    )
    app = FastAPI(title="Streaming Chat Agent")
    app.include_router(build_chat_router(service))
    app.add_exception_handler(RequestValidationError, _validation_error_to_400)
    # Mounted last so /api routes win; html=True serves index.html at /.
    app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True))
    return app


def _validation_error_to_400(request: Request, exc: Exception) -> JSONResponse:
    # The API contract is 400 for bad input; FastAPI defaults to 422.
    return JSONResponse(status_code=400, content={"detail": "invalid request"})


app = create_app()
