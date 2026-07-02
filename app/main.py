from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.chat import build_chat_router


def create_app() -> FastAPI:
    app = FastAPI(title="Streaming Chat Agent")
    app.include_router(build_chat_router())
    app.add_exception_handler(RequestValidationError, _validation_error_to_400)
    return app


def _validation_error_to_400(request: Request, exc: Exception) -> JSONResponse:
    # The API contract is 400 for bad input; FastAPI defaults to 422.
    return JSONResponse(status_code=400, content={"detail": "invalid request"})


app = create_app()
