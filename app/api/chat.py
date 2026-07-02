import logging
from collections.abc import AsyncIterator
from uuid import uuid4

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from app.domain.events import StreamEvent
from app.services.assistant import AssistantService

logger = logging.getLogger(__name__)

MAX_MESSAGE_LENGTH = 4000
MAX_CONVERSATION_ID_LENGTH = 64
NDJSON_MEDIA_TYPE = "application/x-ndjson"


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_LENGTH)
    conversation_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_CONVERSATION_ID_LENGTH,
        # Url-safe ids only: keeps logs unforgeable and ids boring everywhere.
        pattern=r"^[A-Za-z0-9_-]+$",
    )

    @field_validator("message")
    @classmethod
    def message_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must not be blank")
        return value


def build_chat_router(service: AssistantService) -> APIRouter:
    router = APIRouter()

    @router.post("/api/chat/stream")
    async def chat_stream(request: ChatRequest) -> StreamingResponse:
        conversation_id = request.conversation_id or uuid4().hex
        stream = _ndjson_events(service, conversation_id, request.message)
        return StreamingResponse(stream, media_type=NDJSON_MEDIA_TYPE)

    return router


async def _ndjson_events(
    service: AssistantService, conversation_id: str, message: str
) -> AsyncIterator[str]:
    """Terminal-event guarantee.

    HTTP 200 is already on the wire once streaming starts, so failures must
    surface in-band: every stream ends in exactly one message_done or error.
    """
    try:
        async for event in service.stream_reply(conversation_id, message):
            yield event.to_ndjson()
    except Exception:
        # Log event metadata only, never message content.
        logger.exception("stream failed, conversation_id=%s", conversation_id)
        yield StreamEvent(type="error", error="internal error").to_ndjson()
