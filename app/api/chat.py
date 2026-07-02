from collections.abc import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from app.domain.events import StreamEvent

MAX_MESSAGE_LENGTH = 4000
MAX_CONVERSATION_ID_LENGTH = 64
NDJSON_MEDIA_TYPE = "application/x-ndjson"


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_LENGTH)
    conversation_id: str | None = Field(
        default=None, min_length=1, max_length=MAX_CONVERSATION_ID_LENGTH
    )

    @field_validator("message")
    @classmethod
    def message_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must not be blank")
        return value


def build_chat_router() -> APIRouter:
    router = APIRouter()

    @router.post("/api/chat/stream")
    async def chat_stream(request: ChatRequest) -> StreamingResponse:
        return StreamingResponse(_demo_stream(), media_type=NDJSON_MEDIA_TYPE)

    return router


async def _demo_stream() -> AsyncIterator[str]:
    # Walking skeleton: proves NDJSON streaming end to end before the real service exists.
    events = [
        StreamEvent(type="message_start", conversation_id="demo"),
        StreamEvent(type="text_delta", text="Hello "),
        StreamEvent(type="text_delta", text="world"),
        StreamEvent(type="message_done"),
    ]
    for event in events:
        yield event.to_ndjson()
