from typing import Literal

from pydantic import BaseModel

EventType = Literal[
    "message_start",
    "text_delta",
    "tool_start",
    "tool_result",
    "tool_error",
    "message_done",
    "error",
]

TERMINAL_EVENT_TYPES: frozenset[str] = frozenset({"message_done", "error"})


class StreamEvent(BaseModel):
    """Flat event model; unused fields stay None and are omitted on the wire."""

    type: EventType
    conversation_id: str | None = None
    text: str | None = None
    tool_name: str | None = None
    result: str | None = None
    error: str | None = None

    def to_ndjson(self) -> str:
        return self.model_dump_json(exclude_none=True) + "\n"
