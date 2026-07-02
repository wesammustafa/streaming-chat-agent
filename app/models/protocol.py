from collections.abc import AsyncIterator
from typing import Protocol

from app.domain.actions import NextAction
from app.domain.messages import Message
from app.domain.tools import ToolResult


class AssistantModel(Protocol):
    """The seam a real LLM adapter would implement; both methods are async-shaped."""

    async def plan_next_action(self, messages: list[Message]) -> NextAction: ...

    def stream_response(
        self, messages: list[Message], tool_result: ToolResult | None = None
    ) -> AsyncIterator[str]:
        """Sole authority for user-facing text, on both the direct and post-tool paths."""
        ...
