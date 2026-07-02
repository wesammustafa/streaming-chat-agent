from collections import defaultdict
from typing import Protocol

from app.domain.messages import Message

MAX_HISTORY_MESSAGES = 50


class ConversationStore(Protocol):
    def get(self, conversation_id: str) -> list[Message]: ...

    def append(self, conversation_id: str, message: Message) -> None: ...


class InMemoryConversationStore:
    """Per-process store; durable persistence means implementing the same protocol."""

    def __init__(self, max_messages: int = MAX_HISTORY_MESSAGES) -> None:
        self._max_messages = max_messages
        self._conversations: dict[str, list[Message]] = defaultdict(list)

    def get(self, conversation_id: str) -> list[Message]:
        # Copy, so callers can never mutate stored history.
        return list(self._conversations.get(conversation_id, ()))

    def append(self, conversation_id: str, message: Message) -> None:
        history = self._conversations[conversation_id]
        history.append(message)
        if len(history) > self._max_messages:
            # Keep the most recent messages; retention policy proper is a deferral.
            del history[: len(history) - self._max_messages]
