import asyncio
from collections import defaultdict
from collections.abc import AsyncIterator

from app.domain.actions import ToolCall
from app.domain.events import StreamEvent
from app.domain.messages import Message
from app.domain.tools import ToolResult
from app.models.protocol import AssistantModel
from app.services.conversation_store import ConversationStore
from app.services.tool_registry import ToolRegistry

DEFAULT_TOOL_TIMEOUT_SECONDS = 5.0


class AssistantService:
    """Orchestrates one reply: plan, maybe run a tool, stream text, persist.

    Copy-free by design: every user-facing string comes from the model.
    """

    def __init__(
        self,
        model: AssistantModel,
        store: ConversationStore,
        registry: ToolRegistry,
        tool_timeout_seconds: float = DEFAULT_TOOL_TIMEOUT_SECONDS,
    ) -> None:
        self._model = model
        self._store = store
        self._registry = registry
        self._tool_timeout_seconds = tool_timeout_seconds
        # Serializes replies per conversation; shares the in-memory store's lifetime.
        self._conversation_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def stream_reply(
        self, conversation_id: str, user_text: str
    ) -> AsyncIterator[StreamEvent]:
        # message_start goes out before the lock and planning: fast time to first event.
        yield StreamEvent(type="message_start", conversation_id=conversation_id)

        # One reply at a time per conversation, so history writes never interleave.
        async with self._conversation_locks[conversation_id]:
            # The user message persists immediately; the assistant message only on
            # successful completion (disconnect policy, see DESIGN.md).
            self._store.append(conversation_id, Message(role="user", content=user_text))
            messages = self._store.get(conversation_id)

            action = await self._model.plan_next_action(messages)
            tool_result: ToolResult | None = None
            if isinstance(action, ToolCall):
                yield StreamEvent(type="tool_start", tool_name=action.tool_name)
                tool_result = await self._run_tool(action)
                if tool_result.ok:
                    yield StreamEvent(
                        type="tool_result", tool_name=action.tool_name, result=tool_result.content
                    )
                else:
                    yield StreamEvent(
                        type="tool_error", tool_name=action.tool_name, error=tool_result.error
                    )

            chunks: list[str] = []
            async for chunk in self._model.stream_response(messages, tool_result):
                chunks.append(chunk)
                yield StreamEvent(type="text_delta", text=chunk)

            self._store.append(conversation_id, Message(role="assistant", content="".join(chunks)))
        yield StreamEvent(type="message_done", conversation_id=conversation_id)

    async def _run_tool(self, action: ToolCall) -> ToolResult:
        try:
            return await asyncio.wait_for(
                self._registry.run(action.tool_name, action.tool_input),
                timeout=self._tool_timeout_seconds,
            )
        except TimeoutError:
            return ToolResult.failed(f"{action.tool_name} timed out")
        except Exception:
            # A tool must never kill the stream; the model turns failures into copy.
            return ToolResult.failed(f"{action.tool_name} failed unexpectedly")
