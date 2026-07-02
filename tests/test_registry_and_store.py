from app.domain.messages import Message
from app.domain.tools import ToolResult, ToolSpec
from app.services.conversation_store import InMemoryConversationStore
from app.services.tool_registry import ToolRegistry


class EchoTool:
    spec = ToolSpec(name="echo", description="echoes its input")

    async def run(self, tool_input: str) -> ToolResult:
        return ToolResult.succeeded(tool_input)


async def test_registry_runs_registered_tool():
    result = await ToolRegistry([EchoTool()]).run("echo", "hi")
    assert result == ToolResult.succeeded("hi")


async def test_registry_fails_closed_on_unknown_tool():
    result = await ToolRegistry([EchoTool()]).run("weather", "Madrid")
    assert not result.ok
    assert "unknown tool" in result.error


def test_store_returns_history_in_order():
    store = InMemoryConversationStore()
    store.append("c", Message(role="user", content="one"))
    store.append("c", Message(role="assistant", content="two"))
    assert [m.content for m in store.get("c")] == ["one", "two"]


def test_store_isolates_conversations():
    store = InMemoryConversationStore()
    store.append("a", Message(role="user", content="for a"))
    assert store.get("b") == []


def test_store_caps_history_keeping_most_recent():
    store = InMemoryConversationStore(max_messages=3)
    for i in range(5):
        store.append("c", Message(role="user", content=str(i)))
    assert [m.content for m in store.get("c")] == ["2", "3", "4"]


def test_store_get_returns_a_copy():
    store = InMemoryConversationStore()
    store.append("c", Message(role="user", content="hi"))
    store.get("c").append(Message(role="assistant", content="mutated"))
    assert len(store.get("c")) == 1
