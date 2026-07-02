from collections.abc import Iterable

from app.domain.tools import Tool, ToolResult


class ToolRegistry:
    """The backend is the authority on which tools exist, not the model."""

    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools = {tool.spec.name: tool for tool in tools}

    async def run(self, name: str, tool_input: str) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult.failed(f"unknown tool: {name}")
        return await tool.run(tool_input)
