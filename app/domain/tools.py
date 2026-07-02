from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    content: str | None = None
    error: str | None = None

    @classmethod
    def succeeded(cls, content: str) -> "ToolResult":
        return cls(ok=True, content=content)

    @classmethod
    def failed(cls, error: str) -> "ToolResult":
        return cls(ok=False, error=error)


class Tool(Protocol):
    spec: ToolSpec

    async def run(self, tool_input: str) -> ToolResult: ...
