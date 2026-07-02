from dataclasses import dataclass


@dataclass(frozen=True)
class ToolCall:
    """The model wants a tool run before it answers."""

    tool_name: str
    tool_input: str


@dataclass(frozen=True)
class DirectResponse:
    """Bare marker: the model will answer directly via stream_response."""


NextAction = ToolCall | DirectResponse
