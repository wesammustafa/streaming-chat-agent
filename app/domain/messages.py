from typing import Literal

from pydantic import BaseModel

Role = Literal["user", "assistant"]


class Message(BaseModel):
    role: Role
    content: str
