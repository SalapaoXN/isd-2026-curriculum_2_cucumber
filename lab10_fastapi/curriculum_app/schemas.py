"""HTTP request and response schemas for the CUCUMBER FastAPI app."""

from typing import Any

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=500)
    conversation_context: dict[str, Any] | None = Field(default=None)


class AskResponse(BaseModel):
    question: str
    answer: str
    status: str
    action: str | None = None
    route: str | None = None
    provenance: list[dict[str, Any]] = Field(default_factory=list)
    next_context: dict[str, Any] | None = None
