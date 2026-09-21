"""HTTP request and response schemas for the CUCUMBER FastAPI app."""

from typing import Any

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=500)


class AskResponse(BaseModel):
    question: str
    sql: str
    rows: list[dict[str, Any]]
    answer: str