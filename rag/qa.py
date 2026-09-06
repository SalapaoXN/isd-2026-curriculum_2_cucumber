"""Top-level deterministic routing between structured and semantic QA."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from rag.retrieval.retrieve import retrieve
from rag.router import route_question
from rag.structured.qa import ask_structured


SCHEMA_PATH = Path(__file__).with_name("structured") / "schema.sql"


def ask(
    db_path: str | Path,
    question: str,
    structured_model_callable: Callable[[str], str] | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """Route one question to structured QA or semantic retrieval."""
    route = route_question(question)
    if route == "structured":
        if not callable(structured_model_callable):
            raise ValueError(
                "structured_model_callable is required for structured questions"
            )
        result = ask_structured(
            db_path,
            question,
            SCHEMA_PATH.read_text(encoding="utf-8"),
            structured_model_callable,
        )
    else:
        result = retrieve(db_path, question, k=top_k)
    return {"route": route, "result": result}


__all__ = ["ask"]
