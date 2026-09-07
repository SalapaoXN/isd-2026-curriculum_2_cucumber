"""Top-level curriculum QA over one relational and vector database."""

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
    """Retrieve SQL evidence, semantic evidence, or both for one question."""
    route = route_question(question)
    structured_result: dict[str, Any] | None = None
    semantic_result: list[dict[str, Any]] | None = None
    if route in {"structured", "hybrid"}:
        if not callable(structured_model_callable):
            raise ValueError(
                "structured_model_callable is required for structured questions"
            )
        structured_result = ask_structured(
            db_path,
            question,
            SCHEMA_PATH.read_text(encoding="utf-8"),
            structured_model_callable,
        )
    if route in {"semantic", "hybrid"}:
        semantic_result = retrieve(db_path, question, k=top_k)

    if route == "hybrid":
        result: dict[str, Any] | list[dict[str, Any]] = {
            "structured": structured_result,
            "semantic": semantic_result,
        }
    elif route == "structured":
        result = structured_result
    else:
        result = semantic_result
    return {"route": route, "result": result}


__all__ = ["ask"]
