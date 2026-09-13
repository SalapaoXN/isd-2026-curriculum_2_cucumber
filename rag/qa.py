"""Top-level curriculum QA over one relational and vector database."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from rag.query_spec import parse_query_spec
from rag.retrieval.retrieve import retrieve
from rag.resolution import ResolutionOutcome, resolve_query_spec
from rag.router import route_question
from rag.structured.qa import ask_structured


SCHEMA_PATH = Path(__file__).with_name("structured") / "schema.sql"


def _blocked_result(outcome: ResolutionOutcome) -> dict[str, Any]:
    return {
        "status": outcome.action,
        "action": outcome.action,
        "blocking_ambiguity": outcome.blocking_ambiguity,
        "resolved_program": outcome.resolved_program,
        "course_references": [
            {
                "reference_type": reference.reference_type,
                "reference": reference.reference,
                "candidates": [dict(candidate) for candidate in reference.candidates],
            }
            for reference in outcome.course_references
        ],
    }


def ask(
    db_path: str | Path,
    question: str,
    structured_model_callable: Callable[[str], str] | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """Retrieve SQL evidence, semantic evidence, or both for one question."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    resolution = resolve_query_spec(parse_query_spec(question), db_path)
    if resolution.action != "answer":
        return {"route": None, "result": _blocked_result(resolution)}

    route = route_question(question)
    structured_result: dict[str, Any] | None = None
    semantic_result: list[dict[str, Any]] | None = None
    if route in {"structured", "hybrid"}:
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
