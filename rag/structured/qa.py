"""Structured question answering through guarded SQL only."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from .execute import execute_readonly
from .nl_to_sql import question_to_sql


def ask_structured(
    db_path: str | Path,
    question: str,
    schema_text: str,
    model_callable: Callable[[str], str],
) -> dict[str, Any]:
    """Generate guarded SQL, execute it read-only, and return raw results."""
    sql = question_to_sql(question, schema_text, model_callable)
    columns, rows = execute_readonly(db_path, sql)
    return {
        "sql": sql,
        "columns": columns,
        "rows": rows,
    }


__all__ = ["ask_structured"]
