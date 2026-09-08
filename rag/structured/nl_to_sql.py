"""Model-agnostic question-to-SQL boundary."""

from __future__ import annotations

from collections.abc import Callable
import re

from .guard_sql import guard_sql


_QUOTED_SQL_VALUE = re.compile(r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|`(?:``|[^`])*`")


def _has_top_level_limit(sql: str) -> bool:
    masked = _QUOTED_SQL_VALUE.sub(" ", sql)
    depth = 0
    for token in re.finditer(r"\(|\)|\bLIMIT\b", masked, re.IGNORECASE):
        if token.group(0) == "(":
            depth += 1
        elif token.group(0) == ")":
            depth = max(0, depth - 1)
        elif depth == 0:
            return True
    return False


def _guard_generated_sql(sql: str) -> str:
    if _has_top_level_limit(sql):
        return guard_sql(sql)

    stripped = sql.rstrip()
    if stripped.endswith(";"):
        bounded = f"{stripped[:-1].rstrip()} LIMIT 100;"
    else:
        bounded = f"{stripped} LIMIT 100"
    return guard_sql(bounded)


def _remove_code_fence(value: str) -> str:
    text = value.strip()
    lines = text.splitlines()
    if len(lines) >= 2 and lines[0].strip().startswith("```") and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return text


def question_to_sql(
    question: str,
    schema_text: str,
    model_callable: Callable[[str], str],
) -> str:
    """Generate one guarded SQL string using an injected model callable."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    if not isinstance(schema_text, str) or not schema_text.strip():
        raise ValueError("schema_text must be a non-empty string")
    if not callable(model_callable):
        raise TypeError("model_callable must be callable")

    prompt = (
        "Convert the user's curriculum question into exactly one SQLite SELECT "
        "or WITH query. Use only tables and columns in the schema. Return SQL "
        "only, without explanations or Markdown fences. Do not execute the query.\n\n"
        "v_plan_courses does NOT expose course-name columns. When course names "
        "are needed, JOIN courses ON courses.course_id = v_plan_courses.course_id "
        "and use courses.name_th or courses.name_en. Prefer qualified column "
        "names in joins.\n\n"
        f"Schema:\n{schema_text}\n\n"
        f"Question:\n{question.strip()}"
    )
    generated = model_callable(prompt)
    if not isinstance(generated, str):
        raise TypeError("model_callable must return a SQL string")

    return _guard_generated_sql(_remove_code_fence(generated))


def repair_sql(
    question: str,
    schema_text: str,
    failed_sql: str,
    validation_error: str,
    model_callable: Callable[[str], str],
) -> str:
    """Repair one failed SQL query using the same injected model boundary."""
    for value, name in (
        (question, "question"),
        (schema_text, "schema_text"),
        (failed_sql, "failed_sql"),
        (validation_error, "validation_error"),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
    if not callable(model_callable):
        raise TypeError("model_callable must be callable")

    prompt = (
        "Repair the failed SQLite query for the user's original curriculum "
        "question while preserving the original intent. Use only tables and "
        "columns in the provided schema. Return exactly one SQLite SELECT or "
        "WITH query, SQL only, without explanations or Markdown fences. Do not "
        "execute the query.\n\n"
        f"Schema:\n{schema_text}\n\n"
        f"Original question:\n{question.strip()}\n\n"
        f"Failed SQL:\n{failed_sql}\n\n"
        f"Validation or guard error:\n{validation_error}"
    )
    repaired = model_callable(prompt)
    if not isinstance(repaired, str):
        raise TypeError("model_callable must return a SQL string")

    return _guard_generated_sql(_remove_code_fence(repaired))


__all__ = ["question_to_sql", "repair_sql"]
