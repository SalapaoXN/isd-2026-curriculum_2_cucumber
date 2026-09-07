"""Model-agnostic question-to-SQL boundary."""

from __future__ import annotations

from collections.abc import Callable

from .guard_sql import guard_sql


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
        f"Schema:\n{schema_text}\n\n"
        f"Question:\n{question.strip()}"
    )
    generated = model_callable(prompt)
    if not isinstance(generated, str):
        raise TypeError("model_callable must return a SQL string")

    return guard_sql(_remove_code_fence(generated))


__all__ = ["question_to_sql"]
