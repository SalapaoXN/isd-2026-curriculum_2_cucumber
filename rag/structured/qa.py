"""Structured question answering through guarded SQL only."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .execute import execute_readonly, validate_readonly_sql
from .nl_to_sql import question_to_sql, repair_sql


def _is_repairable_validation_error(error: Exception) -> bool:
    if isinstance(error, ValueError):
        return True
    if not isinstance(error, sqlite3.OperationalError):
        return False
    infrastructure_markers = (
        "unable to open database file",
        "database is locked",
        "database table is locked",
        "database or disk is full",
        "disk i/o error",
        "not a database",
    )
    message = str(error).casefold()
    return not any(marker in message for marker in infrastructure_markers)


def _repair_once(
    db_path: str | Path,
    question: str,
    schema_text: str,
    failed_sql: str,
    validation_error: Exception,
    model_callable: Callable[[str], str],
) -> str:
    try:
        repaired_sql = repair_sql(
            question,
            schema_text,
            failed_sql,
            str(validation_error),
            model_callable,
        )
    except ValueError:
        raise ValueError("SQL repair failed validation") from None

    try:
        validate_readonly_sql(db_path, repaired_sql)
    except FileNotFoundError:
        raise
    except (ValueError, sqlite3.OperationalError) as error:
        if not _is_repairable_validation_error(error):
            raise
        raise ValueError("SQL repair failed validation") from None
    return repaired_sql


def ask_structured(
    db_path: str | Path,
    question: str,
    schema_text: str,
    model_callable: Callable[[str], str],
) -> dict[str, Any]:
    """Generate guarded SQL, execute it read-only, and return raw results."""
    generated_sql: str | None = None

    def capture_generated_sql(prompt: str) -> str:
        nonlocal generated_sql
        generated_sql = model_callable(prompt)
        return generated_sql

    try:
        sql = question_to_sql(question, schema_text, capture_generated_sql)
    except ValueError as error:
        if generated_sql is None:
            raise
        sql = _repair_once(
            db_path,
            question,
            schema_text,
            generated_sql,
            error,
            model_callable,
        )
    else:
        try:
            validate_readonly_sql(db_path, sql)
        except FileNotFoundError:
            raise
        except (ValueError, sqlite3.OperationalError) as error:
            if not _is_repairable_validation_error(error):
                raise
            sql = _repair_once(
                db_path,
                question,
                schema_text,
                sql,
                error,
                model_callable,
            )

    columns, rows = execute_readonly(db_path, sql)
    return {
        "sql": sql,
        "columns": columns,
        "rows": rows,
    }


__all__ = ["ask_structured"]
