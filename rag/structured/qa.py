"""Structured question answering through guarded SQL only."""

from __future__ import annotations

import sqlite3
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .execute import execute_readonly, validate_readonly_sql
from .nl_to_sql import question_to_sql, repair_sql


_SQL_RESERVED_WORDS = frozenset(
    {
        "as",
        "cross",
        "full",
        "group",
        "having",
        "inner",
        "join",
        "left",
        "limit",
        "on",
        "order",
        "outer",
        "right",
        "union",
        "where",
    }
)
_RELATION_REFERENCE = re.compile(
    r"\b(?:FROM|JOIN)\s+(?P<table>v_plan_courses|courses)\b"
    r"(?:\s+(?:AS\s+)?(?P<alias>[A-Za-z_]\w*))?",
    re.IGNORECASE,
)
_ON_CLAUSE = re.compile(
    r"\bON\b(?P<condition>.*?)(?=\b(?:JOIN|LEFT|RIGHT|INNER|OUTER|CROSS|FULL|WHERE|GROUP|ORDER|HAVING|LIMIT|UNION)\b|$)",
    re.IGNORECASE | re.DOTALL,
)
_COURSE_CODE_EQUALITY = re.compile(
    r"\b(?P<left>[A-Za-z_]\w*)\.course_code\s*=\s*"
    r"(?P<right>[A-Za-z_]\w*)\.course_code\b",
    re.IGNORECASE,
)


def _reject_cross_catalog_course_code_join(sql: str) -> None:
    normalized_sql = re.sub(r"\s+", " ", sql)
    aliases: dict[str, str] = {}
    for match in _RELATION_REFERENCE.finditer(normalized_sql):
        table = match.group("table").casefold()
        aliases[table] = table
        alias = match.group("alias")
        if alias is not None and alias.casefold() not in _SQL_RESERVED_WORDS:
            aliases[alias.casefold()] = table

    for match in _ON_CLAUSE.finditer(normalized_sql):
        for equality in _COURSE_CODE_EQUALITY.finditer(match.group("condition")):
            joined_tables = {
                aliases.get(equality.group("left").casefold()),
                aliases.get(equality.group("right").casefold()),
            }
            if joined_tables == {"courses", "v_plan_courses"}:
                raise ValueError(
                    "join courses to v_plan_courses using course_id, not course_code"
                )


def _validate_structured_sql(db_path: str | Path, sql: str) -> None:
    _reject_cross_catalog_course_code_join(sql)
    validate_readonly_sql(db_path, sql)


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
        _validate_structured_sql(db_path, repaired_sql)
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
            _validate_structured_sql(db_path, sql)
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
