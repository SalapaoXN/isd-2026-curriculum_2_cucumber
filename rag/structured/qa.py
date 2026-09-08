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
_CANONICAL_PLAN_KEYS = frozenset({"coop", "no_coop", "default", "gened"})
_PREDICATE_START = re.compile(r"\b(?:WHERE|HAVING|ON)\b", re.IGNORECASE)
_PREDICATE_END = re.compile(
    r"\b(?:WHERE|HAVING|GROUP\s+BY|ORDER\s+BY|LIMIT|UNION|JOIN|"
    r"LEFT\s+JOIN|RIGHT\s+JOIN|INNER\s+JOIN|OUTER\s+JOIN|CROSS\s+JOIN)\b",
    re.IGNORECASE,
)
_PLAN_REFERENCE = re.compile(
    r"\b(?:[A-Za-z_]\w*\.)?(?P<column>plan_key|plan_name|plan_code|plan)\b",
    re.IGNORECASE,
)
_PLAN_OPERATOR = re.compile(
    r"\s*(?P<operator>NOT\s+LIKE|LIKE|NOT\s+IN|IN|!=|<>|=)",
    re.IGNORECASE,
)
_STRING_LITERAL = re.compile(r"'((?:''|[^'])*)'")


def _predicate_fragments(sql: str) -> list[str]:
    normalized_sql = re.sub(r"\s+", " ", sql)
    fragments: list[str] = []
    for start_match in _PREDICATE_START.finditer(normalized_sql):
        condition_start = start_match.end()
        end_match = _PREDICATE_END.search(normalized_sql, condition_start)
        condition_end = end_match.start() if end_match else len(normalized_sql)
        fragments.append(normalized_sql[condition_start:condition_end])
    return fragments


def _reject_unsafe_plan_filters(sql: str) -> None:
    """Require curriculum plan predicates to use canonical plan_key values."""
    for fragment in _predicate_fragments(sql):
        references = list(_PLAN_REFERENCE.finditer(fragment))
        for reference in references:
            column = reference.group("column").casefold()
            operator_match = _PLAN_OPERATOR.match(fragment, reference.end())
            if column != "plan_key":
                raise ValueError(
                    "unsafe plan filter: use exact canonical plan_key values"
                )
            if operator_match is None:
                raise ValueError(
                    "unsafe plan filter: use exact canonical plan_key values"
                )

            operator = re.sub(r"\s+", " ", operator_match.group("operator")).upper()
            value_start = operator_match.end()
            if operator == "=":
                value_text = fragment[value_start:].lstrip()
                value_match = _STRING_LITERAL.match(value_text)
                if value_match is None:
                    tokens = value_text.split()
                    if not tokens:
                        raise ValueError(
                            "unsafe plan filter: use exact canonical plan_key values"
                        )
                    value = tokens[0]
                    if value.casefold().endswith(".plan_key"):
                        continue
                    raise ValueError(
                        "unsafe plan filter: use exact canonical plan_key values"
                    )
                values = [value_match.group(1).replace("''", "'").casefold()]
            elif operator == "IN":
                value_text = fragment[value_start:].lstrip()
                if not value_text.startswith("("):
                    raise ValueError(
                        "unsafe plan filter: use exact canonical plan_key values"
                    )
                close = value_text.find(")", 1)
                if close < 0:
                    raise ValueError(
                        "unsafe plan filter: use exact canonical plan_key values"
                    )
                values = [
                    match.group(1).replace("''", "'").casefold()
                    for match in _STRING_LITERAL.finditer(
                        value_text[1:close]
                    )
                ]
                if not values:
                    raise ValueError(
                        "unsafe plan filter: use exact canonical plan_key values"
                    )
            else:
                raise ValueError(
                    "unsafe plan filter: use exact canonical plan_key values"
                )

            if any(value not in _CANONICAL_PLAN_KEYS for value in values):
                raise ValueError(
                    "unsafe plan filter: use exact canonical plan_key values"
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
    _reject_unsafe_plan_filters(sql)
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
