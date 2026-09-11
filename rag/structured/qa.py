"""Structured question answering through guarded SQL only."""

from __future__ import annotations

import sqlite3
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .execute import execute_readonly, validate_readonly_sql
from .nl_to_sql import question_to_sql, repair_sql
from .queries import course_placement


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
_COURSE_ID_EQUALITY = re.compile(
    r"\b(?P<left>[A-Za-z_]\w*)\.course_id\s*=\s*"
    r"(?P<right>[A-Za-z_]\w*)\.course_id\b",
    re.IGNORECASE,
)
_PLAN_KEY_LITERAL = re.compile(
    r"\b(?:[A-Za-z_]\w*\.)?plan_key\s*=\s*'(?P<value>(?:''|[^'])*)'",
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
_EXPLICIT_COURSE_CODE = re.compile(r"(?<![0-9])([0-9]{8})(?![0-9])")
_PLACEMENT_INTENT_TERMS = (
    "เรียนช่วงไหน",
    "อยู่ช่วงไหน",
    "เรียนปีไหน",
    "อยู่ปีไหน",
    "เทอม",
    "ภาคเรียน",
    "ภาคการศึกษา",
    "หน่วยกิต",
    "ช่วงเรียน",
    "กำหนดแน่นอน",
    "ยืดหยุ่น",
    "ลงทะเบียน",
    "ลงวิชา",
    "เรียนเมื่อไหร่",
    "placement",
    "semester",
    "academic year",
)
_MIXED_STRUCTURED_INTENT_TERMS = (
    "prerequisite",
    "pre-requisite",
    "ต้องผ่าน",
    "วิชาที่ต้องเรียนก่อน",
    "วิชาบังคับก่อน",
    "บังคับก่อน",
    "ต้องเรียนก่อน",
    "อะไรมาก่อน",
    "รวมกี่หน่วยกิต",
    "หน่วยกิตรวม",
    "ลงทะเบียนรวม",
    "alternative group",
    "กลุ่มทางเลือก",
    "อย่างใดอย่างหนึ่ง",
    "จำนวนวิชา",
    "นับจำนวน",
    "course count",
    "count statistics",
)
_DIRECT_PLAN_KEY = re.compile(
    r"(?<![a-z0-9_])(?P<plan>no_coop|coop|default|gened)(?![a-z0-9_])"
)
_PROGRAM_CODE = re.compile(
    r"(?<![a-z0-9_])(?P<program>dsba|bit|it|gened)(?![a-z0-9_])"
)
_PLACEMENT_COLUMNS = (
    "placement_id",
    "plan_key",
    "plan_id",
    "catalog_id",
    "course_id",
    "year",
    "semester",
    "flexible_year_semester_raw",
    "credits_raw",
    "alternative_group_id",
    "provenance",
)


def _course_placement_request(
    question: str,
) -> tuple[str, str, list[str]] | None:
    normalized = question.casefold()
    course_codes = list(dict.fromkeys(_EXPLICIT_COURSE_CODE.findall(normalized)))
    if len(course_codes) != 1:
        return None
    program_match = _PROGRAM_CODE.search(normalized)
    if program_match is None:
        return None
    if any(term in normalized for term in _MIXED_STRUCTURED_INTENT_TERMS):
        return None
    if not any(term in normalized for term in _PLACEMENT_INTENT_TERMS):
        return None

    plan_keys: list[str] = []
    for match in _DIRECT_PLAN_KEY.finditer(normalized):
        plan_key = match.group("plan")
        if plan_key not in plan_keys:
            plan_keys.append(plan_key)
    if "ไม่สหกิจ" in normalized and "no_coop" not in plan_keys:
        plan_keys.append("no_coop")
    thai_without_no_coop = normalized.replace("ไม่สหกิจ", "")
    if "สหกิจ" in thai_without_no_coop and "coop" not in plan_keys:
        plan_keys.append("coop")
    if not plan_keys:
        return None
    return program_match.group("program"), course_codes[0], plan_keys


def _course_placement_structured_result(
    result: dict[str, Any],
) -> dict[str, Any]:
    rows = [
        tuple(placement.get(column) for column in _PLACEMENT_COLUMNS)
        for placement in result["placements"]
    ]
    return {
        "operation": "course_placement",
        "status": result["status"],
        "missing_plan_keys": result["missing_plan_keys"],
        "sql": None,
        "columns": list(_PLACEMENT_COLUMNS),
        "rows": rows,
    }


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


def _reject_shared_course_id_cross_plan_join(sql: str) -> None:
    normalized_sql = re.sub(r"\s+", " ", sql)
    aliases: dict[str, str] = {}
    for match in _RELATION_REFERENCE.finditer(normalized_sql):
        table = match.group("table").casefold()
        aliases[table] = table
        alias = match.group("alias")
        if alias is not None and alias.casefold() not in _SQL_RESERVED_WORDS:
            aliases[alias.casefold()] = table

    course_aliases = {
        alias for alias, table in aliases.items() if table == "courses"
    }
    plan_aliases = {
        alias for alias, table in aliases.items() if table == "v_plan_courses"
    }
    plans_by_course_alias: dict[str, set[str]] = {}
    for match in _ON_CLAUSE.finditer(normalized_sql):
        for equality in _COURSE_ID_EQUALITY.finditer(match.group("condition")):
            left = equality.group("left").casefold()
            right = equality.group("right").casefold()
            if left in plan_aliases and right in course_aliases:
                plans_by_course_alias.setdefault(right, set()).add(left)
            elif right in plan_aliases and left in course_aliases:
                plans_by_course_alias.setdefault(left, set()).add(right)

    plan_values = {
        match.group("value").replace("''", "'").casefold()
        for match in _PLAN_KEY_LITERAL.finditer(normalized_sql)
    }
    shared_anchor = any(
        len(plan_aliases_for_anchor) >= 2
        for plan_aliases_for_anchor in plans_by_course_alias.values()
    )
    if shared_anchor and {"coop", "no_coop"}.issubset(plan_values):
        raise ValueError(
            "cross-plan query must not join plan rows through one shared "
            "courses.course_id"
        )


def _validate_structured_sql(db_path: str | Path, sql: str) -> None:
    _reject_unsafe_plan_filters(sql)
    _reject_cross_catalog_course_code_join(sql)
    _reject_shared_course_id_cross_plan_join(sql)
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
    model_callable: Callable[[str], str] | None,
) -> dict[str, Any]:
    """Generate guarded SQL, execute it read-only, and return raw results."""
    placement_request = _course_placement_request(question)
    if placement_request is not None:
        program, course_code, plan_keys = placement_request
        return _course_placement_structured_result(
            course_placement(db_path, program, course_code, plan_keys)
        )
    if not callable(model_callable):
        raise ValueError(
            "structured_model_callable is required for structured questions"
        )
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
