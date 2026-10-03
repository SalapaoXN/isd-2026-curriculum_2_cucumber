"""Standalone, guarded LLM-to-SQL curriculum QA service."""

from __future__ import annotations

import base64
import json
import math
from pathlib import Path
import re
import sqlite3
from typing import Any, Callable

from rag.structured.execute import execute_readonly
from rag.structured.guard_sql import (
    extract_column_predicate_literals,
    guard_sql,
)
from rag.structured.nl_to_sql import question_to_sql, repair_sql
from rag.query_spec import QuerySpec, parse_query_spec


_ALLOWED_RELATIONS = frozenset(
    {
        "v_plan_courses",
        "courses",
        "catalogs",
        "curriculum_plans",
        "programs",
        "prerequisites",
        "v_prerequisite_edges",
        "v_semester_credits",
        "alternative_course_groups",
        "alternative_course_group_members",
        "program_requirements",
        "policy_facts",
        "regulation_rules",
    }
)
_SCHEMA_RELATION = re.compile(
    r"^\s*CREATE\s+(?:TABLE|VIEW)\s+(?:IF\s+NOT\s+EXISTS\s+)?"
    r"[\[\`\"]?([A-Za-z_][A-Za-z0-9_]*)",
    re.IGNORECASE,
)
_SEMESTER_CREDIT_VIEW_SEMANTICS = (
    "Canonical view semantics for v_semester_credits:\n"
    "- Output columns: plan_id, program, plan, year, semester, total_credits.\n"
    "- This is a pre-aggregated semester-level view. Its grain is one row per "
    "plan_id/program/plan/year/semester.\n"
    "- total_credits is the semester-level aggregate measure, already computed "
    "from course-level credit_units. Use total_credits for semester totals; do "
    "not aggregate it again when one row already represents the requested scope.\n"
    "- credit_units is not an output column of v_semester_credits. It is a "
    "course-level source field used inside the view; do not reference it as "
    "v_semester_credits.credit_units. Sum total_credits only when intentionally "
    "combining distinct, well-defined plan rows and that combination is requested."
)
_MAX_RESULT_ROWS = 100
_MAX_PROMPT_ROWS = 20
_MAX_PROMPT_CELL_CHARS = 500
_MAX_CONTEXT_COURSES = 50
_MAX_CONTEXT_CODE_CHARS = 64
_MAX_CONTEXT_NAME_CHARS = 160
_NO_DATA_ANSWER_PHRASES = (
    "ไม่พบข้อมูลที่ตรงกัน",
    "ไม่พบข้อมูลที่ตรงกับคำถาม",
    "ไม่พบข้อมูล",
    "ไม่มีข้อมูลที่ตรงกัน",
    "ไม่มีข้อมูล",
    "no matching data",
    "no data found",
    "no matching information",
)
_PLAN_SCOPE_GUIDANCE = (
    "Service-specific plan-scope rules (these override any general examples above):\n"
    "- A selected program scopes the program only; use program/program_code for it. "
    "A selected program does not select a curriculum plan.\n"
    "- Never infer or invent a plan_key from a program. In particular, do not assume "
    "IT -> no_coop, IT -> coop, DSBA -> default, AIT -> default, or any similar mapping.\n"
    "- Add a plan_key predicate only when the user explicitly names a plan or an "
    "application context explicitly contains a selected plan. selected_program alone "
    "is not selected-plan context.\n"
    "- If no plan is explicitly selected, leave plan_key unfiltered and query across "
    "all applicable plans for the selected program. Use SELECT DISTINCT when needed "
    "to avoid duplicate course rows caused only by the same course appearing in "
    "multiple plans.\n"
    "- Canonical explicit plan keys are coop, no_coop, default, and gened. AIT is a "
    "program code and by itself never selects the default plan."
)
_PROGRAM_SCOPE_GUIDANCE = (
    "Service-specific program-scope rules (these override any general examples above):\n"
    "- When selected_program is supplied, scope results to that program only.\n"
    "- When no active selected program is supplied, all programs remain eligible. "
    "Never infer or choose a program from a course title, course code, or question wording.\n"
    "- For a title or code lookup that may match across programs or catalogs, preserve "
    "all canonical matches across programs and catalogs; do not use LIMIT 1 to choose "
    "an arbitrary match.\n"
    "- Include program identity in the result columns whenever multiple program matches "
    "may exist (use program or program_code). DISTINCT may remove true duplicate rows "
    "only; never collapse distinct program/catalog matches."
)
_UNIQUE_COLUMN_GUIDANCE = (
    "Service-specific result-column rules:\n"
    "- Every SELECT result must have unique output column names. Alias columns when "
    "same-named fields are selected from multiple tables or semantic entities; never "
    "emit duplicate output names.\n"
    "- For prerequisite-style source/target results, use distinct aliases such as "
    "source_course_code, source_name_th, source_name_en, prerequisite_course_code, "
    "prerequisite_name_th, and prerequisite_name_en."
)
_CATALOG_CONTEXT_GUIDANCE = (
    "Edition-context projection rules:\n"
    "- catalog_key from catalogs is the canonical curriculum-edition identity.\n"
    "- When returning course rows that can seed follow-up context, project catalog_key, "
    "program, and course_code together. Get program/course identity from v_plan_courses, "
    "join courses by course_id, and join catalogs through courses.catalog_id.\n"
    "- Never derive catalog_key from filenames or substitute academic_year.\n"
)
_EVIDENCE_PROJECTION_GUIDANCE = (
    "Service-specific evidence-projection rules:\n"
    "- When the user asks about or filters by a factual course attribute, include "
    "that answer-relevant factual attribute in the SELECT list whenever it is needed "
    "to explain why the returned rows satisfy the request. The answer model sees only "
    "the returned columns and rows, not this SQL.\n"
    "- Examples: a 3-credit filter should project course identity and credit_units; "
    "a question asking when a course is taken should project course identity and year; "
    "a semester question should project course identity and semester; prerequisite "
    "questions should project source and prerequisite identities.\n"
    "- Do not require every WHERE column in the SELECT list. Scope-only fields such as "
    "program may remain unprojected unless needed to disambiguate the requested answer."
)
_COMPARISON_COMPLETENESS_GUIDANCE = (
    "Service-specific explicit-comparison completeness rules:\n"
    "- When the user explicitly compares a finite set of categories or scopes, "
    "represent every explicitly requested comparison side in the SQL result, "
    "even when one side has no underlying rows. Never let a missing group silently "
    "disappear from an explicit comparison.\n"
    "- For additive totals such as credits, materialize each requested bucket with "
    "safe read-only SQL semantics, for example an explicit requested-bucket CTE "
    "or equivalent, LEFT JOIN to the canonical aggregate, and COALESCE(total, 0). "
    "Conditional aggregation is also acceptable. Project the category and each "
    "compared value so the answer model can support both sides and the conclusion. "
    "Preserve canonical placement and credit-grain semantics; do not count duplicate "
    "catalog copies or repeated plan placements as extra credits unless they are "
    "part of the explicitly requested comparison scope.\n"
    "- Preserve all requested values when they tie so the answer can report equality. "
    "Do not choose a comparison winner with LIMIT 1 or ordering alone.\n"
    "- Apply zero materialization only to sides explicitly named in the comparison; "
    "do not invent unspecified categories. Ordinary grouped questions such as "
    "asking what each semester has remain limited to canonical groups that exist."
)
_LOGICAL_COURSE_COUNTING_GUIDANCE = (
    "Service-specific logical course identity and counting rules:\n"
    "- course_id identifies a physical course row or catalog copy. course_code "
    "identifies the logical course for curriculum-level counting.\n"
    "- When the user asks for distinct, unique, or non-duplicate courses (including "
    "รายวิชาไม่ซ้ำ), count COUNT(DISTINCT course_code), not COUNT(DISTINCT course_id).\n"
    "- For counts by program, group independently by program and count course_code "
    "within each program. The same course_code may count once in each program.\n"
    "- Repeated placements or plans for one course_code in the same program count "
    "once as a logical course.\n"
    "- course_id remains correct for physical-row joins, catalog-row relationships, "
    "and prerequisite-edge identity; do not globally replace it with course_code."
)
_FOCUS_COURSE_KEYS = frozenset(
    {"catalog_key", "course_code", "course_name", "program"}
)
_RESULT_COURSE_KEYS = frozenset(
    {"catalog_key", "program", "course_code", "course_name"}
)


def parse_focus_course_context(
    value: Any, selected_program: str | None
) -> dict[str, str | None] | None:
    """Validate the small service-owned course focus and invalidate stale scope."""
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) - _FOCUS_COURSE_KEYS:
        raise ValueError("focus_course must contain only course identity fields")
    course_code = value.get("course_code")
    if not isinstance(course_code, str) or not course_code.strip():
        raise ValueError("focus_course.course_code must be a non-empty string")
    course_name = value.get("course_name")
    if course_name is not None and (
        not isinstance(course_name, str) or not course_name.strip()
    ):
        raise ValueError("focus_course.course_name must be a non-empty string")
    focus_program = value.get("program", selected_program)
    if focus_program is not None and (
        not isinstance(focus_program, str) or not focus_program.strip()
    ):
        raise ValueError("focus_course.program must be a non-empty string or null")
    if focus_program != selected_program:
        return None
    catalog_key = value.get("catalog_key")
    if catalog_key is not None and (
        not isinstance(catalog_key, str)
        or not catalog_key.strip()
        or len(catalog_key.strip()) > 128
    ):
        raise ValueError("focus_course.catalog_key must be a non-empty string or null")
    normalized: dict[str, str | None] = {
        "course_code": course_code.strip(),
        "program": selected_program,
    }
    if catalog_key is not None:
        normalized["catalog_key"] = catalog_key.strip()
    if course_name is not None:
        normalized["course_name"] = course_name.strip()
        # Keep the serialized key order stable and the shape course-first.
        normalized = {
            "course_code": course_code.strip(),
            "course_name": course_name.strip(),
            "program": selected_program,
        }
        if catalog_key is not None:
            normalized["catalog_key"] = catalog_key.strip()
    return normalized


def parse_result_courses_context(
    value: Any,
    result_scope_program: Any,
    selected_program: str | None,
    result_set_empty: Any = False,
) -> tuple[list[dict[str, str | None]], str | None] | None:
    """Validate a bounded result set and discard it when its program scope is stale."""
    if value is None:
        if result_set_empty:
            raise ValueError("result_set_empty requires result_courses")
        return None
    if not isinstance(value, list):
        raise ValueError("result_courses must be a list")
    if result_set_empty is not True and not value:
        raise ValueError("result_courses must be a non-empty list")
    if result_set_empty is True and value:
        raise ValueError("result_set_empty requires an empty result_courses list")
    if len(value) > _MAX_RESULT_ROWS:
        raise ValueError("result_courses exceeds the input row bound")
    if result_scope_program is not None and (
        not isinstance(result_scope_program, str)
        or not result_scope_program.strip()
    ):
        raise ValueError("result_scope_program must be a non-empty string or null")
    if result_scope_program != selected_program:
        return None

    if result_set_empty is True:
        return [], selected_program

    normalized: list[dict[str, str | None]] = []
    seen: dict[tuple[str | None, str, str], str | None] = {}
    for item in value:
        if not isinstance(item, dict) or set(item) - _RESULT_COURSE_KEYS:
            raise ValueError("result_courses items must contain course identity only")
        item_program = item.get("program")
        course_code = item.get("course_code")
        course_name = item.get("course_name")
        if not isinstance(item_program, str) or not item_program.strip():
            raise ValueError("result_courses.program must be a non-empty string")
        if not isinstance(course_code, str) or not course_code.strip():
            raise ValueError("result_courses.course_code must be a non-empty string")
        item_program = item_program.strip()
        course_code = course_code.strip()
        if len(item_program) > 32 or len(course_code) > _MAX_CONTEXT_CODE_CHARS:
            raise ValueError("result_courses identity exceeds the context bound")
        catalog_key = item.get("catalog_key")
        if catalog_key is not None and (
            not isinstance(catalog_key, str)
            or not catalog_key.strip()
            or len(catalog_key.strip()) > 128
        ):
            raise ValueError("result_courses.catalog_key is invalid or too long")
        catalog_key = catalog_key.strip() if catalog_key is not None else None
        if course_name is not None and (
            not isinstance(course_name, str)
            or not course_name.strip()
            or len(course_name) > _MAX_CONTEXT_NAME_CHARS
        ):
            raise ValueError("result_courses.course_name is invalid or too long")
        if selected_program is not None and (
            item_program.casefold() != selected_program.casefold()
        ):
            return None

        identity = (
            catalog_key.casefold() if catalog_key is not None else None,
            item_program.casefold(),
            course_code.casefold(),
        )
        normalized_name = course_name.strip() if course_name is not None else None
        if identity in seen:
            if seen[identity] != normalized_name:
                raise ValueError("duplicate result course has conflicting names")
            continue
        seen[identity] = normalized_name
        normalized_item: dict[str, str | None] = {
            "program": item_program,
            "course_code": course_code,
        }
        if catalog_key is not None:
            normalized_item["catalog_key"] = catalog_key
        if normalized_name is not None:
            normalized_item["course_name"] = normalized_name
        normalized.append(normalized_item)
        if len(normalized) > _MAX_CONTEXT_COURSES:
            raise ValueError("result_courses exceeds the context bound")

    if not normalized:
        raise ValueError("result_courses contains no unique course")
    return normalized, selected_program


def _conversation_focus_guidance(focus: dict[str, str | None] | None) -> str:
    if focus is None:
        return ""
    payload = {
        "program": focus["program"],
        "course_code": focus["course_code"],
    }
    if focus.get("catalog_key") is not None:
        payload["catalog_key"] = focus["catalog_key"]
    if focus.get("course_name") is not None:
        payload["course_name"] = focus["course_name"]
    return (
        "Conversation focus (structural context only; not additional evidence):\n"
        + json.dumps(payload, ensure_ascii=False, allow_nan=False)
        + "\nIf catalog_key is present, it is the canonical edition selector and must remain in scope. "
        + "\nThe current question may use a reference such as แล้ว, ตัวนี้, วิชานี้, or อันนี้. "
        "Use the focused course only when needed to resolve the current question. "
        "Explicit entities in the current question override the conversation focus. "
        "Do not force the focused course into unrelated questions.\n"
    )


def _conversation_result_set_guidance(
    courses: list[dict[str, str | None]] | None,
    result_scope_program: str | None,
) -> str:
    if courses is None:
        return ""
    if not courses:
        return (
            "Previous course result set:\nKNOWN EMPTY\n"
            f"Program scope: {result_scope_program or 'all programs'}.\n"
            "If the current question refers to this previous set, it contains zero courses. "
            "Do NOT substitute the whole selected program and do NOT invent course identities. "
            "Keep a referential query empty-scoped (for example, use a safe SELECT predicate "
            "that returns no matching courses). Explicit entities or a new explicit scope in "
            "the CURRENT question override previous result context. Do not force this empty "
            "set into an unrelated new question. Always query canonical SQLite again; never "
            "answer from this context alone.\n"
        )
    lines = [
        f"- {course.get('catalog_key', '[legacy catalog key omitted]')} | "
        f"{course['program']} | {course['course_code']}"
        + (f" | {course['course_name']}" if course.get("course_name") else "")
        for course in courses
    ]
    return (
        "Previous result-set courses (structural context only; not answer data):\n"
        + "\n".join(lines)
        + "\nThe current question may refer to this set with wording such as "
        "ตัวไหน, พวกนี้, ในนี้, อันไหน, or วิชาเหล่านี้. Use this set only when "
        "the current question needs that reference resolved. Explicit course or "
        "program entities in the current question override prior result context. "
        "Do not force this set into an unrelated new question. When using this set "
        "as scope, restrict SQL to these canonical (catalog_key, program, course_code) "
        "identities. catalog_key is the canonical edition selector. "
        "Always query canonical SQLite again; never answer from this context alone.\n"
    )


def _result_courses_from_rows(
    rows: list[dict[str, Any]], selected_program: str | None
) -> tuple[list[dict[str, str | None]] | None, bool]:
    """Return unique canonical identities, or flag a set too large to retain."""
    if not rows:
        return None, False

    courses: dict[tuple[str | None, str, str], dict[str, str | None]] = {}
    for row in rows:
        raw_program = row.get("program", row.get("program_code"))
        if raw_program is None:
            course_program = selected_program or ""
        elif isinstance(raw_program, str) and raw_program.strip():
            course_program = raw_program.strip()
        else:
            return None, False
        if len(course_program) > 32:
            return None, False

        raw_code = row.get("course_code")
        if not isinstance(raw_code, (str, int)) or isinstance(raw_code, bool):
            return None, False
        course_code = str(raw_code).strip()
        if not course_code or len(course_code) > _MAX_CONTEXT_CODE_CHARS:
            return None, False
        if selected_program is not None and (
            course_program.casefold() != selected_program.casefold()
        ):
            return None, False

        raw_catalog_key = row.get("catalog_key")
        if raw_catalog_key is not None and (
            not isinstance(raw_catalog_key, str) or not raw_catalog_key.strip()
        ):
            return None, False
        catalog_key = raw_catalog_key.strip() if isinstance(raw_catalog_key, str) else None
        if catalog_key is not None and len(catalog_key) > 128:
            return None, False
        identity = (
            catalog_key.casefold() if catalog_key is not None else None,
            course_program.casefold(),
            course_code.casefold(),
        )
        name = next(
            (
                row.get(key).strip()
                for key in ("course_name", "name_en", "name_th", "course_title")
                if isinstance(row.get(key), str) and row.get(key).strip()
            ),
            None,
        )
        if name is not None and len(name) > _MAX_CONTEXT_NAME_CHARS:
            name = None
        existing = courses.get(identity)
        if existing is not None:
            old_name = existing.get("course_name")
            if old_name is not None and name is not None and old_name != name:
                return None, False
            if old_name is None and name is not None:
                existing["course_name"] = name
            continue

        item: dict[str, str | None] = {
            "program": course_program,
            "course_code": course_code,
        }
        if catalog_key is not None:
            item["catalog_key"] = catalog_key
        if name is not None:
            item["course_name"] = name
        courses[identity] = item
        if len(courses) > _MAX_CONTEXT_COURSES:
            return None, True

    return list(courses.values()), False


def _next_conversation_context(
    selected_program: str | None,
    prior_focus: dict[str, str | None] | None,
    prior_result_courses: list[dict[str, str | None]] | None,
    rows: list[dict[str, Any]],
    *,
    establish_empty_result_set: bool = False,
) -> dict[str, Any] | None:
    result_courses, over_bound = _result_courses_from_rows(rows, selected_program)
    context: dict[str, Any] = {}
    if selected_program is not None:
        context["program"] = selected_program
    if over_bound:
        return context or None
    if result_courses is not None:
        if len(result_courses) == 1:
            course = result_courses[0]
            context["focus_course"] = {
                "course_code": course["course_code"],
                "program": selected_program,
                **(
                    {"catalog_key": course["catalog_key"]}
                    if course.get("catalog_key") is not None
                    else {}
                ),
                **(
                    {"course_name": course["course_name"]}
                    if course.get("course_name") is not None
                    else {}
                ),
            }
        elif len(result_courses) > 1 and all(
            course["program"] for course in result_courses
        ):
            context["result_courses"] = result_courses
            context["result_scope_program"] = selected_program
        return context or None

    if prior_focus is not None:
        context["focus_course"] = prior_focus
    if prior_result_courses is not None:
        context["result_courses"] = prior_result_courses
        context["result_scope_program"] = selected_program
        if not prior_result_courses:
            context["result_set_empty"] = True
    elif establish_empty_result_set:
        context["result_courses"] = []
        context["result_set_empty"] = True
        context["result_scope_program"] = selected_program
    return context or None


class _SqlModelFailure(Exception):
    """Distinguish provider failures from generated SQL validation failures."""


def _normalize_course_code(value: str) -> str:
    return value.strip().casefold()


def _has_explicit_current_scope(spec: QuerySpec) -> bool:
    return bool(
        spec.course_codes
        or spec.course_name
        or spec.years
        or spec.semesters
        or spec.plans
        or spec.program
        or any(
            operation in {"list", "program_discovery"}
            for operation in spec.operations
        )
    )


def _references_focus_course(question: str) -> bool:
    compact = "".join(question.casefold().split())
    return any(
        "".join(anchor.split()) in compact
        for anchor in ("ตัวนี้", "วิชานี้", "อันนี้")
    )


def _explicit_course_or_plan_override(spec: QuerySpec) -> bool:
    return bool(
        spec.course_codes
        or spec.course_name
        or spec.years
        or spec.semesters
        or spec.plans
        or spec.program
    )


def _unsupported_course_code_literal(
    sql: str,
    allowed_course_codes: set[str],
) -> str | None:
    for literal in extract_column_predicate_literals(
        sql, {"course_code", "course_code_normalized"}
    ):
        if _normalize_course_code(literal) not in allowed_course_codes:
            return literal
    return None


def _repairable_sqlite_error(error: sqlite3.OperationalError) -> str | None:
    """Return a sanitized SQL-generation error category, if repairable."""
    message = str(error).casefold()
    if "no such column:" in message:
        return "SQLite schema validation error: no such column"
    if "ambiguous column name:" in message:
        return "SQLite schema validation error: ambiguous column name"
    if "no such function:" in message:
        return "SQLite query error: no such function"
    if "syntax error" in message:
        return "SQLite query syntax error"
    if "incomplete input" in message:
        return "SQLite query is incomplete"
    if "misuse of aggregate" in message or "misuse of window" in message:
        return "SQLite query contains an invalid aggregate or window expression"
    return None


def _canonical_schema() -> str:
    schema_path = Path(__file__).resolve().parents[1] / "rag" / "structured" / "schema.sql"
    statements = schema_path.read_text(encoding="utf-8").split(";")
    selected: list[str] = []
    for statement in statements:
        match = _SCHEMA_RELATION.match(statement)
        if match and match.group(1).casefold() in _ALLOWED_RELATIONS:
            selected.append(statement.strip() + ";")
    if not selected:
        raise RuntimeError("canonical curriculum schema is unavailable")
    return "\n\n".join((*selected, _SEMESTER_CREDIT_VIEW_SEMANTICS))


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, bytes):
        return {"base64": base64.b64encode(value).decode("ascii")}
    raise TypeError("SQLite returned an unsupported value type")


def _failure(code: str, sql: str = "") -> dict[str, Any]:
    return {
        "status": "error",
        "answer": "",
        "sql": sql,
        "columns": [],
        "rows": [],
        "error": {"code": code},
    }


def _answer_prompt(
    question: str,
    program: str | None,
    columns: list[str],
    rows: list[dict[str, Any]],
) -> str:
    bounded_rows = []
    for row in rows[:_MAX_PROMPT_ROWS]:
        bounded_rows.append(
            {
                key: value[:_MAX_PROMPT_CELL_CHARS] + "…"
                if isinstance(value, str) and len(value) > _MAX_PROMPT_CELL_CHARS
                else value
                for key, value in row.items()
            }
        )
    payload = {
        "question": question,
        "selected_program": program,
        "columns": columns,
        "rows": bounded_rows,
    }
    return (
        "Matching database rows exist. MUST answer from the returned rows. "
        "MUST NOT say that no data was found. "
        "MUST NOT invent facts outside the returned rows.\n"
        "ตอบคำถามเป็นภาษาไทยโดยใช้เฉพาะข้อมูล rows ที่ให้มาเท่านั้น "
        "ห้ามอนุมานหรือแต่งข้อเท็จจริงที่ไม่มีใน rows และห้ามกล่าวว่าไม่พบข้อมูลเมื่อมี rows\n"
        "ข้อมูลสำหรับตอบ (JSON):\n"
        + json.dumps(payload, ensure_ascii=False, allow_nan=False)
    )


def _is_no_data_answer(answer: str) -> bool:
    normalized = "".join(answer.casefold().split())
    return any(
        "".join(phrase.casefold().split()) in normalized
        for phrase in _NO_DATA_ANSWER_PHRASES
    )


def _grounded_rows_fallback(
    columns: list[str], rows: list[dict[str, Any]]
) -> str:
    visible_rows = rows[:_MAX_PROMPT_ROWS]
    lines = []
    for row in visible_rows:
        fields = [
            f"{column}: {json.dumps(row.get(column), ensure_ascii=False, allow_nan=False)}"
            for column in columns
            if column in row
        ]
        lines.append(" — ".join(fields))

    if len(rows) > len(visible_rows):
        heading = f"พบข้อมูล {len(rows)} รายการ (แสดง {len(visible_rows)} รายการแรก):"
    else:
        heading = f"พบข้อมูล {len(rows)} รายการ:"
    return heading + "\n" + "\n".join(lines)


def ask_sql(
    db_path: str | Path,
    question: str,
    program: str | None,
    sql_model_callable: Callable[[str], str],
    answer_model_callable: Callable[[str], str],
    conversation_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate, guard, and execute one curriculum query, then summarize its rows."""
    if not isinstance(question, str) or not question.strip():
        return _failure("invalid_request")
    if program is not None and (not isinstance(program, str) or not program.strip()):
        return _failure("invalid_request")
    if not callable(sql_model_callable) or not callable(answer_model_callable):
        return _failure("invalid_request")

    selected_program = program.strip() if program is not None else None
    if conversation_context is not None and (
        not isinstance(conversation_context, dict)
        or set(conversation_context)
        - {"focus_course", "result_courses", "result_scope_program", "result_set_empty"}
    ):
        return _failure("invalid_context")
    if conversation_context is not None and (
        "result_scope_program" in conversation_context
        and "result_courses" not in conversation_context
    ):
        return _failure("invalid_context")
    if conversation_context is not None and (
        "result_set_empty" in conversation_context
        and (
            conversation_context.get("result_set_empty") is not True
            or "result_courses" not in conversation_context
            or "result_scope_program" not in conversation_context
        )
    ):
        return _failure("invalid_context")
    try:
        prior_focus = parse_focus_course_context(
            conversation_context.get("focus_course")
            if conversation_context is not None
            else None,
            selected_program,
        )
        parsed_results = parse_result_courses_context(
            conversation_context.get("result_courses")
            if conversation_context is not None
            else None,
            conversation_context.get("result_scope_program", selected_program)
            if conversation_context is not None
            else selected_program,
            selected_program,
            conversation_context.get("result_set_empty", False)
            if conversation_context is not None
            else False,
        )
    except (TypeError, ValueError):
        return _failure("invalid_context")
    prior_result_courses = parsed_results[0] if parsed_results is not None else None

    query_spec = parse_query_spec(question.strip())
    has_explicit_override = _explicit_course_or_plan_override(query_spec)
    use_result_set_scope = (
        prior_result_courses is not None
        and query_spec.references_previous_result_set
        and not has_explicit_override
    )
    use_focus_scope = (
        prior_focus is not None
        and _references_focus_course(question.strip())
        and not has_explicit_override
    )
    if use_focus_scope and (
        prior_focus is None
        or not isinstance(prior_focus.get("program"), str)
        or not prior_focus["program"].strip()
    ):
        return _failure("invalid_context")
    course_scope = None
    if use_result_set_scope and prior_result_courses is not None:
        course_scope = tuple(
            (
                course.get("catalog_key"),
                str(course["program"]),
                str(course["course_code"]),
            )
            for course in prior_result_courses
        )
    elif use_focus_scope and prior_focus is not None:
        course_scope = (
            (
                prior_focus.get("catalog_key"),
                prior_focus["program"],
                prior_focus["course_code"],
            ),
        )
    if (
        prior_result_courses == []
        and query_spec.references_previous_result_set
        and not _has_explicit_current_scope(query_spec)
    ):
        return {
            "status": "no_data",
            "answer": "จากรายการก่อนหน้า ไม่พบรายการที่ตรงกับเงื่อนไขนี้",
            "sql": None,
            "columns": [],
            "rows": [],
            "next_context": _next_conversation_context(
                selected_program, prior_focus, prior_result_courses, []
            ),
        }

    explicit_course_codes = {
        _normalize_course_code(code)
        for code in query_spec.course_codes
    }
    context_course_codes = set()
    if prior_focus is not None:
        context_course_codes.add(
            _normalize_course_code(prior_focus["course_code"])
        )
    if prior_result_courses is not None:
        context_course_codes.update(
            _normalize_course_code(course["course_code"])
            for course in prior_result_courses
        )
    allowed_course_codes = explicit_course_codes or context_course_codes

    generation_question = question.strip()
    if selected_program is not None:
        generation_question += (
            f"\nSelected program: {selected_program}. "
            "This is the active curriculum scope and must constrain the SQL."
        )

    def call_sql_model(prompt: str) -> str:
        try:
            generated = sql_model_callable(
                f"{prompt}\n\n{_PLAN_SCOPE_GUIDANCE}\n\n{_PROGRAM_SCOPE_GUIDANCE}\n\n"
                f"{_UNIQUE_COLUMN_GUIDANCE}\n\n"
                f"{_CATALOG_CONTEXT_GUIDANCE}\n\n"
                f"{_EVIDENCE_PROJECTION_GUIDANCE}\n\n"
                f"{_COMPARISON_COMPLETENESS_GUIDANCE}\n\n"
                f"{_LOGICAL_COURSE_COUNTING_GUIDANCE}\n\n"
                f"{_conversation_focus_guidance(prior_focus)}"
                f"{_conversation_result_set_guidance(prior_result_courses, selected_program)}"
            )
        except Exception as exc:
            raise _SqlModelFailure from exc
        if not isinstance(generated, str):
            raise _SqlModelFailure("SQL model must return text")
        return generated

    repair_attempted = False

    def repair_generated_sql(
        failed_sql: str,
        validation_error: str,
    ) -> tuple[str | None, dict[str, Any] | None]:
        nonlocal repair_attempted
        if repair_attempted:
            return None, _failure("invalid_sql", failed_sql)
        repair_attempted = True
        try:
            repaired_sql = repair_sql(
                generation_question,
                schema,
                failed_sql,
                validation_error,
                call_sql_model,
            )
        except _SqlModelFailure:
            return None, _failure("sql_model_failure", failed_sql)
        except ValueError as repair_error:
            code = (
                "disallowed_relation"
                if "relation is not allowed" in str(repair_error)
                else "invalid_sql"
            )
            return None, _failure(code, failed_sql)
        except Exception:
            return None, _failure("invalid_sql", failed_sql)

        try:
            safe_repaired_sql = guard_sql(
                repaired_sql,
                max_limit=_MAX_RESULT_ROWS,
                allowed_relations=_ALLOWED_RELATIONS,
            )
        except ValueError as repair_error:
            code = (
                "disallowed_relation"
                if "relation is not allowed" in str(repair_error)
                else "invalid_sql"
            )
            return None, _failure(code, repaired_sql)

        unsupported_literal = _unsupported_course_code_literal(
            safe_repaired_sql,
            allowed_course_codes,
        )
        if unsupported_literal is not None:
            return None, _failure("unsupported_course_code_literal", safe_repaired_sql)
        return safe_repaired_sql, None

    try:
        schema = _canonical_schema()
        generated_sql = question_to_sql(
            generation_question,
            schema,
            call_sql_model,
        )
    except _SqlModelFailure:
        return _failure("sql_model_failure")
    except (OSError, RuntimeError):
        return _failure("schema_unavailable")
    except Exception:
        return _failure("invalid_sql")

    try:
        safe_sql = guard_sql(
            generated_sql,
            max_limit=_MAX_RESULT_ROWS,
            allowed_relations=_ALLOWED_RELATIONS,
        )
    except ValueError as exc:
        code = "disallowed_relation" if "relation is not allowed" in str(exc) else "invalid_sql"
        return _failure(code, generated_sql)

    unsupported_literal = _unsupported_course_code_literal(
        safe_sql,
        allowed_course_codes,
    )
    if unsupported_literal is not None:
        sanitized_reason = (
            "Generated SQL contains unsupported course-code literal "
            f"{json.dumps(unsupported_literal[:_MAX_CONTEXT_CODE_CHARS])}. "
            "Course-code predicates may use only codes explicitly present in the "
            "user question or structured conversation context."
        )
        safe_sql, failure = repair_generated_sql(safe_sql, sanitized_reason)
        if failure is not None:
            return failure

    try:
        if course_scope is None:
            columns, raw_rows = execute_readonly(db_path, safe_sql)
        else:
            columns, raw_rows = execute_readonly(
                db_path, safe_sql, course_scope=course_scope
            )
    except sqlite3.OperationalError as exc:
        if repair_attempted:
            return _failure("sqlite_error", safe_sql)
        repair_reason = _repairable_sqlite_error(exc)
        if repair_reason is None:
            return _failure("sqlite_error", safe_sql)
        safe_sql, failure = repair_generated_sql(safe_sql, repair_reason)
        if failure is not None:
            return failure

        try:
            if course_scope is None:
                columns, raw_rows = execute_readonly(db_path, safe_sql)
            else:
                columns, raw_rows = execute_readonly(
                    db_path, safe_sql, course_scope=course_scope
                )
        except ValueError:
            return _failure("invalid_context", safe_sql)
        except (sqlite3.Error, OSError):
            return _failure("sqlite_error", safe_sql)
    except ValueError:
        return _failure("invalid_context", safe_sql)
    except (sqlite3.Error, OSError):
        return _failure("sqlite_error", safe_sql)

    if len(columns) != len(set(columns)):
        return _failure("invalid_result", safe_sql)
    try:
        rows = [
            {column: _json_safe(value) for column, value in zip(columns, raw_row)}
            for raw_row in raw_rows
        ]
    except (TypeError, ValueError):
        return _failure("invalid_result", safe_sql)

    if not rows:
        course_identity_projection = (
            "course_code" in columns
            and any(name in columns for name in ("course_name", "name_en", "name_th", "course_title"))
        )
        establish_empty_result_set = (
            prior_result_courses is None
            and prior_focus is None
            and not query_spec.course_codes
            and query_spec.operations == ("list",)
            and course_identity_projection
        )
        return {
            "status": "no_data",
            "answer": (
                "จากรายการก่อนหน้า ไม่พบรายการที่ตรงกับเงื่อนไขนี้"
                if prior_result_courses is not None
                else "ไม่พบข้อมูลที่ตรงกับคำถาม"
            ),
            "sql": safe_sql,
            "columns": columns,
            "rows": [],
            "next_context": _next_conversation_context(
                selected_program,
                prior_focus,
                prior_result_courses,
                [],
                establish_empty_result_set=establish_empty_result_set,
            ),
        }

    try:
        answer = answer_model_callable(
            _answer_prompt(question.strip(), selected_program, columns, rows)
        )
        if not isinstance(answer, str) or not answer.strip():
            raise TypeError("answer model must return non-empty text")
    except Exception:
        return _failure("answer_model_failure", safe_sql)

    return {
        "status": "answer",
        "answer": (
            _grounded_rows_fallback(columns, rows)
            if _is_no_data_answer(answer.strip())
            else answer.strip()
        ),
        "sql": safe_sql,
        "columns": columns,
        "rows": rows,
        "next_context": _next_conversation_context(
            selected_program,
            prior_focus,
            prior_result_courses,
            rows,
        ),
    }


__all__ = [
    "ask_sql",
    "parse_focus_course_context",
    "parse_result_courses_context",
]
