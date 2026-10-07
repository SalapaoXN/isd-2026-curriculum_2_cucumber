"""Standalone, guarded LLM-to-SQL curriculum QA service."""

from __future__ import annotations

import base64
from collections.abc import Mapping
from dataclasses import dataclass, replace
import json
import math
from contextlib import closing
from pathlib import Path
import re
import sqlite3
from typing import Any, Callable

from rag.structured.execute import execute_readonly
from rag.structured.guard_sql import (
    extract_column_predicate_literals,
    guard_sql,
)
from rag.structured.provenance import hydrate_sql_row_provenance
from rag.structured.aggregate_provenance import verify_aggregate_evidence
from rag.structured.nl_to_sql import question_to_sql, repair_sql
from rag.query_spec import QuerySpec, _COURSE_CODE_PATTERN, parse_query_spec
from rag.resolution import (
    QueryContext,
    has_answerable_target_or_scope,
    resolve_ordinal_course_reference,
)
from rag.structured.queries import (
    catalog_keys_for_program,
    edition_catalog_keys_for_program,
    exact_course_candidates,
)


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
_PLAN_COURSES_VIEW_SEMANTICS = (
    "Canonical view semantics for v_plan_courses:\n"
    "- v_plan_courses has no catalog_id column. Never reference "
    "v_plan_courses.catalog_id.\n"
    "- When selected_catalog_key is present, the execution environment already "
    "restricts canonical relations to that catalog. Never query outside the "
    "scoped relation set or add catalog predicates to relations that do not "
    "expose catalog identity.\n"
    "- To project catalog_key, join v_plan_courses.course_id to "
    "courses.course_id, then courses.catalog_id to catalogs.catalog_id."
)
_SELECTED_CATALOG_SQL_GUIDANCE = (
    "Selected-catalog execution contract:\n"
    "- selected_catalog_key is the authoritative application scope. The "
    "execution environment already restricts canonical relations to it.\n"
    "- Never query outside the scoped relation set or invent catalog predicates "
    "on relations that do not expose catalog identity."
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
    "- For an unscoped title or code lookup that may match across programs or catalogs, "
    "preserve all canonical matches across programs and catalogs. When execution scope "
    "is supplied, preserve all matches within that scope. Do not use LIMIT 1 to choose "
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
_RESCUE_EVIDENCE_ID_GUIDANCE = (
    "Rescue evidence-ID projection (applies only to this opt-in request):\n"
    "Canonical provenance is derived after SQL execution by Python code, never by "
    "this SQL. When selecting row-level factual entities, also project the "
    "corresponding canonical evidence ID alongside the answer-relevant fields:\n"
    "- row-level course facts: project course_id.\n"
    "- row-level plan-placement facts: project placement_id (and course_id where "
    "the row also carries course identity).\n"
    "- row-level prerequisite relationship facts: prerequisite_id is the primary "
    "provenance identity for the relationship row; any extra course IDs must use "
    "unique aliases and are not provenance identities.\n"
    "- row-level program requirement facts: project requirement_id.\n"
    "- row-level policy facts: project fact_id.\n"
    "- row-level alternative-group facts where directly queried: project "
    "alternative_group_id.\n"
    "- Project only the IDs of the factual entities the row actually reports; do "
    "not add unrelated IDs merely to satisfy provenance.\n"
    "- Do not query provenance or *_provenance tables for answer citation purposes, "
    "and do not SELECT source filenames, source pages, or provenance fields. "
    "Python hydration remains the citation authority.\n"
    "- Do not fabricate IDs. Never invent an entity ID value.\n"
    "- Output columns must remain uniquely named.\n"
    "- Do not change the result grain merely to expose IDs: do not add IDs to a "
    "SELECT DISTINCT list when that would alter logical deduplication, do not "
    "change GROUP BY grouping, and do not alter ordering meaning.\n"
    "- Aggregate rows without one row-level canonical entity (for example "
    "total_credits from v_semester_credits) may omit evidence IDs; such rows "
    "intentionally remain non-rescuable and must not receive fake entity IDs."
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
    value: Any,
    selected_program: str | None,
    selected_catalog_key: str | None = None,
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
    if selected_catalog_key is not None and (
        catalog_key is None
        or catalog_key.strip().casefold() != selected_catalog_key.casefold()
    ):
        return None
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
    selected_catalog_key: str | None = None,
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
        if selected_catalog_key is not None and (
            catalog_key is None
            or catalog_key.casefold() != selected_catalog_key.casefold()
        ):
            return None
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


def _canonical_identities_for_named_codes(
    named: list[str],
    selected_program: str | None,
    selected_catalog_key: str | None,
    db_path: str | Path,
) -> tuple[list[dict[str, str | None]], set[str | None]]:
    """Validate answer-named codes directly against canonical course identity.

    Fallback only, used when model-generated SQL rows disagree with the
    grounded answer. Each named code must resolve to exactly one canonical
    candidate inside the active program/catalog scope; unknown, ambiguous,
    or out-of-scope codes are dropped. Mention order is preserved. Identity
    fields only; never facts.
    """
    items: list[dict[str, str | None]] = []
    catalogs: set[str | None] = set()
    try:
        for code in named[:_MAX_CONTEXT_COURSES]:
            candidates = exact_course_candidates(
                db_path,
                course_code=code,
                program=selected_program,
                catalog_key=selected_catalog_key,
            )
            if len(candidates) != 1:
                continue
            raw_program = candidates[0].get("program")
            if not isinstance(raw_program, str) or not raw_program.strip():
                continue
            catalog_key: str | None = (
                selected_catalog_key.strip()
                if isinstance(selected_catalog_key, str)
                and selected_catalog_key.strip()
                else None
            )
            catalogs.add(
                catalog_key.casefold() if catalog_key is not None else None
            )
            item: dict[str, str | None] = {
                "program": raw_program.strip(),
                "course_code": code,
            }
            if catalog_key is not None:
                item["catalog_key"] = catalog_key
            items.append(item)
    except (FileNotFoundError, OSError, sqlite3.Error, TypeError, ValueError):
        return [], set()
    return items, catalogs


def _retained_identities_from_answer(
    answer: Any,
    rows: list[dict[str, Any]],
    selected_program: str | None,
    selected_catalog_key: str | None,
    semantic_topic: str | None = None,
    db_path: str | Path | None = None,
) -> dict[str, Any] | None:
    """Retain only canonical identities the grounded answer actually named.

    Ordinal references (ตัวแรก/ตัวที่ N) resolve against courses named in the
    answer prose, in mention order, restricted to validated canonical row
    identities. When model-generated SQL rows are narrower than (or disagree
    with) the grounded answer, answer-named codes are additionally validated
    directly against canonical relational course identity inside the active
    scope; SQL-row agreement stays preferred and runs first. Single-course
    answers stay owned by the focus machinery; answers naming nothing retain
    nothing. Truncated to the context bound; out-of-range ordinals fail
    closed downstream. Never retains facts.
    """
    if not isinstance(answer, str) or not answer.strip():
        return None
    named: list[str] = []
    for match in _COURSE_CODE_PATTERN.finditer(answer):
        code = match.group(1)
        if code not in named:
            named.append(code)
    if len(named) < 2:
        return None
    by_code: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        raw_code = row.get("course_code")
        if not isinstance(raw_code, (str, int)) or isinstance(raw_code, bool):
            continue
        key = str(raw_code).strip()
        if key and key not in by_code:
            by_code[key] = row
    items: list[dict[str, str | None]] = []
    catalogs: set[str | None] = set()
    for code in named[:_MAX_CONTEXT_COURSES]:
        row = by_code.get(code)
        if row is None:
            continue
        raw_program = row.get("program", row.get("program_code"))
        if not isinstance(raw_program, str) or not raw_program.strip():
            continue
        raw_catalog = row.get("catalog_key")
        if isinstance(raw_catalog, str) and raw_catalog.strip():
            catalog_key: str | None = raw_catalog.strip()
        elif selected_catalog_key is not None:
            # Rows come from execution already scoped to the selected catalog.
            catalog_key = selected_catalog_key.strip()
        else:
            catalog_key = None
        catalogs.add(catalog_key.casefold() if catalog_key is not None else None)
        item: dict[str, str | None] = {
            "program": raw_program.strip(),
            "course_code": code,
        }
        if catalog_key is not None:
            item["catalog_key"] = catalog_key
        items.append(item)
    if len(items) < 2 and db_path is not None:
        # SQL-row agreement is preferred; only when it is insufficient,
        # validate the answer-named codes directly against canonical
        # relational identity (live SQL rows may be narrower than the
        # grounded answer). Scope coherence is enforced by the exact
        # candidate lookup and the result-context validation below.
        items, catalogs = _canonical_identities_for_named_codes(
            named, selected_program, selected_catalog_key, db_path
        )
    if len(items) < 2:
        return None
    if len({catalog for catalog in catalogs if catalog is not None}) > 1:
        return None
    try:
        parsed = parse_result_courses_context(
            items, selected_program, selected_program, False, selected_catalog_key
        )
    except (TypeError, ValueError):
        return None
    if parsed is None:
        return None
    retained: dict[str, Any] = {}
    if selected_program is not None:
        retained["program"] = selected_program
    if selected_catalog_key is not None:
        retained["catalog_key"] = selected_catalog_key
    if semantic_topic is not None:
        retained["semantic_topic"] = semantic_topic
    retained["result_courses"] = parsed[0]
    retained["result_scope_program"] = parsed[1]
    return retained


def _next_conversation_context(
    selected_program: str | None,
    prior_focus: dict[str, str | None] | None,
    prior_result_courses: list[dict[str, str | None]] | None,
    rows: list[dict[str, Any]],
    *,
    establish_empty_result_set: bool = False,
    selected_catalog_key: str | None = None,
    current_spec: QuerySpec | None = None,
    prior_scope: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    result_courses, over_bound = _result_courses_from_rows(rows, selected_program)
    context: dict[str, Any] = {}
    if selected_program is not None:
        context["program"] = selected_program
    if selected_catalog_key is not None:
        context["catalog_key"] = selected_catalog_key
    focus_values: dict[str, tuple[int, ...]] = {}
    for axis, row_key, max_value in (
        ("years", "year", 5),
        ("semesters", "semester", 2),
    ):
        explicit = tuple(getattr(current_spec, axis, ())) if current_spec is not None else ()
        row_values = {
            row.get(row_key)
            for row in rows
            if isinstance(row.get(row_key), int)
            and not isinstance(row.get(row_key), bool)
        }
        if len(explicit) == 1 and 1 <= explicit[0] <= max_value:
            focus_values[axis] = explicit
        elif len(row_values) == 1:
            focus_values[axis] = (next(iter(row_values)),)
        elif prior_scope is not None:
            prior_values = tuple(prior_scope.get(axis, ()))
            if len(prior_values) == 1 and 1 <= prior_values[0] <= max_value:
                focus_values[axis] = prior_values
    plans = tuple(getattr(current_spec, "plans", ())) if current_spec is not None else ()
    row_plans = {
        row.get("plan") for row in rows
        if isinstance(row.get("plan"), str) and row.get("plan").strip()
    }
    selected_plan = plans[0] if len(plans) == 1 else None
    if selected_plan is None and len(row_plans) == 1:
        selected_plan = next(iter(row_plans))
    if selected_plan is None and prior_scope is not None:
        raw_plan = prior_scope.get("plan")
        selected_plan = raw_plan if isinstance(raw_plan, str) and raw_plan.strip() else None
    if selected_plan is not None:
        context["plan"] = selected_plan
    operations = tuple(getattr(current_spec, "operations", ())) if current_spec is not None else ()
    if not operations and prior_scope is not None:
        operations = tuple(prior_scope.get("operations", ()))
    if operations and operations != ("identity",):
        context["operations"] = list(operations)
    context.update({axis: list(values) for axis, values in focus_values.items()})
    if focus_values and selected_catalog_key is not None:
        context["focus_catalog_key"] = selected_catalog_key
    if over_bound:
        return context or None
    if result_courses is not None:
        if selected_catalog_key is not None:
            if any(
                course.get("catalog_key") is not None
                and str(course["catalog_key"]).strip().casefold()
                != selected_catalog_key.casefold()
                for course in result_courses
            ):
                return context or None
            for course in result_courses:
                course["catalog_key"] = selected_catalog_key
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
        for anchor in ("ตัวนี้", "ตัวนั้น", "วิชานี้", "อันนี้")
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
    raw_message = str(error)
    message = raw_message.casefold()
    if "no such column:" in message:
        missing_column = re.search(
            r"no such column:\s*(.*?)\s*$", raw_message, re.IGNORECASE
        )
        identifier = missing_column.group(1).strip() if missing_column else ""
        if re.fullmatch(
            r"(?:[A-Za-z_][A-Za-z0-9_]{0,63}\.)?[A-Za-z_][A-Za-z0-9_]{0,63}",
            identifier,
        ):
            return (
                "SQLite schema validation error: column "
                f"{json.dumps(identifier)} does not exist in the supplied schema. "
                "Repair the query using only columns exposed by the supplied relations."
            )
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
    return "\n\n".join(
        (*selected, _SEMESTER_CREDIT_VIEW_SEMANTICS, _PLAN_COURSES_VIEW_SEMANTICS)
    )


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


@dataclass(frozen=True, slots=True)
class RescueAdmission:
    """Outcome of the deterministic rescue admission gate (HSQL-5A)."""

    allowed: bool
    reason: str


_CURRICULUM_ID_COLUMNS = (
    "course_id",
    "placement_id",
    "prerequisite_id",
    "requirement_id",
    "alternative_group_id",
)
_POLICY_ID_COLUMNS = ("fact_id",)


def _is_canonical_id(value: Any) -> bool:
    """Mirror the HSQL-1 identity shape without resolving anything."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _rescue_admission(
    *,
    query_spec: QuerySpec,
    question: str,
    selected_program: str | None,
    selected_catalog_key: str | None,
    focus_scope: dict[str, Any],
    prior_focus: dict[str, str | None] | None,
    prior_result_courses: list[dict[str, str | None]] | None,
    rows: list[dict[str, Any]],
) -> RescueAdmission:
    """Decide whether SQL rows may even be considered for rescue.

    Groundable rows are necessary but not sufficient: the current turn must
    also carry a legitimate bounded request scope. Curriculum-backed rows
    need a validated curriculum anchor plus request substance; rows backed
    only by global policy facts use a narrow bounded exception. Runs before
    provenance hydration so inadmissible requests never touch canonical
    provenance tables.
    """
    has_curriculum_identity = False
    all_fact_backed = True
    for row in rows:
        if not isinstance(row, Mapping):
            all_fact_backed = False
            continue
        if any(
            _is_canonical_id(row.get(column)) for column in _CURRICULUM_ID_COLUMNS
        ):
            has_curriculum_identity = True
        elif not any(
            _is_canonical_id(row.get(column)) for column in _POLICY_ID_COLUMNS
        ):
            all_fact_backed = False
    if not has_curriculum_identity and not all_fact_backed:
        if (
            len(rows) == 1
            and isinstance(rows[0], Mapping)
            and "total_credits" in rows[0]
        ):
            families_aggregate_candidate = True
        else:
            return RescueAdmission(False, "no-supported-row-identity")
    else:
        families_aggregate_candidate = False

    if query_spec.judgement not in (None, "none"):
        return RescueAdmission(False, "unsupported-judgement-shape")

    if not has_curriculum_identity and all_fact_backed:
        if len(rows) > _MAX_PROMPT_ROWS:
            return RescueAdmission(False, "policy-row-bound-exceeded")
        return RescueAdmission(True, "global-policy-facts")

    if families_aggregate_candidate:
        # An ID-less single total row is only a candidate: the aggregate
        # verifier still revalidates scope, intent, value equality, and
        # provenance. Admission here requires just the curriculum anchor.
        plan = focus_scope.get("plan") if isinstance(focus_scope, dict) else None
        has_anchor = bool(
            (isinstance(selected_program, str) and selected_program.strip())
            or (isinstance(selected_catalog_key, str) and selected_catalog_key.strip())
            or (isinstance(plan, str) and plan.strip())
            or prior_focus is not None
            or prior_result_courses is not None
            or bool(query_spec.program)
            or bool(query_spec.course_codes)
            or query_spec.course_name is not None
        )
        if not has_anchor:
            return RescueAdmission(False, "no-curriculum-anchor")
        return RescueAdmission(True, "aggregate-candidate")

    plan = focus_scope.get("plan") if isinstance(focus_scope, dict) else None
    has_anchor = bool(
        (isinstance(selected_program, str) and selected_program.strip())
        or (isinstance(selected_catalog_key, str) and selected_catalog_key.strip())
        or (isinstance(plan, str) and plan.strip())
        or prior_focus is not None
        or prior_result_courses is not None
        or bool(query_spec.program)
        or bool(query_spec.course_codes)
        or query_spec.course_name is not None
    )
    if not has_anchor:
        return RescueAdmission(False, "no-curriculum-anchor")

    has_substance = bool(
        query_spec.operations
        or query_spec.course_codes
        or query_spec.course_name is not None
        or query_spec.plans
        or query_spec.years
        or query_spec.semesters
        or query_spec.category is not None
        or query_spec.references_previous_result_set
        or query_spec.result_ordinal is not None
        or query_spec.credit_units is not None
        or query_spec.group_by
        or (has_anchor and _references_focus_course(question))
    )
    if not has_substance:
        return RescueAdmission(False, "no-bounded-request")
    return RescueAdmission(True, "scoped-curriculum-request")


def _grounded_row_rescue(
    db_path: str | Path,
    question: str,
    selected_program: str | None,
    columns: list[str],
    rows: list[dict[str, Any]],
    safe_sql: str,
    answer_model_callable: Callable[[str], str],
    prior_focus: dict[str, str | None] | None,
    prior_result_courses: list[dict[str, str | None]] | None,
    selected_catalog_key: str | None,
    query_spec: QuerySpec,
    focus_scope: dict[str, Any],
) -> dict[str, Any] | None:
    """Attempt the opt-in SQL-row rescue after deterministic grounding failed.

    Returns a row-grounded answer only when every executed row resolves to
    non-empty canonical provenance; otherwise returns None so the caller
    keeps its existing fail-closed result. Partial hydrated provenance is
    never surfaced. When normal row hydration cannot cover ID-less aggregate
    rows, a supported aggregate verifier is tried secondarily under the same
    admission gate.
    """
    admission = _rescue_admission(
        query_spec=query_spec,
        question=question,
        selected_program=selected_program,
        selected_catalog_key=selected_catalog_key,
        focus_scope=focus_scope,
        prior_focus=prior_focus,
        prior_result_courses=prior_result_courses,
        rows=rows,
    )
    if not admission.allowed:
        return None
    provenance: list[dict[str, Any]] = []
    try:
        hydration = hydrate_sql_row_provenance(db_path, rows)
    except (OSError, sqlite3.Error, ValueError, TypeError):
        return None
    if (
        hydration.status == "complete"
        and hydration.covered_rows == hydration.total_rows
        and hydration.covered_rows == len(rows)
        and hydration.provenance
    ):
        provenance = [dict(reference) for reference in hydration.provenance]
    else:
        try:
            plan = focus_scope.get("plan") if isinstance(focus_scope, dict) else None
            aggregate = verify_aggregate_evidence(
                db_path,
                rows=rows,
                columns=columns,
                program=selected_program,
                catalog_key=selected_catalog_key,
                plan=plan if isinstance(plan, str) else None,
                years=tuple(query_spec.years),
                semesters=tuple(query_spec.semesters),
                query_spec=query_spec,
            )
        except (OSError, sqlite3.Error, ValueError, TypeError):
            return None
        if aggregate.status != "complete" or not aggregate.provenance:
            return None
        provenance = [dict(reference) for reference in aggregate.provenance]
    try:
        answer = answer_model_callable(
            _answer_prompt(question, selected_program, columns, rows)
        )
        if not isinstance(answer, str) or not answer.strip():
            raise TypeError("answer model must return non-empty text")
    except Exception:
        return None
    return {
        "status": "answer",
        "answer": (
            _grounded_rows_fallback(columns, rows)
            if _is_no_data_answer(answer.strip())
            else answer.strip()
        ),
        "provenance": provenance,
        "sql": safe_sql,
        "columns": columns,
        "rows": rows,
        "next_context": _next_conversation_context(
            selected_program,
            prior_focus,
            prior_result_courses,
            rows,
            selected_catalog_key=selected_catalog_key,
            current_spec=query_spec,
            prior_scope=focus_scope,
        ),
    }


def ask_sql(
    db_path: str | Path,
    question: str,
    program: str | None,
    sql_model_callable: Callable[[str], str],
    answer_model_callable: Callable[[str], str],
    conversation_context: dict[str, Any] | None = None,
    *,
    grounding_callable: Callable[[str], dict[str, Any]] | None = None,
    allow_grounded_row_rescue: bool = False,
) -> dict[str, Any]:
    """Generate, guard, and execute one curriculum query, then summarize its rows."""
    if not isinstance(question, str) or not question.strip():
        return _failure("invalid_request")
    if program is not None and (not isinstance(program, str) or not program.strip()):
        return _failure("invalid_request")
    if not callable(sql_model_callable) or not callable(answer_model_callable):
        return _failure("invalid_request")
    if not isinstance(allow_grounded_row_rescue, bool):
        return _failure("invalid_request")

    selected_program = program.strip() if program is not None else None
    if conversation_context is not None and (
        not isinstance(conversation_context, dict)
        or set(conversation_context)
        - {
            "program", "catalog_key", "focus_catalog_key", "plan", "years", "semesters",
            "focus_course", "result_courses", "result_scope_program", "result_set_empty",
            "operations", "semantic_topic",
        }
    ):
        return _failure("invalid_context")
    selected_catalog_key = None
    context_program = (
        conversation_context.get("program")
        if conversation_context is not None
        else None
    )
    if context_program is not None and (
        not isinstance(context_program, str)
        or not context_program.strip()
        or (selected_program is not None
            and context_program.strip().casefold() != selected_program.strip().casefold())
    ):
        return _failure("invalid_context")
    if selected_program is None and isinstance(context_program, str):
        selected_program = context_program.strip()
    if conversation_context is not None and "catalog_key" in conversation_context:
        raw_catalog_key = conversation_context["catalog_key"]
        if (
            not isinstance(raw_catalog_key, str)
            or not raw_catalog_key.strip()
            or len(raw_catalog_key.strip()) > 128
        ):
            return _failure("invalid_context")
        try:
            database_uri = f"{Path(db_path).resolve().as_uri()}?mode=ro"
            with closing(sqlite3.connect(database_uri, uri=True)) as connection:
                catalog_rows = connection.execute(
                    "SELECT catalog_key FROM catalogs "
                    "WHERE lower(trim(catalog_key)) = ?",
                    (raw_catalog_key.strip().casefold(),),
                ).fetchall()
        except sqlite3.Error:
            return _failure("invalid_context")
        if len(catalog_rows) != 1 or not catalog_rows[0][0]:
            return _failure("invalid_context")
        selected_catalog_key = str(catalog_rows[0][0]).strip()
    if selected_catalog_key is None and conversation_context is not None:
        contextual_keys: set[str] = set()
        raw_focus_catalog = conversation_context.get("focus_catalog_key")
        if isinstance(raw_focus_catalog, str) and raw_focus_catalog.strip():
            contextual_keys.add(raw_focus_catalog.strip())
        raw_focus = conversation_context.get("focus_course")
        if isinstance(raw_focus, dict) and isinstance(raw_focus.get("catalog_key"), str):
            contextual_keys.add(raw_focus["catalog_key"].strip())
        raw_results = conversation_context.get("result_courses")
        if isinstance(raw_results, (list, tuple)) and raw_results:
            if all(
                isinstance(item, dict)
                and isinstance(item.get("catalog_key"), str)
                and item["catalog_key"].strip()
                for item in raw_results
            ):
                contextual_keys.update(item["catalog_key"].strip() for item in raw_results)
        if len(contextual_keys) == 1:
            contextual_key = next(iter(contextual_keys))
            try:
                database_uri = f"{Path(db_path).resolve().as_uri()}?mode=ro"
                with closing(sqlite3.connect(database_uri, uri=True)) as connection:
                    catalog_rows = connection.execute(
                        "SELECT catalog_key FROM catalogs "
                        "WHERE lower(trim(catalog_key)) = ?",
                        (contextual_key.casefold(),),
                    ).fetchall()
            except sqlite3.Error:
                return _failure("invalid_context")
            if len(catalog_rows) == 1 and isinstance(catalog_rows[0][0], str):
                selected_catalog_key = catalog_rows[0][0].strip()
    query_spec = parse_query_spec(question.strip())
    scope_program = selected_program or query_spec.program
    prior_scope: dict[str, Any] = {}
    focus_catalog_key = (
        conversation_context.get("focus_catalog_key")
        if conversation_context is not None
        else None
    )
    if focus_catalog_key is not None and (
        not isinstance(focus_catalog_key, str)
        or not focus_catalog_key.strip()
        or len(focus_catalog_key.strip()) > 128
    ):
        return _failure("invalid_context")
    focus_is_stale = (
        isinstance(focus_catalog_key, str)
        and selected_catalog_key is not None
        and focus_catalog_key.strip().casefold() != selected_catalog_key.casefold()
    )
    if not focus_is_stale and selected_catalog_key is not None and conversation_context is not None:
        prior_keys: set[str] = set()
        raw_focus_course = conversation_context.get("focus_course")
        if isinstance(raw_focus_course, dict) and isinstance(
            raw_focus_course.get("catalog_key"), str
        ):
            prior_keys.add(raw_focus_course["catalog_key"].strip())
        raw_result_courses = conversation_context.get("result_courses")
        if isinstance(raw_result_courses, (list, tuple)):
            prior_keys.update(
                item["catalog_key"].strip()
                for item in raw_result_courses
                if isinstance(item, dict)
                and isinstance(item.get("catalog_key"), str)
                and item["catalog_key"].strip()
            )
        focus_is_stale = bool(
            prior_keys
            and any(key.casefold() != selected_catalog_key.casefold() for key in prior_keys)
        )
    if not focus_is_stale and conversation_context is not None:
        raw_plan = conversation_context.get("plan")
        if raw_plan is not None:
            if not isinstance(raw_plan, str) or raw_plan.strip().casefold() not in {
                "coop", "no_coop", "default", "gened"
            }:
                return _failure("invalid_context")
            prior_scope["plan"] = raw_plan.strip().casefold()
        for field_name, maximum in (("years", 5), ("semesters", 2)):
            raw_values = conversation_context.get(field_name, ())
            if not isinstance(raw_values, (list, tuple)):
                return _failure("invalid_context")
            values = tuple(raw_values)
            if any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 1 <= value <= maximum
                for value in values
            ):
                return _failure("invalid_context")
            if values:
                prior_scope[field_name] = values
        raw_operations = conversation_context.get("operations", ())
        allowed_operations = {
            "list", "program_discovery", "describe", "count", "sum_credits",
            "existence", "placement", "prerequisite", "similarity", "earliest",
            "compare", "identity",
        }
        if (
            not isinstance(raw_operations, (list, tuple))
            or any(
                not isinstance(operation, str)
                or operation not in allowed_operations
                for operation in raw_operations
            )
        ):
            return _failure("invalid_context")
        if raw_operations:
            prior_scope["operations"] = tuple(raw_operations)
    if scope_program is not None:
        try:
            program_catalogs = catalog_keys_for_program(db_path, scope_program)
            edition_catalogs = edition_catalog_keys_for_program(db_path, scope_program)
        except (FileNotFoundError, OSError, sqlite3.Error, TypeError, ValueError):
            program_catalogs = ()
            edition_catalogs = ()
        matching_catalog = next(
            (
                key for key in program_catalogs
                if isinstance(key, str)
                and selected_catalog_key is not None
                and key.casefold() == selected_catalog_key.casefold()
            ),
            None,
        )
        if selected_catalog_key is not None and matching_catalog is None:
            return _failure("invalid_context")
        if selected_catalog_key is None and edition_catalogs:
            keys = list(edition_catalogs)
            return {
                "status": "clarification_required",
                "action": "catalog_required",
                "answer": (
                    f"โปรดเลือกฉบับหลักสูตรของ {scope_program}: "
                    + ", ".join(keys)
                ),
                "catalog_keys": keys,
                "sql": None,
                "columns": [],
                "rows": [],
            }
        if selected_catalog_key is None and len(program_catalogs) == 1:
            only_catalog = program_catalogs[0]
            if isinstance(only_catalog, str) and only_catalog.strip():
                selected_catalog_key = only_catalog.strip()
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
        prior_focus = None if focus_is_stale else parse_focus_course_context(
            conversation_context.get("focus_course")
            if conversation_context is not None
            else None,
            selected_program,
            selected_catalog_key,
        )
        parsed_results = None if focus_is_stale else parse_result_courses_context(
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
            selected_catalog_key,
        )
    except (TypeError, ValueError):
        return _failure("invalid_context")
    prior_result_courses = parsed_results[0] if parsed_results is not None else None

    # Ordinal references (ตัวแรก/ตัวที่ N) resolve against the retained set
    # only; explicit codes/names win, everything else fails closed downstream.
    ordinal_target = resolve_ordinal_course_reference(
        query_spec, prior_result_courses
    )
    ordinal_code = (
        ordinal_target.get("course_code").strip()
        if isinstance(ordinal_target, dict)
        and isinstance(ordinal_target.get("course_code"), str)
        and ordinal_target.get("course_code").strip()
        else None
    )

    prior_focus_code = (
        prior_focus.get("course_code")
        if isinstance(prior_focus, dict)
        else None
    )
    if not isinstance(prior_focus_code, str) or not prior_focus_code.strip():
        prior_focus_code = None
    else:
        prior_focus_code = prior_focus_code.strip()
    raw_service_topic = (
        conversation_context.get("semantic_topic")
        if isinstance(conversation_context, dict)
        else None
    )
    if raw_service_topic is None:
        service_topic = None
    elif (
        not isinstance(raw_service_topic, str)
        or not raw_service_topic.strip()
        or len(raw_service_topic.strip()) > 80
    ):
        return _failure("invalid_context")
    else:
        service_topic = raw_service_topic.strip()
    has_validated_scope = bool(
        grounding_callable is not None
        and (
            selected_program
            or selected_catalog_key
            or prior_scope
            or prior_focus is not None
            or prior_result_courses is not None
            or service_topic is not None
        )
    )
    if has_validated_scope:
        # Parse with the same scope awareness as the grounded pipeline so the
        # shared eligibility verdict matches it exactly; without any scope at
        # all the service keeps its direct zero-scope contract.
        eligibility_spec = parse_query_spec(
            question.strip(), has_validated_context_scope=True
        )
        # Mirror the pipeline's missing-field merge so the eligibility verdict
        # matches the grounded path: stored validated scope fills only gaps.
        guard_updates: dict[str, Any] = {}
        if not eligibility_spec.plans and prior_scope.get("plan") is not None:
            guard_updates["plans"] = (prior_scope["plan"],)
        if not eligibility_spec.years and prior_scope.get("years"):
            guard_updates["years"] = tuple(prior_scope["years"])
        if not eligibility_spec.semesters and prior_scope.get("semesters"):
            guard_updates["semesters"] = tuple(prior_scope["semesters"])
        if not eligibility_spec.course_codes and prior_focus_code is not None:
            guard_updates["course_codes"] = (prior_focus_code,)
        if not eligibility_spec.course_codes and ordinal_code is not None:
            guard_updates["course_codes"] = (ordinal_code,)
        if not eligibility_spec.topic and service_topic is not None:
            guard_updates["topic"] = service_topic
        guard_spec = (
            replace(eligibility_spec, **guard_updates) if guard_updates else eligibility_spec
        )
        guard_context = QueryContext(
            program=selected_program,
            catalog_key=selected_catalog_key,
            plan=prior_scope.get("plan"),
            years=tuple(prior_scope.get("years", ())),
            semesters=tuple(prior_scope.get("semesters", ())),
            operations=tuple(prior_scope.get("operations", ())),
            course_code=prior_focus_code,
        )
        eligible = has_answerable_target_or_scope(guard_spec, guard_context)
    else:
        eligible = True
    if not eligible:
        # Structurally unanswerable dump-capable requests fail closed here,
        # before any SQL generation, so the outcome cannot vary with model
        # behavior. Valid targets and scopes flow through unchanged, and the
        # preserved next_context is left to the caller-owned fallback. This
        # applies only on the grounded product path; bare SQL-service calls
        # without grounding keep their direct mechanical contract.
        return {
            "status": "insufficient_evidence",
            "answer": "ไม่พบหลักฐานที่มีแหล่งอ้างอิงเพียงพอสำหรับคำตอบนี้",
            "provenance": [],
            "sql": None,
            "columns": [],
            "rows": [],
            "next_context": None,
        }

    focus_scope = dict(prior_scope)
    if query_spec.plans:
        focus_scope["plan"] = query_spec.plans[0] if len(query_spec.plans) == 1 else None
    for axis in ("years", "semesters"):
        explicit_values = tuple(getattr(query_spec, axis, ()))
        if explicit_values:
            focus_scope[axis] = explicit_values
    if query_spec.operations:
        focus_scope["operations"] = tuple(query_spec.operations)

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
    if ordinal_target is not None:
        course_scope = (
            (
                ordinal_target.get("catalog_key"),
                str(ordinal_target.get("program")),
                str(ordinal_target.get("course_code")),
            ),
        )
    elif use_result_set_scope and prior_result_courses is not None:
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
                selected_program,
                prior_focus,
                prior_result_courses,
                [],
                selected_catalog_key=selected_catalog_key,
                current_spec=query_spec,
                prior_scope=focus_scope,
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
    if selected_catalog_key is not None:
        generation_question += (
            f"\nSelected catalog_key: {selected_catalog_key}. "
            "This is the authoritative application scope. The execution "
            "environment already restricts canonical relations to this catalog. "
            "Never query outside the scoped relation set or add catalog predicates "
            "to relations that do not expose catalog identity."
        )
    if focus_scope:
        generation_question += (
            f"\nPreserve follow-up focus as SQL predicates: {focus_scope}. "
            "Do not remove a plan, year, or semester restriction inherited from "
            "validated conversation context."
        )

    # Rescue evidence-ID guidance is prompt-level only and applies solely to
    # the opt-in rescue request; the default prompt stays unchanged.
    rescue_evidence_guidance = (
        f"{_RESCUE_EVIDENCE_ID_GUIDANCE}\n\n"
        if allow_grounded_row_rescue is True
        else ""
    )

    def call_sql_model(prompt: str) -> str:
        try:
            generated = sql_model_callable(
                f"{prompt}\n\n{_PLAN_SCOPE_GUIDANCE}\n\n{_PROGRAM_SCOPE_GUIDANCE}\n\n"
                f"{_UNIQUE_COLUMN_GUIDANCE}\n\n"
                f"{_CATALOG_CONTEXT_GUIDANCE}\n\n"
                f"{_EVIDENCE_PROJECTION_GUIDANCE}\n\n"
                f"{rescue_evidence_guidance}"
                f"{_COMPARISON_COMPLETENESS_GUIDANCE}\n\n"
                f"{_LOGICAL_COURSE_COUNTING_GUIDANCE}\n\n"
                f"{_conversation_focus_guidance(prior_focus)}"
                f"{_conversation_result_set_guidance(prior_result_courses, selected_program)}"
                + (
                    f"Validated follow-up plan/year/semester focus: {focus_scope}. "
                    "Keep these predicates in SQL.\n"
                    if focus_scope
                    else ""
                )
                + (
                    f"Selected curriculum catalog_key: {selected_catalog_key}. "
                    f"{_SELECTED_CATALOG_SQL_GUIDANCE}\n"
                    if selected_catalog_key is not None
                    else ""
                )
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
        execution_scope = {
            "program": scope_program,
            "plan_key": focus_scope.get("plan"),
            "years": focus_scope.get("years", ()),
            "semesters": focus_scope.get("semesters", ()),
        }
        if course_scope is None:
            columns, raw_rows = execute_readonly(
                db_path,
                safe_sql,
                catalog_key=selected_catalog_key,
                **execution_scope,
            )
        else:
            columns, raw_rows = execute_readonly(
                db_path,
                safe_sql,
                course_scope=course_scope,
                plan_key=execution_scope.get("plan_key"),
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
                columns, raw_rows = execute_readonly(
                    db_path,
                    safe_sql,
                    catalog_key=selected_catalog_key,
                    **execution_scope,
                )
            else:
                columns, raw_rows = execute_readonly(
                    db_path,
                    safe_sql,
                    course_scope=course_scope,
                    plan_key=execution_scope.get("plan_key"),
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

    if grounding_callable is not None:
        try:
            grounded = grounding_callable(question.strip())
        except Exception:
            grounded = None
        if isinstance(grounded, Mapping):
            grounded_status = grounded.get("status")
            grounded_answer = grounded.get("final_answer")
            raw_provenance = grounded.get("provenance")
            next_context = grounded.get("next_context")
            provenance: list[dict[str, Any]] = []
            if isinstance(raw_provenance, (list, tuple)):
                for reference in raw_provenance:
                    if (
                        not isinstance(reference, Mapping)
                        or not isinstance(reference.get("source_filename"), str)
                        or not reference.get("source_filename")
                        or isinstance(reference.get("source_page"), bool)
                        or not isinstance(reference.get("source_page"), int)
                    ):
                        provenance = []
                        break
                    provenance.append(dict(reference))
            if (
                grounded_status == "answer"
                and isinstance(grounded_answer, str)
                and grounded_answer.strip()
                and provenance
            ):
                if not (
                    isinstance(next_context, dict)
                    and next_context.get("course_code")
                ):
                    # Answers naming several courses retain those canonical
                    # identities (in mention order) for ordinal follow-ups.
                    # Single-course targets stay owned by the focus machinery.
                    answer_named = _retained_identities_from_answer(
                        grounded_answer,
                        rows,
                        selected_program,
                        selected_catalog_key,
                        query_spec.topic or service_topic,
                        db_path,
                    )
                    if answer_named is not None:
                        if isinstance(next_context, dict):
                            next_context = {**next_context, **answer_named}
                        else:
                            next_context = answer_named
                if ordinal_target is not None and parsed_results is not None:
                    # The selected course is a child of this validated result
                    # set, not an independent replacement target. Keep the
                    # parent identities for subsequent sibling ordinals.
                    next_context = {
                        **(next_context if isinstance(next_context, dict) else {}),
                        "result_courses": parsed_results[0],
                        "result_scope_program": parsed_results[1],
                    }
                    if service_topic is not None:
                        next_context["semantic_topic"] = service_topic
                return {
                    "status": "answer",
                    "answer": grounded_answer.strip(),
                    "provenance": provenance,
                    "sql": safe_sql,
                    "columns": columns,
                    "rows": rows,
                    "next_context": next_context,
                }
            if grounded_status == "clarify_catalog":
                return {
                    "status": "clarification_required",
                    "action": "catalog_required",
                    "answer": "โปรดเลือกฉบับหลักสูตรก่อนค้นหาข้อมูล",
                    "provenance": [],
                    "next_context": None,
                }
        if allow_grounded_row_rescue is True and rows:
            # Opt-in rescue seam: deterministic grounding already failed to
            # produce a usable answer, so already-executed rows may ground one
            # only when every row carries complete canonical provenance.
            rescued = _grounded_row_rescue(
                db_path,
                question.strip(),
                selected_program,
                columns,
                rows,
                safe_sql,
                answer_model_callable,
                prior_focus,
                prior_result_courses,
                selected_catalog_key,
                query_spec,
                focus_scope,
            )
            if rescued is not None:
                return rescued
        return {
            "status": "insufficient_evidence",
            "answer": "ไม่พบหลักฐานที่มีแหล่งอ้างอิงเพียงพอสำหรับคำตอบนี้",
            "provenance": [],
            "sql": safe_sql,
            "columns": columns,
            "rows": rows,
            "next_context": None,
        }

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
                selected_catalog_key=selected_catalog_key,
                current_spec=query_spec,
                prior_scope=focus_scope,
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
            selected_catalog_key=selected_catalog_key,
            current_spec=query_spec,
            prior_scope=focus_scope,
        ),
    }


__all__ = [
    "ask_sql",
    "parse_focus_course_context",
    "parse_result_courses_context",
]
