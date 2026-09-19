"""One-shot, guarded SQL fallback boundary for approved structured requests.

This module intentionally stops at read-only candidate selection.  Its rows are
internal SQL output and are not grounded answer evidence or user-facing text.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Literal

from .execute import execute_readonly
from .guard_sql import guard_sql
from .nl_to_sql import question_to_sql
from .queries import (
    applicable_plan_keys,
    course_facts,
    placement_year_semester_choices,
    scoped_course_set,
)


FALLBACK_ALLOWLIST = frozenset(
    {
        "courses",
        "v_plan_courses",
        "v_semester_credits",
        "v_prerequisite_edges",
        "curriculum_plans",
        "programs",
    }
)

FALLBACK_SCHEMA = """
courses(
    course_id, catalog_id, course_code, course_code_normalized,
    name_th, name_en, credits, credit_units, credits_raw, category,
    course_type, prerequisite_text, notes
)
v_plan_courses(
    plan_id, program_id, program, plan, plan_code, plan_key,
    placement_id, placement_order, year, semester,
    flexible_year_number, flexible_semester_number,
    flexible_year_semester_raw, course_id, course, course_code,
    credits, credit_units, credits_raw, alternative_group_id,
    alternative_member_order, minimum_choices, maximum_choices,
    is_alternative, category, requirement_type, credits_override,
    raw_text, notes
)
v_semester_credits(plan_id, program, plan, year, semester, total_credits)
v_prerequisite_edges(
    prerequisite_id, source_course_id, source_course, source_course_code,
    prerequisite_course_id, prerequisite_course, prerequisite_code,
    alternative_group_id, alternative_member_order, prerequisite_order,
    requirement_type, raw_text
)
curriculum_plans(
    plan_id, catalog_id, program_id, program_code, plan_key, plan_code,
    plan_name, version, notes
)
programs(program_id, catalog_id, program_code, program_code_normalized)
""".strip()

_COURSE_LIST_SELECTOR_CONTRACT = (
    "COURSE LIST/FILTER FALLBACK CONTRACT: SQL is only a candidate selector, "
    "not a presentation query. The result MUST contain the canonical "
    "course_id column; prefer selecting only DISTINCT course_id (for example, "
    "SELECT DISTINCT p.course_id AS course_id ...). Filters may use other "
    "allowed columns, but do not return course_code, names, credits, or "
    "provenance as output fields. The deterministic scope above is "
    "authoritative and must not be widened or replaced."
)

_PLACEMENT_SELECTOR_CONTRACT = (
    "PLACEMENT FALLBACK CONTRACT: SQL is only a candidate selector, not a "
    "presentation query. The result MUST contain the canonical placement_id "
    "column; prefer selecting only DISTINCT placement_id (for example, "
    "SELECT DISTINCT p.placement_id AS placement_id ...). Filters may use "
    "other allowed columns, but do not return course names, credits, year, "
    "semester, plan labels, or provenance as output fields. The "
    "deterministic scope above is authoritative and must not be widened or "
    "replaced."
)

_COURSE_CREDIT_SELECTOR_CONTRACT = (
    "COURSE CREDIT FALLBACK CONTRACT: SQL is only a candidate selector, not a "
    "presentation query. The result MUST contain the canonical course_id "
    "column; prefer selecting only DISTINCT course_id (for example, SELECT "
    "DISTINCT p.course_id AS course_id ...). Filters may use other allowed "
    "columns, but do not return course names, credits, or provenance as output "
    "fields. Select exactly one logical course target. The deterministic scope "
    "above is authoritative and must not be widened or replaced."
)


@dataclass(frozen=True)
class StructuredFallbackScope:
    """Deterministic scope that the SQL model must treat as authoritative."""

    program: str
    plans: tuple[str, ...] = ()
    years: tuple[int, ...] = ()
    semesters: tuple[int, ...] = ()
    course_ids: tuple[int, ...] = ()
    course_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.program, str) or not self.program.strip():
            raise ValueError("program is required for SQL fallback scope")
        for name, values in (
            ("plans", self.plans),
            ("years", self.years),
            ("semesters", self.semesters),
            ("course_ids", self.course_ids),
            ("course_codes", self.course_codes),
        ):
            if isinstance(values, (str, bytes)):
                raise TypeError(f"{name} must be an iterable of values")
            try:
                tuple(values)
            except TypeError as error:
                raise TypeError(f"{name} must be an iterable of values") from error


@dataclass(frozen=True)
class StructuredFallbackResult:
    """Internal candidate output; rows are not grounded answer evidence."""

    status: Literal["success", "error"]
    sql: str | None = None
    columns: tuple[str, ...] = ()
    rows: tuple[tuple[Any, ...], ...] = ()
    error_category: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class GroundedCourseListResult:
    """Canonical list evidence produced after SQL candidate selection."""

    status: Literal["complete", "valid_empty", "insufficient_evidence"]
    records: tuple[Mapping[str, Any], ...] = ()
    error: str | None = None


@dataclass(frozen=True)
class GroundedCourseCreditResult:
    """Canonical single-course credit evidence after SQL selection."""

    status: Literal["complete", "valid_empty", "insufficient_evidence"]
    records: tuple[Mapping[str, Any], ...] = ()
    credit_units: Any = None
    credits: Any = None
    provenance: tuple[Mapping[str, Any], ...] = ()
    error: str | None = None


@dataclass(frozen=True)
class GroundedPlacementResult:
    """Canonical placement evidence produced after SQL candidate selection."""

    status: Literal["complete", "valid_empty", "insufficient_evidence"]
    records: tuple[Mapping[str, Any], ...] = ()
    error: str | None = None


def _scope_prompt(scope: StructuredFallbackScope) -> str:
    def render(values: Iterable[Any]) -> str:
        values_tuple = tuple(values)
        return ", ".join(repr(value) for value in values_tuple) or "(none)"

    return (
        "AUTHORITATIVE DETERMINISTIC SCOPE — these values are already resolved "
        "and MUST be preserved as exact SQL constraints. Do not widen, infer, "
        "replace, or omit them. Never infer a program from a course-code prefix.\n"
        f"program: {scope.program!r}\n"
        f"plans: {render(scope.plans)}\n"
        f"years: {render(scope.years)}\n"
        f"semesters: {render(scope.semesters)}\n"
        f"course_ids: {render(scope.course_ids)}\n"
        f"course_codes: {render(scope.course_codes)}"
    )


def _validate_result(
    value: Any,
) -> tuple[tuple[str, ...], tuple[tuple[Any, ...], ...]]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError("read-only execution returned an invalid result")
    columns, rows = value
    if not isinstance(columns, list) or not all(
        isinstance(column, str) for column in columns
    ):
        raise TypeError("read-only execution returned invalid columns")
    if not isinstance(rows, list) or any(
        not isinstance(row, tuple) or len(row) != len(columns) for row in rows
    ):
        raise TypeError("read-only execution returned invalid rows")
    return tuple(columns), tuple(rows)


def _canonical_course_ids(record: Mapping[str, Any]) -> set[int]:
    identifiers: set[int] = set()
    course_id = record.get("course_id")
    if isinstance(course_id, int) and not isinstance(course_id, bool):
        identifiers.add(course_id)
    members = record.get("alternative_courses", ())
    if isinstance(members, (list, tuple)):
        for member in members:
            if not isinstance(member, Mapping):
                continue
            member_id = member.get("course_id")
            if isinstance(member_id, int) and not isinstance(member_id, bool):
                identifiers.add(member_id)
    return identifiers


def _canonical_course_codes(record: Mapping[str, Any]) -> set[str]:
    codes: set[str] = set()
    for value in (record.get("course_code"),):
        if isinstance(value, str) and value.strip():
            codes.add(value.strip())
    members = record.get("alternative_courses", ())
    if isinstance(members, (list, tuple)):
        for member in members:
            if not isinstance(member, Mapping):
                continue
            value = member.get("course_code")
            if isinstance(value, str) and value.strip():
                codes.add(value.strip())
    return codes


def _record_fingerprint(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        "alternative" if record.get("is_alternative") else "course",
        record.get("placement_id"),
        record.get("alternative_group_id"),
    )


def ground_course_list(
    db_path: str | Path,
    sql_result: StructuredFallbackResult,
    scope: StructuredFallbackScope,
) -> GroundedCourseListResult:
    """Hydrate SQL-selected IDs through canonical scoped list evidence.

    SQL output is treated only as a candidate selector.  Names, credits,
    categories, and provenance from SQL are ignored; all factual records and
    provenance come from ``scoped_course_set``.
    """
    if sql_result.status != "success":
        return GroundedCourseListResult(
            status="insufficient_evidence",
            error="SQL fallback did not produce a successful selector result",
        )

    course_id_index = next(
        (
            index
            for index, column in enumerate(sql_result.columns)
            if isinstance(column, str) and column.casefold() == "course_id"
        ),
        None,
    )
    if course_id_index is None:
        return GroundedCourseListResult(
            status="insufficient_evidence",
            error="SQL selector must return course_id",
        )

    selected_ids: list[int] = []
    for row in sql_result.rows:
        if course_id_index >= len(row):
            return GroundedCourseListResult(
                status="insufficient_evidence",
                error="SQL selector row does not contain course_id",
            )
        course_id = row[course_id_index]
        if isinstance(course_id, bool) or not isinstance(course_id, int):
            return GroundedCourseListResult(
                status="insufficient_evidence",
                error="SQL selector course_id must be an integer",
            )
        if course_id not in selected_ids:
            selected_ids.append(course_id)

    try:
        plans = scope.plans or applicable_plan_keys(db_path, scope.program)
        if not plans:
            return GroundedCourseListResult(
                status="insufficient_evidence",
                error="no canonical plans exist for the authoritative program",
            )
        canonical = scoped_course_set(
            db_path,
            scope.program,
            plans,
            years=scope.years,
            semesters=scope.semesters,
            course_targets=[{"course_id": course_id} for course_id in selected_ids],
        )
    except Exception as error:
        return GroundedCourseListResult(
            status="insufficient_evidence",
            error=f"canonical list grounding failed: {error}",
        )

    if not isinstance(canonical, Mapping):
        return GroundedCourseListResult(
            status="insufficient_evidence",
            error="canonical list grounding returned an invalid result",
        )

    records = canonical.get("courses", ())
    if not isinstance(records, (list, tuple)):
        return GroundedCourseListResult(
            status="insufficient_evidence",
            error="canonical list grounding returned invalid records",
        )

    if not selected_ids:
        return GroundedCourseListResult(status="valid_empty")

    found_ids: set[int] = set()
    found_codes: set[str] = set()
    selected_id_set = set(selected_ids)
    ordered_records: list[Mapping[str, Any]] = []
    seen_records: set[tuple[Any, ...]] = set()
    for record in records:
        if not isinstance(record, Mapping):
            return GroundedCourseListResult(
                status="insufficient_evidence",
                error="canonical list grounding returned a malformed record",
            )
        record_ids = _canonical_course_ids(record)
        if selected_id_set.isdisjoint(record_ids):
            continue
        provenance = record.get("provenance")
        if not isinstance(provenance, (list, tuple)) or not provenance:
            return GroundedCourseListResult(
                status="insufficient_evidence",
                error="canonical record is missing provenance",
            )
        found_ids.update(record_ids & selected_id_set)
        found_codes.update(_canonical_course_codes(record))
        fingerprint = _record_fingerprint(record)
        if fingerprint not in seen_records:
            seen_records.add(fingerprint)
            ordered_records.append(record)

    if found_ids != selected_id_set:
        return GroundedCourseListResult(
            status="insufficient_evidence",
            error="SQL-selected course_id is outside canonical scope or unknown",
        )

    known_ids = set(scope.course_ids)
    if known_ids and not selected_id_set.issubset(known_ids):
        return GroundedCourseListResult(
            status="insufficient_evidence",
            error="SQL selector widened the authoritative course scope",
        )
    known_codes = set(scope.course_codes)
    if known_codes and not found_codes.intersection(known_codes):
        return GroundedCourseListResult(
            status="insufficient_evidence",
            error="SQL selector widened the authoritative course-code scope",
        )

    return GroundedCourseListResult(
        status="complete",
        records=tuple(ordered_records),
    )


def _normalized_credit_number(value: Any) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        return None


def _course_credit_signature(record: Mapping[str, Any]) -> tuple[Any, ...] | None:
    credit_units = _normalized_credit_number(record.get("credit_units"))
    credits = record.get("credits")
    if not isinstance(credits, str) or not credits.strip():
        credits = record.get("credits_raw")
    normalized_credits = (
        " ".join(credits.split()) if isinstance(credits, str) and credits.strip() else None
    )
    if credit_units is None and normalized_credits is None:
        return None
    return credit_units, normalized_credits


def _course_fact_matches_scope(
    record: Mapping[str, Any],
    scope: StructuredFallbackScope,
) -> bool:
    if not (scope.plans or scope.years or scope.semesters):
        return True

    placements = record.get("placements")
    if not isinstance(placements, (list, tuple)):
        return False
    normalized_plans = {plan.strip().casefold() for plan in scope.plans}
    for placement in placements:
        if not isinstance(placement, Mapping):
            continue
        plan_key = placement.get("plan_key")
        if normalized_plans and (
            not isinstance(plan_key, str)
            or plan_key.strip().casefold() not in normalized_plans
        ):
            continue
        choices = placement_year_semester_choices(
            placement.get("year"),
            placement.get("semester"),
            placement.get("flexible_year_semester_raw"),
        )
        if not choices:
            continue
        if scope.years and not any(year in scope.years for year, _ in choices):
            continue
        if scope.semesters and not any(
            semester in scope.semesters for _, semester in choices
        ):
            continue
        if scope.years and scope.semesters and not any(
            year in scope.years and semester in scope.semesters
            for year, semester in choices
        ):
            continue
        return True
    return False


def _canonical_fact_provenance(
    records: Iterable[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...] | None:
    references: list[Mapping[str, Any]] = []
    seen: set[Any] = set()
    for record in records:
        provenance = record.get("provenance")
        if not isinstance(provenance, (list, tuple)) or not provenance:
            return None
        for reference in provenance:
            if not isinstance(reference, Mapping):
                return None
            provenance_id = reference.get("provenance_id")
            key = ("id", provenance_id) if provenance_id is not None else (
                "value",
                repr(dict(reference)),
            )
            if key not in seen:
                seen.add(key)
                references.append(reference)
    return tuple(references)


def ground_course_credit(
    db_path: str | Path,
    sql_result: StructuredFallbackResult,
    scope: StructuredFallbackScope,
) -> GroundedCourseCreditResult:
    """Hydrate one logical course's credit fact from canonical course data.

    SQL rows are candidate course selectors only.  Canonical course facts,
    credit values, and provenance come exclusively from ``course_facts``.
    """
    if sql_result.status != "success":
        return GroundedCourseCreditResult(
            status="insufficient_evidence",
            error="SQL fallback did not produce a successful selector result",
        )

    course_id_index = next(
        (
            index
            for index, column in enumerate(sql_result.columns)
            if isinstance(column, str) and column.casefold() == "course_id"
        ),
        None,
    )
    if course_id_index is None:
        return GroundedCourseCreditResult(
            status="insufficient_evidence",
            error="SQL selector must return course_id",
        )

    selected_ids: list[int] = []
    for row in sql_result.rows:
        if course_id_index >= len(row):
            return GroundedCourseCreditResult(
                status="insufficient_evidence",
                error="SQL selector row does not contain course_id",
            )
        course_id = row[course_id_index]
        if isinstance(course_id, bool) or not isinstance(course_id, int):
            return GroundedCourseCreditResult(
                status="insufficient_evidence",
                error="SQL selector course_id must be an integer",
            )
        if course_id not in selected_ids:
            selected_ids.append(course_id)

    course_codes = tuple(dict.fromkeys(
        code.strip()
        for code in scope.course_codes
        if isinstance(code, str) and code.strip()
    ))
    if len(course_codes) != 1:
        return GroundedCourseCreditResult(
            status="insufficient_evidence",
            error="course credit fallback requires one exact course code",
        )

    try:
        canonical = course_facts(db_path, course_codes[0], scope.program)
    except Exception as error:
        return GroundedCourseCreditResult(
            status="insufficient_evidence",
            error=f"canonical course grounding failed: {error}",
        )
    if not isinstance(canonical, Mapping):
        return GroundedCourseCreditResult(
            status="insufficient_evidence",
            error="canonical course grounding returned an invalid result",
        )
    facts = canonical.get("courses", ())
    if not isinstance(facts, (list, tuple)):
        return GroundedCourseCreditResult(
            status="insufficient_evidence",
            error="canonical course grounding returned invalid facts",
        )

    in_scope: list[Mapping[str, Any]] = []
    for fact in facts:
        if not isinstance(fact, Mapping):
            return GroundedCourseCreditResult(
                status="insufficient_evidence",
                error="canonical course grounding returned a malformed fact",
            )
        if fact.get("course_code") != course_codes[0]:
            continue
        if _course_fact_matches_scope(fact, scope):
            in_scope.append(fact)

    if not selected_ids:
        if not in_scope:
            return GroundedCourseCreditResult(status="valid_empty")
        return GroundedCourseCreditResult(
            status="insufficient_evidence",
            error="empty selector omitted an in-scope canonical course",
        )

    canonical_ids = {
        fact.get("course_id")
        for fact in in_scope
        if isinstance(fact.get("course_id"), int)
        and not isinstance(fact.get("course_id"), bool)
    }
    if not in_scope or not set(selected_ids).issubset(canonical_ids):
        return GroundedCourseCreditResult(
            status="insufficient_evidence",
            error="SQL-selected course_id is outside canonical scope or unknown",
        )

    signatures = {_course_credit_signature(fact) for fact in in_scope}
    if None in signatures or len(signatures) != 1:
        return GroundedCourseCreditResult(
            status="insufficient_evidence",
            error="canonical course identities have conflicting or missing credits",
        )
    provenance = _canonical_fact_provenance(in_scope)
    if provenance is None:
        return GroundedCourseCreditResult(
            status="insufficient_evidence",
            error="canonical course fact is missing provenance",
        )

    representative = in_scope[0]
    return GroundedCourseCreditResult(
        status="complete",
        records=tuple(in_scope),
        credit_units=representative.get("credit_units"),
        credits=representative.get("credits") or representative.get("credits_raw"),
        provenance=provenance,
    )


def _record_matches_placement_scope(
    record: Mapping[str, Any],
    scope: StructuredFallbackScope,
) -> bool:
    program = record.get("program")
    if not isinstance(program, str) or program.strip().upper() != scope.program.strip().upper():
        return False

    if scope.plans:
        plan_key = record.get("plan_key")
        if not isinstance(plan_key, str) or plan_key.strip().lower() not in {
            value.strip().lower() for value in scope.plans
        }:
            return False

    choices = record.get("year_semester_choices")
    if isinstance(choices, (list, tuple)) and choices:
        normalized_choices = {
            (choice[0], choice[1])
            for choice in choices
            if isinstance(choice, (list, tuple)) and len(choice) == 2
        }
    else:
        year = record.get("year_number")
        semester = record.get("semester_number")
        normalized_choices = {(year, semester)}

    if scope.years and not any(year in scope.years for year, _ in normalized_choices):
        return False
    if scope.semesters and not any(
        semester in scope.semesters for _, semester in normalized_choices
    ):
        return False
    if scope.years and scope.semesters and not any(
        year in scope.years and semester in scope.semesters
        for year, semester in normalized_choices
    ):
        return False

    if scope.course_ids and not _canonical_course_ids(record).intersection(
        scope.course_ids
    ):
        return False
    if scope.course_codes and not _canonical_course_codes(record).intersection(
        {code.strip() for code in scope.course_codes}
    ):
        return False
    return True


def ground_placement(
    db_path: str | Path,
    sql_result: StructuredFallbackResult,
    scope: StructuredFallbackScope,
) -> GroundedPlacementResult:
    """Hydrate SQL-selected placement IDs through canonical scoped records.

    SQL output is treated only as a candidate selector.  Placement facts and
    provenance come exclusively from ``scoped_course_set``.
    """
    if sql_result.status != "success":
        return GroundedPlacementResult(
            status="insufficient_evidence",
            error="SQL fallback did not produce a successful selector result",
        )

    placement_id_index = next(
        (
            index
            for index, column in enumerate(sql_result.columns)
            if isinstance(column, str) and column.casefold() == "placement_id"
        ),
        None,
    )
    if placement_id_index is None:
        return GroundedPlacementResult(
            status="insufficient_evidence",
            error="SQL selector must return placement_id",
        )

    selected_ids: list[int] = []
    for row in sql_result.rows:
        if placement_id_index >= len(row):
            return GroundedPlacementResult(
                status="insufficient_evidence",
                error="SQL selector row does not contain placement_id",
            )
        placement_id = row[placement_id_index]
        if isinstance(placement_id, bool) or not isinstance(placement_id, int):
            return GroundedPlacementResult(
                status="insufficient_evidence",
                error="SQL selector placement_id must be an integer",
            )
        if placement_id not in selected_ids:
            selected_ids.append(placement_id)

    try:
        plans = scope.plans or applicable_plan_keys(db_path, scope.program)
        if not plans:
            return GroundedPlacementResult(
                status="insufficient_evidence",
                error="no canonical plans exist for the authoritative program",
            )
        course_targets = [
            {"course_id": course_id} for course_id in scope.course_ids
        ]
        if not course_targets:
            course_targets = [
                {"course_code": course_code} for course_code in scope.course_codes
            ]
        canonical = scoped_course_set(
            db_path,
            scope.program,
            plans,
            years=scope.years,
            semesters=scope.semesters,
            course_targets=course_targets,
        )
    except Exception as error:
        return GroundedPlacementResult(
            status="insufficient_evidence",
            error=f"canonical placement grounding failed: {error}",
        )

    if not isinstance(canonical, Mapping):
        return GroundedPlacementResult(
            status="insufficient_evidence",
            error="canonical placement grounding returned an invalid result",
        )
    records = canonical.get("courses", ())
    if not isinstance(records, (list, tuple)):
        return GroundedPlacementResult(
            status="insufficient_evidence",
            error="canonical placement grounding returned invalid records",
        )
    if not selected_ids:
        return GroundedPlacementResult(status="valid_empty")

    by_placement_id: dict[int, list[Mapping[str, Any]]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            return GroundedPlacementResult(
                status="insufficient_evidence",
                error="canonical placement grounding returned a malformed record",
            )
        placement_id = record.get("placement_id")
        if isinstance(placement_id, bool) or not isinstance(placement_id, int):
            continue
        if placement_id in selected_ids:
            by_placement_id.setdefault(placement_id, []).append(record)

    ordered_records: list[Mapping[str, Any]] = []
    for placement_id in selected_ids:
        matches = by_placement_id.get(placement_id, [])
        if len(matches) != 1:
            return GroundedPlacementResult(
                status="insufficient_evidence",
                error="SQL-selected placement_id is outside canonical scope, unknown, or ambiguous",
            )
        record = matches[0]
        if not _record_matches_placement_scope(record, scope):
            return GroundedPlacementResult(
                status="insufficient_evidence",
                error="SQL selector widened the authoritative placement scope",
            )
        provenance = record.get("provenance")
        if not isinstance(provenance, (list, tuple)) or not provenance:
            return GroundedPlacementResult(
                status="insufficient_evidence",
                error="canonical placement record is missing provenance",
            )
        ordered_records.append(record)

    return GroundedPlacementResult(
        status="complete",
        records=tuple(ordered_records),
    )


def run_structured_fallback(
    db_path: str | Path,
    question: str,
    scope: StructuredFallbackScope,
    model_callable: Callable[[str], str],
    *,
    selector_mode: Literal["course_list", "placement"] = "course_list",
) -> StructuredFallbackResult:
    """Make one model-to-SQL attempt and execute it read-only.

    The returned rows are internal candidate-selector output only.  Callers
    must not render them as an answer or treat them as grounded evidence until
    a later canonical grounding step attaches curriculum provenance.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    if not callable(model_callable):
        raise TypeError("model_callable must be callable")
    if selector_mode not in {"course_list", "placement", "course_credit"}:
        raise ValueError("unsupported structured fallback selector mode")

    selector_contract = {
        "course_list": _COURSE_LIST_SELECTOR_CONTRACT,
        "placement": _PLACEMENT_SELECTOR_CONTRACT,
        "course_credit": _COURSE_CREDIT_SELECTOR_CONTRACT,
    }[selector_mode]

    scoped_question = (
        f"{_scope_prompt(scope)}\n\n"
        f"{selector_contract}\n\n"
        f"USER QUESTION:\n{question.strip()}"
    )
    generated_sql: str | None = None
    try:
        generated_sql = question_to_sql(
            scoped_question,
            FALLBACK_SCHEMA,
            model_callable,
        )
    except Exception as error:
        return StructuredFallbackResult(
            status="error",
            error_category="generation",
            error=str(error),
        )

    try:
        safe_sql = guard_sql(
            generated_sql,
            allowed_relations=FALLBACK_ALLOWLIST,
        )
    except Exception as error:
        return StructuredFallbackResult(
            status="error",
            sql=generated_sql,
            error_category="relation_guard",
            error=str(error),
        )

    try:
        columns, rows = _validate_result(execute_readonly(db_path, safe_sql))
    except Exception as error:
        return StructuredFallbackResult(
            status="error",
            sql=safe_sql,
            error_category="execution",
            error=str(error),
        )

    return StructuredFallbackResult(
        status="success",
        sql=safe_sql,
        columns=columns,
        rows=rows,
    )


__all__ = [
    "FALLBACK_ALLOWLIST",
    "FALLBACK_SCHEMA",
    "GroundedCourseListResult",
    "GroundedCourseCreditResult",
    "GroundedPlacementResult",
    "StructuredFallbackResult",
    "StructuredFallbackScope",
    "ground_course_list",
    "ground_course_credit",
    "ground_placement",
    "run_structured_fallback",
]
