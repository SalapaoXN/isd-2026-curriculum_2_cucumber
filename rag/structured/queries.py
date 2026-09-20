"""Deterministic read-only queries over the structured curriculum schema."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterable, Mapping
from contextlib import closing, contextmanager
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterator


Database = str | Path | sqlite3.Connection
_CREDIT_RE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)")
_COURSE_CODE_RE = re.compile(r"[0-9]{8}")
_COURSE_NAME_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_FLEXIBLE_YEAR_SEMESTER_PART = re.compile(r"\s*([1-5])\s*/\s*([1-2])\s*")
_CANONICAL_PLAN_KEYS = frozenset({"coop", "no_coop", "default", "gened"})


@contextmanager
def _open_database(database: Database) -> Iterator[sqlite3.Connection]:
    if isinstance(database, sqlite3.Connection):
        previous_factory = database.row_factory
        database.row_factory = sqlite3.Row
        try:
            yield database
        finally:
            database.row_factory = previous_factory
        return

    database_path = Path(database)
    if not database_path.exists():
        raise FileNotFoundError(database_path)
    with closing(sqlite3.connect(str(database_path))) as connection:
        connection.row_factory = sqlite3.Row
        yield connection


def _provenance_for(
    connection: sqlite3.Connection,
    link_table: str,
    entity_column: str,
    entity_id: int,
) -> list[dict[str, Any]]:
    query = f"""
        SELECT
            provenance.provenance_id,
            provenance.program,
            provenance.source_filename,
            provenance.source_page,
            provenance.document_page,
            provenance.document_category,
            provenance.source_uri,
            provenance.source_locator,
            provenance.excerpt
        FROM {link_table} AS links
        JOIN provenance
          ON provenance.provenance_id = links.provenance_id
        WHERE links.{entity_column} = ?
        ORDER BY provenance.provenance_id
    """
    return [dict(row) for row in connection.execute(query, (entity_id,))]


def _merge_provenance(
    *reference_lists: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[int] = set()
    for references in reference_lists:
        for reference in references:
            provenance_id = int(reference["provenance_id"])
            if provenance_id in seen:
                continue
            seen.add(provenance_id)
            merged.append(dict(reference))
    return merged


def _source_pages(references: Iterable[Mapping[str, Any]]) -> list[int]:
    pages: list[int] = []
    for reference in references:
        page = reference.get("source_page")
        if page is not None and page not in pages:
            pages.append(page)
    return pages


def _course_fields(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "course_id": int(row["course_id"]),
        "catalog_id": row["catalog_id"],
        "course_code": row["course_code"],
        "name_th": row["name_th"],
        "name_en": row["name_en"],
        "credits": row["credits"],
        "credits_raw": row["credits_raw"],
        "description_th": row["description_th"],
        "description_en": row["description_en"],
        "category": row["category"],
        "course_type": row["course_type"],
        "prerequisite_text": row["prerequisite_text"],
        "notes": row["notes"],
    }


def _course_select(prefix: str = "courses") -> str:
    return f"""
        {prefix}.course_id,
        {prefix}.catalog_id,
        {prefix}.course_code,
        {prefix}.name_th,
        {prefix}.name_en,
        {prefix}.credits,
        {prefix}.credits_raw,
        {prefix}.description_th,
        {prefix}.description_en,
        {prefix}.category,
        {prefix}.course_type,
        {prefix}.prerequisite_text,
        {prefix}.notes
    """


def _placement_rows(
    connection: sqlite3.Connection,
    plan_id: int,
    year_number: int,
    semester_number: int,
) -> list[sqlite3.Row]:
    return connection.execute(
        f"""
        SELECT
            placements.placement_id,
            placements.plan_id,
            placements.course_id,
            placements.alternative_group_id,
            placements.year_number,
            placements.semester_number,
            placements.category,
            placements.requirement_type,
            placements.placement_order,
            placements.credits_override,
            placements.raw_text,
            placements.notes,
            plans.program_code,
            plans.plan_code,
            {_course_select('courses')},
            groups.group_key,
            groups.label,
            groups.minimum_choices,
            groups.maximum_choices,
            groups.notes AS group_notes
        FROM plan_placements AS placements
        JOIN curriculum_plans AS plans ON plans.plan_id = placements.plan_id
        LEFT JOIN courses ON courses.course_id = placements.course_id
        LEFT JOIN alternative_course_groups AS groups
            ON groups.alternative_group_id = placements.alternative_group_id
        WHERE placements.plan_id = ?
          AND placements.year_number = ?
          AND placements.semester_number = ?
        ORDER BY placements.placement_order IS NULL,
                 placements.placement_order,
                 placements.placement_id
        """,
        (plan_id, year_number, semester_number),
    ).fetchall()


def _alternative_members(
    connection: sqlite3.Connection, alternative_group_id: int
) -> list[dict[str, Any]]:
    members: list[dict[str, Any]] = []
    rows = connection.execute(
        f"""
        SELECT
            members.alternative_group_member_id,
            members.alternative_group_id,
            members.course_id,
            members.member_order,
            {_course_select('courses')}
        FROM alternative_course_group_members AS members
        JOIN courses ON courses.course_id = members.course_id
        WHERE members.alternative_group_id = ?
        ORDER BY members.member_order, members.alternative_group_member_id
        """,
        (alternative_group_id,),
    ).fetchall()
    for row in rows:
        course = _course_fields(row)
        references = _merge_provenance(
            _provenance_for(
                connection,
                "course_provenance",
                "course_id",
                int(row["course_id"]),
            ),
            _provenance_for(
                connection,
                "alternative_group_member_provenance",
                "alternative_group_member_id",
                int(row["alternative_group_member_id"]),
            ),
        )
        members.append(
            {
                "alternative_group_member_id": int(
                    row["alternative_group_member_id"]
                ),
                "member_order": row["member_order"],
                **course,
                "provenance": references,
                "source_pages": _source_pages(references),
            }
        )
    return members


def _decimal_credits(value: Any) -> Decimal | None:
    if value is None:
        return None
    match = _CREDIT_RE.search(str(value))
    if match is None:
        return None
    try:
        return Decimal(match.group(1))
    except InvalidOperation:
        return None


def _credits_for_placement(
    connection: sqlite3.Connection, placement: Mapping[str, Any]
) -> Decimal:
    override = placement.get("credits_override")
    if override not in (None, ""):
        value = _decimal_credits(override)
        return value if value is not None else Decimal(0)

    group_id = placement.get("alternative_group_id")
    if group_id is None:
        value = _decimal_credits(placement.get("credits"))
        return value if value is not None else Decimal(0)

    members = _alternative_members(connection, int(group_id))
    choices = int(placement.get("minimum_choices") or 1)
    total = Decimal(0)
    for member in members[:choices]:
        value = _decimal_credits(member.get("credits"))
        if value is not None:
            total += value
    return total


def parse_flexible_year_semester(value: Any) -> list[tuple[int, int]]:
    """Parse a complete comma-separated list of valid year/semester choices."""
    if not isinstance(value, str) or not value.strip():
        return []

    choices: list[tuple[int, int]] = []
    for part in value.split(","):
        match = _FLEXIBLE_YEAR_SEMESTER_PART.fullmatch(part)
        if match is None:
            return []
        choice = (int(match.group(1)), int(match.group(2)))
        if choice not in choices:
            choices.append(choice)
    return choices


def placement_year_semester_choices(
    year: Any,
    semester: Any,
    flexible_year_semester_raw: Any,
) -> list[tuple[int, int]]:
    """Return comparable fixed or flexible placement choices without guessing."""
    if year is not None or semester is not None:
        if (
            isinstance(year, int)
            and not isinstance(year, bool)
            and isinstance(semester, int)
            and not isinstance(semester, bool)
            and 1 <= year <= 5
            and 1 <= semester <= 2
        ):
            return [(year, semester)]
        return []
    return parse_flexible_year_semester(flexible_year_semester_raw)


def earliest_year_semester_from_choices(
    choices: Any,
) -> tuple[int, int] | None:
    """Return the earliest choice from an already-normalized placement."""
    if not isinstance(choices, (list, tuple)):
        return None

    valid_choices: list[tuple[int, int]] = []
    for choice in choices:
        if (
            not isinstance(choice, (list, tuple))
            or len(choice) != 2
            or isinstance(choice[0], bool)
            or isinstance(choice[1], bool)
            or not isinstance(choice[0], int)
            or not isinstance(choice[1], int)
            or not 1 <= choice[0] <= 5
            or not 1 <= choice[1] <= 2
        ):
            return None
        valid_choices.append((choice[0], choice[1]))
    return min(valid_choices) if valid_choices else None


def earliest_year_semester(
    year: Any,
    semester: Any,
    flexible_year_semester_raw: Any,
) -> tuple[int, int] | None:
    """Return the earliest valid fixed/flexible placement chronologically."""
    choices = placement_year_semester_choices(
        year,
        semester,
        flexible_year_semester_raw,
    )
    return earliest_year_semester_from_choices(choices)


def _normalize_course_placement_inputs(
    program: str,
    course_code: str,
    plan_keys: str | Iterable[str],
) -> tuple[str, str, list[str]]:
    if not isinstance(program, str) or not program.strip():
        raise ValueError("program must be a non-empty string")
    if not isinstance(course_code, str) or _COURSE_CODE_RE.fullmatch(course_code) is None:
        raise ValueError("course_code must contain exactly 8 ASCII digits")

    raw_plan_keys = [plan_keys] if isinstance(plan_keys, str) else list(plan_keys)
    normalized_plan_keys: list[str] = []
    for plan_key in raw_plan_keys:
        if not isinstance(plan_key, str):
            raise ValueError("plan_keys must contain strings")
        normalized = plan_key.strip().lower()
        if normalized not in _CANONICAL_PLAN_KEYS:
            raise ValueError(f"unsupported canonical plan_key: {plan_key!r}")
        if normalized not in normalized_plan_keys:
            normalized_plan_keys.append(normalized)
    if not normalized_plan_keys:
        raise ValueError("at least one canonical plan_key is required")

    return program.strip().upper(), course_code, normalized_plan_keys


def _course_placement_provenance(
    connection: sqlite3.Connection,
    placement_id: int,
    course_id: int,
    alternative_group_id: int | None,
) -> list[dict[str, Any]]:
    reference_lists: list[list[dict[str, Any]]] = [
        _provenance_for(
            connection,
            "plan_placement_provenance",
            "placement_id",
            placement_id,
        ),
        _provenance_for(
            connection,
            "course_provenance",
            "course_id",
            course_id,
        ),
    ]
    if alternative_group_id is not None:
        reference_lists.append(
            _provenance_for(
                connection,
                "alternative_group_provenance",
                "alternative_group_id",
                alternative_group_id,
            )
        )
        member_ids = connection.execute(
            """
            SELECT alternative_group_member_id
            FROM alternative_course_group_members
            WHERE alternative_group_id = ? AND course_id = ?
            ORDER BY alternative_group_member_id
            """,
            (alternative_group_id, course_id),
        )
        for member in member_ids:
            reference_lists.append(
                _provenance_for(
                    connection,
                    "alternative_group_member_provenance",
                    "alternative_group_member_id",
                    int(member["alternative_group_member_id"]),
                )
            )
    return _merge_provenance(*reference_lists)


def course_placement(
    db_path: Database,
    program: str,
    course_code: str,
    plan_keys: str | Iterable[str],
) -> dict[str, Any]:
    """Return independent placement rows for a course in requested plans."""
    normalized_program, normalized_code, normalized_plan_keys = (
        _normalize_course_placement_inputs(program, course_code, plan_keys)
    )
    placeholders = ", ".join("?" for _ in normalized_plan_keys)
    query = f"""
        SELECT
            plan_courses.placement_id,
            plan_courses.plan_key,
            plan_courses.plan_id,
            plans.catalog_id,
            plan_courses.course_id,
            plan_courses.course_code,
            courses.name_th,
            courses.name_en,
            plan_courses.year,
            plan_courses.semester,
            plan_courses.flexible_year_semester_raw,
            plan_courses.credits_raw,
            plan_courses.alternative_group_id
        FROM v_plan_courses AS plan_courses
        JOIN curriculum_plans AS plans
          ON plans.plan_id = plan_courses.plan_id
        JOIN courses
          ON courses.course_id = plan_courses.course_id
        WHERE plan_courses.program = ?
          AND plan_courses.course_code = ?
          AND plan_courses.plan_key IN ({placeholders})
        ORDER BY
            plan_courses.plan_key,
            plans.catalog_id,
            plan_courses.plan_id,
            plan_courses.placement_id,
            plan_courses.course_id
    """

    with _open_database(db_path) as connection:
        rows = connection.execute(
            query,
            (normalized_program, normalized_code, *normalized_plan_keys),
        ).fetchall()
        placements: list[dict[str, Any]] = []
        for row in rows:
            alternative_group_id = (
                int(row["alternative_group_id"])
                if row["alternative_group_id"] is not None
                else None
            )
            placement_id = int(row["placement_id"])
            course_id = int(row["course_id"])
            year_semester_choices = placement_year_semester_choices(
                row["year"],
                row["semester"],
                row["flexible_year_semester_raw"],
            )
            placements.append(
                {
                    "placement_id": placement_id,
                    "plan_key": row["plan_key"],
                    "plan_id": int(row["plan_id"]),
                    "catalog_id": int(row["catalog_id"]),
                    "course_id": course_id,
                    "course_code": row["course_code"],
                    "name_th": row["name_th"],
                    "name_en": row["name_en"],
                    "year": row["year"],
                    "semester": row["semester"],
                    "flexible_year_semester_raw": row[
                        "flexible_year_semester_raw"
                    ],
                    "year_semester_choices": year_semester_choices,
                    "credits_raw": row["credits_raw"],
                    "alternative_group_id": alternative_group_id,
                    "provenance": _course_placement_provenance(
                        connection,
                        placement_id,
                        course_id,
                        alternative_group_id,
                    ),
                }
            )

    plan_order = {key: index for index, key in enumerate(normalized_plan_keys)}
    placements.sort(
        key=lambda placement: (
            plan_order[placement["plan_key"]],
            placement["catalog_id"],
            placement["plan_id"],
            placement["placement_id"],
            placement["course_id"],
        )
    )
    found_plan_keys = {placement["plan_key"] for placement in placements}
    missing_plan_keys = [
        plan_key
        for plan_key in normalized_plan_keys
        if plan_key not in found_plan_keys
    ]
    status = (
        "no_data"
        if not placements
        else "partial"
        if missing_plan_keys
        else "ok"
    )
    return {
        "status": status,
        "program": normalized_program,
        "course_code": normalized_code,
        "requested_plan_keys": normalized_plan_keys,
        "missing_plan_keys": missing_plan_keys,
        "placements": placements,
    }


def alternative_group_placements(
    db_path: Database,
    program: str,
    course_codes: Iterable[str],
    plan_keys: str | Iterable[str],
) -> dict[str, Any]:
    """Return grouped placements matching multiple alternative members."""
    raw_course_codes = list(course_codes)
    if isinstance(course_codes, str) or not raw_course_codes:
        raise ValueError("course_codes must contain at least two codes")
    normalized_program, _, normalized_plan_keys = (
        _normalize_course_placement_inputs(program, raw_course_codes[0], plan_keys)
    )
    normalized_codes: list[str] = []
    for course_code in raw_course_codes:
        if not isinstance(course_code, str) or _COURSE_CODE_RE.fullmatch(course_code) is None:
            raise ValueError("course_codes must contain exactly 8 ASCII digits")
        if course_code not in normalized_codes:
            normalized_codes.append(course_code)
    if len(normalized_codes) < 2:
        raise ValueError("course_codes must contain at least two distinct codes")

    code_placeholders = ", ".join("?" for _ in normalized_codes)
    plan_placeholders = ", ".join("?" for _ in normalized_plan_keys)
    query = f"""
        SELECT
            plan_courses.placement_id,
            plan_courses.plan_key,
            plan_courses.plan_id,
            plans.catalog_id,
            plan_courses.course_id,
            plan_courses.course_code,
            courses.name_th,
            courses.name_en,
            plan_courses.year,
            plan_courses.semester,
            plan_courses.flexible_year_semester_raw,
            plan_courses.credits_raw,
            plan_courses.alternative_group_id,
            groups.group_key,
            groups.label,
            groups.minimum_choices,
            groups.maximum_choices,
            groups.notes AS group_notes
        FROM v_plan_courses AS plan_courses
        JOIN curriculum_plans AS plans
          ON plans.plan_id = plan_courses.plan_id
        JOIN courses
          ON courses.course_id = plan_courses.course_id
        LEFT JOIN alternative_course_groups AS groups
          ON groups.alternative_group_id = plan_courses.alternative_group_id
        WHERE plan_courses.program = ?
          AND plan_courses.course_code IN ({code_placeholders})
          AND plan_courses.plan_key IN ({plan_placeholders})
        ORDER BY
            plan_courses.plan_key,
            plans.catalog_id,
            plan_courses.plan_id,
            plan_courses.placement_id,
            plan_courses.course_id
    """

    with _open_database(db_path) as connection:
        rows = connection.execute(
            query,
            (
                normalized_program,
                *normalized_codes,
                *normalized_plan_keys,
            ),
        ).fetchall()
        placements: list[dict[str, Any]] = []
        seen_placements: set[tuple[str, int, int]] = set()
        for row in rows:
            group_id = row["alternative_group_id"]
            if group_id is None:
                continue
            group_id = int(group_id)
            members = _alternative_members(connection, group_id)
            member_codes = {member["course_code"] for member in members}
            if not set(normalized_codes).issubset(member_codes):
                continue

            placement_key = (
                row["plan_key"],
                int(row["plan_id"]),
                int(row["placement_id"]),
            )
            if placement_key in seen_placements:
                continue
            seen_placements.add(placement_key)
            references = _merge_provenance(
                _provenance_for(
                    connection,
                    "plan_placement_provenance",
                    "placement_id",
                    int(row["placement_id"]),
                ),
                _provenance_for(
                    connection,
                    "alternative_group_provenance",
                    "alternative_group_id",
                    group_id,
                ),
                *(member["provenance"] for member in members),
            )
            placements.append(
                {
                    "placement_id": int(row["placement_id"]),
                    "plan_key": row["plan_key"],
                    "plan_id": int(row["plan_id"]),
                    "catalog_id": int(row["catalog_id"]),
                    "course_id": int(row["course_id"]),
                    "course_code": row["course_code"],
                    "name_th": row["name_th"],
                    "name_en": row["name_en"],
                    "year": row["year"],
                    "semester": row["semester"],
                    "flexible_year_semester_raw": row[
                        "flexible_year_semester_raw"
                    ],
                    "year_semester_choices": placement_year_semester_choices(
                        row["year"],
                        row["semester"],
                        row["flexible_year_semester_raw"],
                    ),
                    "credits_raw": row["credits_raw"],
                    "alternative_group_id": group_id,
                    "group_key": row["group_key"],
                    "label": row["label"],
                    "minimum_choices": row["minimum_choices"],
                    "maximum_choices": row["maximum_choices"],
                    "group_notes": row["group_notes"],
                    "alternative_courses": members,
                    "provenance": references,
                }
            )

    plan_order = {key: index for index, key in enumerate(normalized_plan_keys)}
    placements.sort(
        key=lambda placement: (
            plan_order[placement["plan_key"]],
            placement["catalog_id"],
            placement["plan_id"],
            placement["placement_id"],
            placement["course_id"],
        )
    )
    found_plan_keys = {placement["plan_key"] for placement in placements}
    missing_plan_keys = [
        plan_key
        for plan_key in normalized_plan_keys
        if plan_key not in found_plan_keys
    ]
    return {
        "status": (
            "no_data"
            if not placements
            else "partial"
            if missing_plan_keys
            else "ok"
        ),
        "program": normalized_program,
        "course_codes": normalized_codes,
        "requested_plan_keys": normalized_plan_keys,
        "missing_plan_keys": missing_plan_keys,
        "placements": placements,
    }


def course_facts(
    db_path: Database,
    course_code: str,
    program: str | None = None,
) -> dict[str, Any]:
    """Return exact course facts and directly linked provenance."""
    if not isinstance(course_code, str) or _COURSE_CODE_RE.fullmatch(course_code) is None:
        raise ValueError("course_code must contain exactly 8 ASCII digits")
    if program is not None:
        if not isinstance(program, str) or not program.strip():
            raise ValueError("program must be a non-empty string when provided")
        normalized_program = program.strip().upper()
    else:
        normalized_program = None

    with _open_database(db_path) as connection:
        where = ["courses.course_code_normalized = ?"]
        parameters: list[Any] = [course_code]
        if normalized_program is not None:
            where.append(
                "EXISTS ("
                "SELECT 1 FROM programs AS scoped_programs "
                "WHERE scoped_programs.catalog_id = courses.catalog_id "
                "AND scoped_programs.program_code_normalized = ?"
                ")"
            )
            parameters.append(normalized_program.casefold())
        rows = connection.execute(
            f"""
            SELECT {_course_select('courses')}, courses.credit_units
            FROM courses
            WHERE {' AND '.join(where)}
            ORDER BY courses.catalog_id, courses.course_id
            """,
            parameters,
        ).fetchall()

        facts: list[dict[str, Any]] = []
        for row in rows:
            course_id = int(row["course_id"])
            references = _provenance_for(
                connection, "course_provenance", "course_id", course_id
            )
            placement_rows = connection.execute(
                """
                SELECT DISTINCT
                    plan_courses.placement_id,
                    plan_courses.plan_key,
                    plan_courses.plan_id,
                    plans.catalog_id,
                    plan_courses.year,
                    plan_courses.semester,
                    plan_courses.flexible_year_semester_raw,
                    plan_courses.alternative_group_id
                FROM v_plan_courses AS plan_courses
                JOIN curriculum_plans AS plans
                  ON plans.plan_id = plan_courses.plan_id
                WHERE plan_courses.course_id = ?
                """
                + (
                    " AND plan_courses.program = ?"
                    if normalized_program is not None
                    else ""
                )
                + " ORDER BY plan_courses.plan_key, plans.catalog_id, "
                "plan_courses.plan_id, plan_courses.placement_id",
                (course_id, normalized_program)
                if normalized_program is not None
                else (course_id,),
            ).fetchall()
            placements: list[dict[str, Any]] = []
            for placement_row in placement_rows:
                placement_id = int(placement_row["placement_id"])
                placement = {
                    "placement_id": placement_id,
                    "plan_key": placement_row["plan_key"],
                    "plan_id": int(placement_row["plan_id"]),
                    "catalog_id": int(placement_row["catalog_id"]),
                    "year": placement_row["year"],
                    "semester": placement_row["semester"],
                    "flexible_year_semester_raw": placement_row[
                        "flexible_year_semester_raw"
                    ],
                    "alternative_group_id": placement_row["alternative_group_id"],
                }
                placements.append(placement)
                references = _merge_provenance(
                    references,
                    _provenance_for(
                        connection,
                        "plan_placement_provenance",
                        "placement_id",
                        placement_id,
                    ),
                )

            program_rows = connection.execute(
                """
                SELECT DISTINCT program_code
                FROM programs
                WHERE catalog_id = ?
                ORDER BY program_code
                """,
                (int(row["catalog_id"]),),
            ).fetchall()
            facts.append(
                {
                    **_course_fields(row),
                    "credit_units": row["credit_units"],
                    "programs": [item["program_code"] for item in program_rows],
                    "placements": placements,
                    "provenance": references,
                }
            )

    return {
        "status": "ok" if facts else "no_data",
        "program": normalized_program,
        "course_code": course_code,
        "courses": facts,
    }


def _normalize_course_name(value: str) -> str:
    return " ".join(value.casefold().split())


def _course_name_matches(
    reference: str, candidate: str | None, *, exact_only: bool = False
) -> bool:
    if not candidate:
        return False
    normalized_reference = _normalize_course_name(reference)
    normalized_candidate = _normalize_course_name(candidate)
    if not normalized_reference or not normalized_candidate:
        return False
    if normalized_reference == normalized_candidate:
        return True
    if exact_only:
        return False

    reference_tokens = _COURSE_NAME_TOKEN_RE.findall(normalized_reference)
    candidate_tokens = _COURSE_NAME_TOKEN_RE.findall(normalized_candidate)
    if not reference_tokens or len(reference_tokens) > len(candidate_tokens):
        return False
    width = len(reference_tokens)
    return any(
        candidate_tokens[index : index + width] == reference_tokens
        for index in range(len(candidate_tokens) - width + 1)
    )


def _append_distinct_non_empty(values: list[str], value: Any) -> None:
    if isinstance(value, str) and value.strip() and value not in values:
        values.append(value)


def exact_course_candidates(
    db_path: Database,
    *,
    course_code: str | None = None,
    course_name: str | None = None,
    program: str | None = None,
    exact_title: bool = False,
) -> list[dict[str, Any]]:
    """Return exact relational course identities for one code or name reference.

    Course names use deterministic lexical matching against canonical ``name_th``
    and ``name_en`` values. Results that differ only by plan, catalog duplicate,
    or repeated relational joins are collapsed by logical ``(program, code)``.
    This helper only returns candidates; it does not decide ambiguity or policy.

    Pass ``exact_title=True`` to skip contiguous token-subsequence matching and
    keep only normalized title equality. Course-code lookup is unaffected.
    """
    if (course_code is None) == (course_name is None):
        raise ValueError("provide exactly one of course_code or course_name")
    if program is not None:
        if not isinstance(program, str) or not program.strip():
            raise ValueError("program must be a non-empty string when provided")
        normalized_program = program.strip().casefold()
    else:
        normalized_program = None

    if course_code is not None:
        if not isinstance(course_code, str):
            return []
        normalized_code = course_code.strip()
        if _COURSE_CODE_RE.fullmatch(normalized_code) is None:
            return []
    else:
        if not isinstance(course_name, str) or not course_name.strip():
            return []
        normalized_code = None

    with _open_database(db_path) as connection:
        where = ["courses.course_code_normalized = ?"] if normalized_code else []
        parameters: list[Any] = [normalized_code] if normalized_code else []
        if normalized_program is not None:
            where.append("programs.program_code_normalized = ?")
            parameters.append(normalized_program)
        rows = connection.execute(
            f"""
            SELECT
                courses.course_id,
                courses.catalog_id,
                courses.course_code,
                courses.course_code_normalized,
                courses.name_th,
                courses.name_en,
                programs.program_code,
                programs.program_code_normalized
            FROM courses
            JOIN programs ON programs.catalog_id = courses.catalog_id
            {f"WHERE {' AND '.join(where)}" if where else ""}
            ORDER BY programs.program_code_normalized,
                     courses.course_code_normalized,
                     courses.catalog_id,
                     courses.course_id
            """,
            parameters,
        ).fetchall()

        matched_keys: set[tuple[str, str]] = set()
        if normalized_code is None:
            for row in rows:
                logical_key = (
                    str(row["program_code_normalized"]),
                    str(row["course_code_normalized"]),
                )
                if _course_name_matches(
                    course_name or "", row["name_th"], exact_only=exact_title
                ) or _course_name_matches(
                    course_name or "", row["name_en"], exact_only=exact_title
                ):
                    matched_keys.add(logical_key)

        candidates_by_key: dict[tuple[str, str], dict[str, Any]] = {}
        for row in rows:
            logical_key = (
                str(row["program_code_normalized"]),
                str(row["course_code_normalized"]),
            )
            if normalized_code is None and logical_key not in matched_keys:
                continue
            candidate = candidates_by_key.get(logical_key)
            if candidate is None:
                candidate = {
                    "course_id": int(row["course_id"]),
                    "catalog_id": int(row["catalog_id"]),
                    "program": row["program_code"],
                    "course_code": row["course_code"],
                    "_name_th_values": [],
                    "_name_en_values": [],
                    "provenance": [],
                }
                candidates_by_key[logical_key] = candidate
            _append_distinct_non_empty(candidate["_name_th_values"], row["name_th"])
            _append_distinct_non_empty(candidate["_name_en_values"], row["name_en"])
            candidate["provenance"] = _merge_provenance(
                candidate["provenance"],
                _provenance_for(
                    connection,
                    "course_provenance",
                    "course_id",
                    int(row["course_id"]),
                ),
            )

    candidates = list(candidates_by_key.values())
    for candidate in candidates:
        name_th_values = candidate.pop("_name_th_values")
        name_en_values = candidate.pop("_name_en_values")
        candidate["name_th"] = name_th_values[0] if len(name_th_values) == 1 else None
        candidate["name_en"] = name_en_values[0] if len(name_en_values) == 1 else None
        candidate["name_th_variants"] = name_th_values
        candidate["name_en_variants"] = name_en_values
    return candidates


def semester_total_credits(
    db_path: Database,
    plan_id: int,
    year_number: int,
    semester_number: int,
) -> int | float:
    """Return credits for a plan term, counting each placement once."""
    with _open_database(db_path) as connection:
        total = sum(
            (
                _credits_for_placement(connection, dict(placement))
                for placement in _placement_rows(
                    connection, plan_id, year_number, semester_number
                )
            ),
            Decimal(0),
        )
    return int(total) if total == total.to_integral_value() else float(total)


def _normalize_semester_credit_inputs(
    program: str,
    plan_key: str,
    year_number: int,
    semester_number: int,
) -> tuple[str, str, int, int]:
    if not isinstance(program, str) or not program.strip():
        raise ValueError("program must be a non-empty string")
    if not isinstance(plan_key, str):
        raise ValueError("plan_key must be a string")
    normalized_plan_key = plan_key.strip().lower()
    if normalized_plan_key not in _CANONICAL_PLAN_KEYS:
        raise ValueError(f"unsupported canonical plan_key: {plan_key!r}")
    if isinstance(year_number, bool) or not isinstance(year_number, int):
        raise ValueError("year_number must be an integer")
    if isinstance(semester_number, bool) or not isinstance(semester_number, int):
        raise ValueError("semester_number must be an integer")
    if not 1 <= year_number <= 5:
        raise ValueError("year_number must be between 1 and 5")
    if not 1 <= semester_number <= 2:
        raise ValueError("semester_number must be between 1 and 2")
    return program.strip().upper(), normalized_plan_key, year_number, semester_number


def _credit_target_filter(
    program: str,
    course_targets: Iterable[Mapping[str, Any]] | None,
) -> tuple[set[int], set[str]] | None:
    if course_targets is None:
        return None

    target_ids: set[int] = set()
    target_codes: set[str] = set()
    for target in course_targets:
        if not isinstance(target, Mapping):
            raise ValueError("course_targets must contain mappings")
        target_program = target.get("program")
        if target_program is not None:
            if not isinstance(target_program, str):
                raise ValueError("course target program must be a string")
            if target_program.strip().upper() != program:
                continue
        course_id = target.get("course_id")
        if isinstance(course_id, bool):
            raise ValueError("course target course_id must be an integer")
        if isinstance(course_id, int):
            target_ids.add(course_id)
        course_code = target.get("course_code")
        if course_code is not None:
            if not isinstance(course_code, str) or not course_code.strip():
                raise ValueError("course target course_code must be a string")
            target_codes.add(course_code.strip())
    return target_ids, target_codes


def _placement_matches_credit_targets(
    connection: sqlite3.Connection,
    placement: Mapping[str, Any],
    target_filter: tuple[set[int], set[str]],
) -> bool:
    placement = dict(placement)
    target_ids, target_codes = target_filter
    group_id = placement.get("alternative_group_id")
    if group_id is not None:
        return any(
            member.get("course_id") in target_ids
            or member.get("course_code") in target_codes
            for member in _alternative_members(connection, int(group_id))
        )
    return (
        placement.get("course_id") in target_ids
        or placement.get("course_code") in target_codes
    )


def get_semester_credits(
    db_path: Database,
    program: str,
    plan_key: str,
    year_number: int,
    semester_number: int,
    *,
    course_targets: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return one plan term's deterministic credit total and components.

    When ``course_targets`` is supplied, retain only placements containing
    one of those exact logical course identities.  Existing placement credit
    calculation remains authoritative, including alternative groups.
    """
    (
        normalized_program,
        normalized_plan_key,
        normalized_year,
        normalized_semester,
    ) = _normalize_semester_credit_inputs(
        program, plan_key, year_number, semester_number
    )
    target_filter = _credit_target_filter(normalized_program, course_targets)
    with _open_database(db_path) as connection:
        plan_rows = connection.execute(
            """
            SELECT plan_id, catalog_id, program_code, plan_key
            FROM curriculum_plans
            WHERE program_code = ? AND plan_key = ?
            ORDER BY plan_id
            """,
            (normalized_program, normalized_plan_key),
        ).fetchall()
        plans: list[dict[str, Any]] = []
        flat_components: list[dict[str, Any]] = []

        for plan_row in plan_rows:
            plan_id = int(plan_row["plan_id"])
            catalog_id = int(plan_row["catalog_id"])
            placements = _placement_rows(
                connection, plan_id, normalized_year, normalized_semester
            )
            if target_filter is not None:
                placements = [
                    placement
                    for placement in placements
                    if _placement_matches_credit_targets(
                        connection, placement, target_filter
                    )
                ]
            if not placements:
                continue

            components: list[dict[str, Any]] = []
            total = Decimal(0)
            for raw_placement in placements:
                placement = dict(raw_placement)
                group_id = placement.get("alternative_group_id")
                counted_credits = _credits_for_placement(connection, placement)
                total += counted_credits
                placement_id = int(placement["placement_id"])
                placement_references = _provenance_for(
                    connection,
                    "plan_placement_provenance",
                    "placement_id",
                    placement_id,
                )

                alternative_courses: list[dict[str, Any]] = []
                if group_id is None:
                    course_id = (
                        int(placement["course_id"])
                        if placement["course_id"] is not None
                        else None
                    )
                    references = _merge_provenance(
                        placement_references,
                        _provenance_for(
                            connection,
                            "course_provenance",
                            "course_id",
                            course_id,
                        )
                        if course_id is not None
                        else [],
                    )
                else:
                    alternative_courses = _alternative_members(
                        connection, int(group_id)
                    )
                    references = _merge_provenance(
                        placement_references,
                        _provenance_for(
                            connection,
                            "alternative_group_provenance",
                            "alternative_group_id",
                            int(group_id),
                        ),
                        *(member["provenance"] for member in alternative_courses),
                    )

                component = {
                    "placement_id": placement_id,
                    "course_id": (
                        int(placement["course_id"])
                        if placement["course_id"] is not None
                        else None
                    ),
                    "course_code": placement["course_code"],
                    "name_th": placement["name_th"],
                    "name_en": placement["name_en"],
                    "credits_raw": placement["credits_raw"],
                    "credit_units": (
                        _decimal_credits(placement["credits"])
                        if group_id is None
                        else None
                    ),
                    "counted_credit_units": (
                        int(counted_credits)
                        if counted_credits == counted_credits.to_integral_value()
                        else float(counted_credits)
                    ),
                    "alternative_group_id": (
                        int(group_id) if group_id is not None else None
                    ),
                    "alternative_courses": alternative_courses,
                    "year": placement["year_number"],
                    "semester": placement["semester_number"],
                    "provenance": references,
                }
                components.append(component)
                flat_components.append(
                    {
                        **component,
                        "plan_id": plan_id,
                        "catalog_id": catalog_id,
                        "program": plan_row["program_code"],
                        "plan_key": plan_row["plan_key"],
                        "total_credits": None,
                    }
                )

            numeric_total = (
                int(total) if total == total.to_integral_value() else float(total)
            )
            for component in components:
                component["total_credits"] = numeric_total
            for component in flat_components:
                if component["plan_id"] == plan_id:
                    component["total_credits"] = numeric_total
            plans.append(
                {
                    "plan_id": plan_id,
                    "catalog_id": catalog_id,
                    "program": plan_row["program_code"],
                    "plan_key": plan_row["plan_key"],
                    "year": normalized_year,
                    "semester": normalized_semester,
                    "total_credits": numeric_total,
                    "components": components,
                }
            )

    return {
        "status": "ok" if plans else "no_data",
        "program": normalized_program,
        "plan_key": normalized_plan_key,
        "year": normalized_year,
        "semester": normalized_semester,
        "total_credits": plans[0]["total_credits"] if len(plans) == 1 else None,
        "plans": plans,
        "components": flat_components,
    }


def semester_credits_and_prerequisites(
    db_path: Database,
    program: str,
    plan_key: str,
    year_number: int,
    semester_number: int,
    course_code: str,
) -> dict[str, Any]:
    """Return one plan term's credits and a course's prerequisites."""
    normalized_program, normalized_code, normalized_plan_keys = (
        _normalize_course_placement_inputs(program, course_code, [plan_key])
    )
    if isinstance(year_number, bool) or not isinstance(year_number, int):
        raise ValueError("year_number must be an integer")
    if isinstance(semester_number, bool) or not isinstance(semester_number, int):
        raise ValueError("semester_number must be an integer")

    normalized_plan_key = normalized_plan_keys[0]
    with _open_database(db_path) as connection:
        plan_rows = connection.execute(
            """
            SELECT plan_id, catalog_id, program_code, plan_key
            FROM curriculum_plans
            WHERE program_code = ? AND plan_key = ?
            ORDER BY plan_id
            """,
            (normalized_program, normalized_plan_key),
        ).fetchall()
        plans: list[dict[str, Any]] = []
        has_credits = False
        has_prerequisites = False

        for plan_row in plan_rows:
            plan_id = int(plan_row["plan_id"])
            catalog_id = int(plan_row["catalog_id"])
            credit_row = connection.execute(
                """
                SELECT total_credits
                FROM v_semester_credits
                WHERE plan_id = ? AND year = ? AND semester = ?
                """,
                (plan_id, year_number, semester_number),
            ).fetchone()
            total_credits = (
                credit_row["total_credits"] if credit_row is not None else None
            )
            has_credits = has_credits or total_credits is not None

            course_rows = connection.execute(
                """
                SELECT course_id
                FROM courses
                WHERE catalog_id = ? AND course_code = ?
                ORDER BY course_id
                """,
                (catalog_id, normalized_code),
            ).fetchall()
            placement_rows = connection.execute(
                """
                SELECT DISTINCT placement_id
                FROM v_plan_courses
                WHERE plan_id = ?
                  AND program = ?
                  AND course_code = ?
                  AND year = ?
                  AND semester = ?
                ORDER BY placement_id
                """,
                (
                    plan_id,
                    normalized_program,
                    normalized_code,
                    year_number,
                    semester_number,
                ),
            ).fetchall()
            placement_references = [
                reference
                for placement_row in placement_rows
                for reference in _provenance_for(
                    connection,
                    "plan_placement_provenance",
                    "placement_id",
                    int(placement_row["placement_id"]),
                )
            ]
            prerequisites: list[dict[str, Any]] = []
            course_ids: list[int] = []
            for course_row in course_rows:
                course_id = int(course_row["course_id"])
                course_ids.append(course_id)
                for prerequisite in _prerequisite_records(connection, course_id):
                    prerequisites.append(
                        {
                            **prerequisite,
                            "provenance": _merge_provenance(
                                placement_references,
                                prerequisite["provenance"],
                            ),
                        }
                    )
            has_prerequisites = has_prerequisites or bool(prerequisites)
            plans.append(
                {
                    "plan_id": plan_id,
                    "catalog_id": catalog_id,
                    "program": plan_row["program_code"],
                    "plan_key": plan_row["plan_key"],
                    "year": year_number,
                    "semester": semester_number,
                    "total_credits": total_credits,
                    "course_code": normalized_code,
                    "course_ids": course_ids,
                    "prerequisites": prerequisites,
                }
            )

    if not plans or not (has_credits or has_prerequisites):
        status = "no_data"
    elif has_credits and has_prerequisites:
        status = "ok"
    else:
        status = "partial"
    return {
        "status": status,
        "program": normalized_program,
        "plan_key": normalized_plan_key,
        "year": year_number,
        "semester": semester_number,
        "course_code": normalized_code,
        "plans": plans,
    }


def courses_in_year_semester(
    db_path: Database,
    plan_id: int,
    year_number: int,
    semester_number: int,
) -> list[dict[str, Any]]:
    """Return ordered placement records for one plan year and semester."""
    with _open_database(db_path) as connection:
        result: list[dict[str, Any]] = []
        for raw_row in _placement_rows(
            connection, plan_id, year_number, semester_number
        ):
            row = dict(raw_row)
            placement_id = int(row["placement_id"])
            group_id = row["alternative_group_id"]
            placement_references = _provenance_for(
                connection,
                "plan_placement_provenance",
                "placement_id",
                placement_id,
            )

            base = {
                "placement_id": placement_id,
                "plan_id": int(row["plan_id"]),
                "year_number": row["year_number"],
                "semester_number": row["semester_number"],
                "category": row["category"],
                "requirement_type": row["requirement_type"],
                "placement_order": row["placement_order"],
                "credits_override": row["credits_override"],
                "raw_text": row["raw_text"],
                "notes": row["notes"],
                "program_code": row["program_code"],
                "plan_code": row["plan_code"],
            }

            if group_id is None:
                course_id = int(row["course_id"])
                course_references = _provenance_for(
                    connection, "course_provenance", "course_id", course_id
                )
                references = _merge_provenance(
                    course_references, placement_references
                )
                result.append(
                    {
                        **base,
                        "course_id": course_id,
                        "alternative_group_id": None,
                        "is_alternative": False,
                        "course_code": row["course_code"],
                        "name_th": row["name_th"],
                        "name_en": row["name_en"],
                        "credits": row["credits"],
                        "placement_credits": row["credits_override"]
                        or row["credits"],
                        "provenance": references,
                        "source_pages": _source_pages(references),
                    }
                )
                continue

            group_id = int(group_id)
            members = _alternative_members(connection, group_id)
            group_references = _provenance_for(
                connection,
                "alternative_group_provenance",
                "alternative_group_id",
                group_id,
            )
            references = _merge_provenance(
                placement_references,
                group_references,
                *(member["provenance"] for member in members),
            )
            result.append(
                {
                    **base,
                    "course_id": None,
                    "alternative_group_id": group_id,
                    "is_alternative": True,
                    "course_code": None,
                    "name_th": None,
                    "name_en": None,
                    "credits": None,
                    "placement_credits": row["credits_override"],
                    "group_key": row["group_key"],
                    "label": row["label"],
                    "minimum_choices": row["minimum_choices"],
                    "maximum_choices": row["maximum_choices"],
                    "group_notes": row["group_notes"],
                    "alternative_courses": members,
                    "provenance": references,
                    "source_pages": _source_pages(references),
                }
            )
        return result


def scoped_course_set(
    db_path: Database,
    program: str,
    plan_keys: str | Iterable[str],
    *,
    years: Iterable[int] = (),
    semesters: Iterable[int] = (),
    category: str | None = None,
    course_targets: Iterable[Mapping[str, Any]] = (),
    exact_term_placements: bool = False,
) -> dict[str, Any]:
    """Return one deterministic, structurally scoped course-set relation.

    This helper materializes only the supplied structural filters.  It does
    not count, aggregate, compare, or perform semantic retrieval.  The
    executor is responsible for enumerating applicable plan or term values.
    """
    if not isinstance(program, str) or not program.strip():
        raise ValueError("program must be a non-empty string")
    normalized_program = program.strip().upper()
    raw_plan_keys = [plan_keys] if isinstance(plan_keys, str) else list(plan_keys)
    normalized_plan_keys: list[str] = []
    for plan_key in raw_plan_keys:
        if not isinstance(plan_key, str):
            raise ValueError("plan_keys must contain strings")
        normalized = plan_key.strip().lower()
        if normalized not in _CANONICAL_PLAN_KEYS:
            raise ValueError(f"unsupported canonical plan_key: {plan_key!r}")
        if normalized not in normalized_plan_keys:
            normalized_plan_keys.append(normalized)
    if not normalized_plan_keys:
        raise ValueError("at least one canonical plan_key is required")

    normalized_years = tuple(years)
    normalized_semesters = tuple(semesters)
    if any(
        isinstance(year, bool) or not isinstance(year, int) or not 1 <= year <= 5
        for year in normalized_years
    ):
        raise ValueError("years must contain integers between 1 and 5")
    if any(
        isinstance(semester, bool)
        or not isinstance(semester, int)
        or not 1 <= semester <= 2
        for semester in normalized_semesters
    ):
        raise ValueError("semesters must contain integers between 1 and 2")
    normalized_category = (
        category.strip().casefold()
        if isinstance(category, str) and category.strip()
        else None
    )
    is_elective_category = normalized_category == "วิชาเลือก"

    targets = tuple(course_targets)
    target_ids = {
        target.get("course_id")
        for target in targets
        if isinstance(target, Mapping)
        and isinstance(target.get("course_id"), int)
        and not isinstance(target.get("course_id"), bool)
    }
    target_codes = {
        str(target.get("course_code")).strip()
        for target in targets
        if isinstance(target, Mapping)
        and isinstance(target.get("course_code"), str)
        and target.get("course_code").strip()
    }
    plan_placeholders = ", ".join("?" for _ in normalized_plan_keys)
    query = f"""
        SELECT
            placements.placement_id,
            placements.plan_id,
            placements.course_id,
            placements.alternative_group_id,
            placements.year_number,
            placements.semester_number,
            placements.flexible_year_number,
            placements.flexible_semester_number,
            placements.flexible_year_semester_raw,
            placements.category,
            placements.requirement_type,
            placements.placement_order,
            placements.credits_override,
            placements.raw_text,
            placements.notes,
            plans.catalog_id,
            plans.program_code,
            plans.plan_key,
            {_course_select('courses')},
            groups.group_key,
            groups.label,
            groups.minimum_choices,
            groups.maximum_choices,
            groups.notes AS group_notes
        FROM plan_placements AS placements
        JOIN curriculum_plans AS plans ON plans.plan_id = placements.plan_id
        LEFT JOIN courses ON courses.course_id = placements.course_id
        LEFT JOIN alternative_course_groups AS groups
          ON groups.alternative_group_id = placements.alternative_group_id
        WHERE plans.program_code = ?
          AND plans.plan_key IN ({plan_placeholders})
        ORDER BY plans.plan_key, plans.catalog_id, plans.plan_id,
                 placements.placement_order IS NULL,
                 placements.placement_order, placements.placement_id
    """

    with _open_database(db_path) as connection:
        rows = connection.execute(
            query,
            (normalized_program, *normalized_plan_keys),
        ).fetchall()
        courses: list[dict[str, Any]] = []
        for raw_row in rows:
            row = dict(raw_row)
            choices = placement_year_semester_choices(
                row["year_number"],
                row["semester_number"],
                row["flexible_year_semester_raw"],
            )
            if exact_term_placements:
                if normalized_years and row["year_number"] not in normalized_years:
                    continue
                if normalized_semesters and row["semester_number"] not in normalized_semesters:
                    continue
            else:
                if normalized_years and not any(
                    choice[0] in normalized_years for choice in choices
                ):
                    continue
                if normalized_semesters and not any(
                    choice[1] in normalized_semesters for choice in choices
                ):
                    continue
                if normalized_years and normalized_semesters and not any(
                    choice[0] in normalized_years and choice[1] in normalized_semesters
                    for choice in choices
                ):
                    continue
            if normalized_category is not None and not is_elective_category and (
                not isinstance(row["category"], str)
                or row["category"].strip().casefold() != normalized_category
            ):
                continue

            alternative_group_id = row["alternative_group_id"]
            members: list[dict[str, Any]] = []
            if alternative_group_id is not None:
                members = _alternative_members(connection, int(alternative_group_id))
                if is_elective_category and not any(
                    isinstance(member.get("course_type"), str)
                    and member["course_type"].strip() == "เลือก"
                    for member in members
                ):
                    continue
                if targets and not any(
                    member.get("course_id") in target_ids
                    or member.get("course_code") in target_codes
                    for member in members
                ):
                    continue
            else:
                if is_elective_category and not (
                    isinstance(row["course_type"], str)
                    and row["course_type"].strip() == "เลือก"
                ):
                    continue
                if targets and (
                    row["course_id"] not in target_ids
                    and row["course_code"] not in target_codes
                ):
                    continue

            placement_id = int(row["placement_id"])
            placement_references = _provenance_for(
                connection,
                "plan_placement_provenance",
                "placement_id",
                placement_id,
            )
            if alternative_group_id is None:
                course_id = int(row["course_id"])
                references = _merge_provenance(
                    _provenance_for(
                        connection, "course_provenance", "course_id", course_id
                    ),
                    placement_references,
                )
                component = {
                    "placement_id": placement_id,
                    "plan_id": int(row["plan_id"]),
                    "catalog_id": int(row["catalog_id"]),
                    "program": row["program_code"],
                    "plan_key": row["plan_key"],
                    "year_number": row["year_number"],
                    "semester_number": row["semester_number"],
                    "year_semester_choices": choices,
                    "category": row["category"],
                    "requirement_type": row["requirement_type"],
                    "placement_order": row["placement_order"],
                    "credits_override": row["credits_override"],
                    "raw_text": row["raw_text"],
                    "notes": row["notes"],
                    "course_id": course_id,
                    "alternative_group_id": None,
                    "is_alternative": False,
                    "course_code": row["course_code"],
                    "name_th": row["name_th"],
                    "name_en": row["name_en"],
                    "course_type": row["course_type"],
                    "credits": row["credits"],
                    "placement_credits": row["credits_override"] or row["credits"],
                    "provenance": references,
                    "source_pages": _source_pages(references),
                }
            else:
                group_id = int(alternative_group_id)
                references = _merge_provenance(
                    placement_references,
                    _provenance_for(
                        connection,
                        "alternative_group_provenance",
                        "alternative_group_id",
                        group_id,
                    ),
                    *(member["provenance"] for member in members),
                )
                component = {
                    "placement_id": placement_id,
                    "plan_id": int(row["plan_id"]),
                    "catalog_id": int(row["catalog_id"]),
                    "program": row["program_code"],
                    "plan_key": row["plan_key"],
                    "year_number": row["year_number"],
                    "semester_number": row["semester_number"],
                    "year_semester_choices": choices,
                    "category": row["category"],
                    "requirement_type": row["requirement_type"],
                    "placement_order": row["placement_order"],
                    "credits_override": row["credits_override"],
                    "raw_text": row["raw_text"],
                    "notes": row["notes"],
                    "course_id": None,
                    "alternative_group_id": group_id,
                    "is_alternative": True,
                    "course_code": None,
                    "name_th": None,
                    "name_en": None,
                    "credits": None,
                    "placement_credits": row["credits_override"],
                    "group_key": row["group_key"],
                    "label": row["label"],
                    "minimum_choices": row["minimum_choices"],
                    "maximum_choices": row["maximum_choices"],
                    "group_notes": row["group_notes"],
                    "alternative_courses": members,
                    "provenance": references,
                    "source_pages": _source_pages(references),
                }
            courses.append(component)

    return {
        "status": "ok" if courses else "no_data",
        "program": normalized_program,
        "plan_keys": tuple(normalized_plan_keys),
        "years": normalized_years,
        "semesters": normalized_semesters,
        "category": category,
        "courses": courses,
    }


def applicable_plan_keys(
    db_path: Database,
    program: str,
) -> tuple[str, ...]:
    """Return deterministic plan keys available for one program."""
    if not isinstance(program, str) or not program.strip():
        raise ValueError("program must be a non-empty string")
    with _open_database(db_path) as connection:
        rows = connection.execute(
            """
            SELECT DISTINCT plan_key
            FROM curriculum_plans
            WHERE program_code = ?
            ORDER BY plan_id
            """,
            (program.strip().upper(),),
        ).fetchall()
    return tuple(str(row["plan_key"]) for row in rows)


def _prerequisite_records(
    connection: sqlite3.Connection, course_id: int
) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT
            prerequisites.prerequisite_id,
            prerequisites.course_id,
            prerequisites.prerequisite_course_id,
            prerequisites.alternative_group_id,
            prerequisites.prerequisite_order,
            prerequisites.requirement_type,
            prerequisites.raw_text,
            target.course_code AS prerequisite_code,
            target.name_th AS prerequisite_name_th,
            target.name_en AS prerequisite_name_en
        FROM prerequisites
        LEFT JOIN courses AS target
          ON target.course_id = prerequisites.prerequisite_course_id
        WHERE prerequisites.course_id = ?
        ORDER BY prerequisites.prerequisite_order,
                 prerequisites.prerequisite_id
        """,
        (course_id,),
    ).fetchall()

    records: list[dict[str, Any]] = []
    for raw_row in rows:
        row = dict(raw_row)
        prerequisite_id = int(row["prerequisite_id"])
        references = _provenance_for(
            connection,
            "prerequisite_provenance",
            "prerequisite_id",
            prerequisite_id,
        )
        group_id = row["alternative_group_id"]
        members: list[dict[str, Any]] = []
        group = None
        if group_id is not None:
            group_id = int(group_id)
            members = _alternative_members(connection, group_id)
            group_row = connection.execute(
                """
                SELECT group_key, label, minimum_choices, maximum_choices, notes
                FROM alternative_course_groups
                WHERE alternative_group_id = ?
                """,
                (group_id,),
            ).fetchone()
            group = {
                "alternative_group_id": group_id,
                "group_key": group_row["group_key"],
                "label": group_row["label"],
                "minimum_choices": group_row["minimum_choices"],
                "maximum_choices": group_row["maximum_choices"],
                "notes": group_row["notes"],
                "alternative_courses": members,
            }
            references = _merge_provenance(
                references,
                _provenance_for(
                    connection,
                    "alternative_group_provenance",
                    "alternative_group_id",
                    group_id,
                ),
                *(member["provenance"] for member in members),
            )
        elif row["prerequisite_course_id"] is not None:
            references = _merge_provenance(
                references,
                _provenance_for(
                    connection,
                    "course_provenance",
                    "course_id",
                    int(row["prerequisite_course_id"]),
                ),
            )

        records.append(
            {
                "prerequisite_id": prerequisite_id,
                "course_id": int(row["course_id"]),
                "prerequisite_course_id": row["prerequisite_course_id"],
                "prerequisite_code": row["prerequisite_code"],
                "prerequisite_name_th": row["prerequisite_name_th"],
                "prerequisite_name_en": row["prerequisite_name_en"],
                "alternative_group_id": group_id,
                "is_alternative": group_id is not None,
                "alternative_group": group,
                "alternative_courses": members,
                "prerequisite_order": row["prerequisite_order"],
                "requirement_type": row["requirement_type"],
                "raw_text": row["raw_text"],
                "provenance": references,
                "source_pages": _source_pages(references),
            }
        )
    return records


def prerequisites_of_course(
    db_path: Database, course_id: int
) -> list[dict[str, Any]]:
    """Return direct, alternative-group, and unresolved prerequisites by ID."""
    with _open_database(db_path) as connection:
        return _prerequisite_records(connection, course_id)


_EXPLICIT_NONE_PREREQUISITE_TEXT = frozenset(
    {
        "ไม่มี",
        "ไม่มีวิชาบังคับก่อน",
        "none",
        "no prerequisite",
        "no prerequisites",
        "-",
    }
)


def prerequisite_state(
    db_path: Database, course_id: int
) -> dict[str, Any]:
    """Classify one physical course's prerequisite evidence conservatively."""
    if isinstance(course_id, bool) or not isinstance(course_id, int):
        raise ValueError("course_id must be an integer")
    with _open_database(db_path) as connection:
        row = connection.execute(
            "SELECT prerequisite_text FROM courses WHERE course_id = ?",
            (course_id,),
        ).fetchone()
        if row is None:
            return {
                "state": "unknown",
                "records": (),
                "prerequisite_text": None,
                "provenance": (),
            }
        records = tuple(_prerequisite_records(connection, course_id))
        provenance = tuple(
            _provenance_for(connection, "course_provenance", "course_id", course_id)
        )
        prerequisite_text = row["prerequisite_text"]
        normalized_text = (
            " ".join(prerequisite_text.casefold().split())
            if isinstance(prerequisite_text, str)
            else ""
        )
        has_description_provenance = any(
            reference.get("document_category") == "description"
            for reference in provenance
        )
        if records:
            state = "required"
        elif (
            normalized_text in _EXPLICIT_NONE_PREREQUISITE_TEXT
            and has_description_provenance
        ):
            state = "explicit_none"
        else:
            state = "unknown"
        return {
            "state": state,
            "records": records,
            "prerequisite_text": prerequisite_text,
            "provenance": provenance,
        }


def courses_requiring_prerequisite(
    db_path: Database, prerequisite_course_id: int
) -> list[dict[str, Any]]:
    """Return each course that directly or alternatively requires a course ID."""
    with _open_database(db_path) as connection:
        rows = connection.execute(
            f"""
            SELECT DISTINCT {_course_select('courses')}
            FROM courses
            JOIN prerequisites
              ON prerequisites.course_id = courses.course_id
            LEFT JOIN alternative_course_group_members AS members
              ON members.alternative_group_id = prerequisites.alternative_group_id
            WHERE prerequisites.prerequisite_course_id = ?
               OR members.course_id = ?
            ORDER BY courses.course_id
            """,
            (prerequisite_course_id, prerequisite_course_id),
        ).fetchall()

        result: list[dict[str, Any]] = []
        for row in rows:
            course = _course_fields(row)
            all_prerequisites = _prerequisite_records(
                connection, int(row["course_id"])
            )
            matching = [
                prerequisite
                for prerequisite in all_prerequisites
                if prerequisite["prerequisite_course_id"] == prerequisite_course_id
                or any(
                    member["course_id"] == prerequisite_course_id
                    for member in prerequisite["alternative_courses"]
                )
            ]
            references = _merge_provenance(
                _provenance_for(
                    connection,
                    "course_provenance",
                    "course_id",
                    int(row["course_id"]),
                ),
                *(prerequisite["provenance"] for prerequisite in matching),
            )
            result.append(
                {
                    **course,
                    "matching_prerequisites": matching,
                    "provenance": references,
                    "source_pages": _source_pages(references),
                }
            )
        return result


__all__ = [
    "alternative_group_placements",
    "applicable_plan_keys",
    "course_facts",
    "course_placement",
    "courses_in_year_semester",
    "courses_requiring_prerequisite",
    "earliest_year_semester",
    "earliest_year_semester_from_choices",
    "exact_course_candidates",
    "get_semester_credits",
    "parse_flexible_year_semester",
    "placement_year_semester_choices",
    "prerequisites_of_course",
    "prerequisite_state",
    "semester_credits_and_prerequisites",
    "semester_total_credits",
    "scoped_course_set",
]
