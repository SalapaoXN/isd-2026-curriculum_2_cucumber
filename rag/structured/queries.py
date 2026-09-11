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
            plan_courses.year,
            plan_courses.semester,
            plan_courses.flexible_year_semester_raw,
            plan_courses.credits_raw,
            plan_courses.alternative_group_id
        FROM v_plan_courses AS plan_courses
        JOIN curriculum_plans AS plans
          ON plans.plan_id = plan_courses.plan_id
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
            placements.append(
                {
                    "placement_id": placement_id,
                    "plan_key": row["plan_key"],
                    "plan_id": int(row["plan_id"]),
                    "catalog_id": int(row["catalog_id"]),
                    "course_id": course_id,
                    "year": row["year"],
                    "semester": row["semester"],
                    "flexible_year_semester_raw": row[
                        "flexible_year_semester_raw"
                    ],
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
    "course_placement",
    "courses_in_year_semester",
    "courses_requiring_prerequisite",
    "prerequisites_of_course",
    "semester_credits_and_prerequisites",
    "semester_total_credits",
]
