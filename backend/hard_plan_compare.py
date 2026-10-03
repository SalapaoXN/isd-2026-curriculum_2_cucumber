"""Deterministic, provenance-carrying comparison of explicit curriculum plans."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from rag.structured.queries import _merge_provenance, _provenance_for


def _failure(
    status: str, program: Any, left_plan: Any, right_plan: Any, detail: str
) -> dict[str, Any]:
    return {
        "status": status,
        "program": program,
        "left_plan": left_plan,
        "right_plan": right_plan,
        "left_only_courses": [],
        "right_only_courses": [],
        "shared_course_count": None,
        "left_course_count": None,
        "right_course_count": None,
        "left_plan_evidence": [],
        "right_plan_evidence": [],
        "shared_courses": [],
        "left_courses": [],
        "right_courses": [],
        "limitations": [detail],
        "warnings": [],
    }


def _resolve_plan(
    connection: sqlite3.Connection, program: str, plan_key: str
) -> tuple[dict[str, Any] | None, str | None]:
    rows = connection.execute(
        """SELECT cp.plan_id, cp.catalog_id, cp.program_id,
                  cp.program_code, cp.plan_key
           FROM curriculum_plans AS cp
           JOIN programs AS p ON p.program_id = cp.program_id
           WHERE UPPER(cp.program_code) = ?
             AND UPPER(p.program_code_normalized) = ?
             AND p.catalog_id = cp.catalog_id
             AND LOWER(cp.plan_key) = ?
           ORDER BY cp.plan_id""",
        (program.upper(), program.upper(), plan_key.casefold()),
    ).fetchall()
    if not rows:
        return None, "plan_not_found"
    if len(rows) != 1:
        return None, "ambiguous_plan"
    row = rows[0]
    return {
        "plan_id": int(row["plan_id"]),
        "catalog_id": int(row["catalog_id"]),
        "program_id": int(row["program_id"]),
        "program": str(row["program_code"]),
        "plan": str(row["plan_key"]),
    }, None


def _course_provenance(
    connection: sqlite3.Connection,
    placement_id: int,
    course_id: int,
    alternative_group_id: int | None,
    alternative_group_member_id: int | None,
) -> list[dict[str, Any]]:
    references = [
        _provenance_for(
            connection, "plan_placement_provenance", "placement_id", placement_id
        ),
        _provenance_for(connection, "course_provenance", "course_id", course_id),
    ]
    if alternative_group_id is not None:
        references.append(
            _provenance_for(
                connection,
                "alternative_group_provenance",
                "alternative_group_id",
                alternative_group_id,
            )
        )
    if alternative_group_member_id is not None:
        references.append(
            _provenance_for(
                connection,
                "alternative_group_member_provenance",
                "alternative_group_member_id",
                alternative_group_member_id,
            )
        )
    return _merge_provenance(*references)


def _load_plan_courses(
    connection: sqlite3.Connection, plan: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], str | None]:
    plan_references = _provenance_for(
        connection, "curriculum_plan_provenance", "plan_id", plan["plan_id"]
    )
    rows = connection.execute(
        """SELECT pl.placement_id, pl.course_id, pl.alternative_group_id,
                  NULL AS group_catalog_id, NULL AS group_plan_id,
                  NULL AS alternative_group_member_id,
                  c.course_id AS member_course_id,
                  c.catalog_id AS course_catalog_id,
                  c.course_code, c.course_code_normalized, c.name_th, c.name_en
           FROM plan_placements AS pl
           LEFT JOIN courses AS c ON c.course_id = pl.course_id
           WHERE pl.plan_id = ? AND pl.course_id IS NOT NULL
           UNION ALL
           SELECT pl.placement_id, NULL AS course_id, pl.alternative_group_id,
                  ag.catalog_id AS group_catalog_id, ag.plan_id AS group_plan_id,
                  gm.alternative_group_member_id,
                  c.course_id AS member_course_id,
                  c.catalog_id AS course_catalog_id,
                  c.course_code, c.course_code_normalized, c.name_th, c.name_en
           FROM plan_placements AS pl
           LEFT JOIN alternative_course_groups AS ag
             ON ag.alternative_group_id = pl.alternative_group_id
           LEFT JOIN alternative_course_group_members AS gm
             ON gm.alternative_group_id = ag.alternative_group_id
           LEFT JOIN courses AS c ON c.course_id = gm.course_id
           WHERE pl.plan_id = ? AND pl.alternative_group_id IS NOT NULL
           ORDER BY placement_id, member_course_id""",
        (plan["plan_id"], plan["plan_id"]),
    ).fetchall()

    courses: dict[str, dict[str, Any]] = {}
    for row in rows:
        normalized_code = row["course_code_normalized"]
        if not isinstance(normalized_code, str) or not normalized_code.strip():
            return {}, "course membership has no normalized course code"
        if int(row["course_catalog_id"]) != plan["catalog_id"]:
            return {}, "course membership crosses the resolved catalog scope"
        if (
            row["group_catalog_id"] is not None
            and int(row["group_catalog_id"]) != plan["catalog_id"]
        ):
            return {}, "alternative group crosses the resolved catalog scope"
        if (
            row["group_plan_id"] is not None
            and int(row["group_plan_id"]) != plan["plan_id"]
        ):
            return {}, "alternative group crosses the resolved plan scope"

        identity = normalized_code
        placement_id = int(row["placement_id"])
        course_id = int(row["member_course_id"])
        group_id = (
            int(row["alternative_group_id"])
            if row["alternative_group_id"] is not None
            else None
        )
        member_id = (
            int(row["alternative_group_member_id"])
            if row["alternative_group_member_id"] is not None
            else None
        )
        item = courses.setdefault(
            identity,
            {
                "course_code_normalized": identity,
                "_course_codes": set(),
                "_names_th": set(),
                "_names_en": set(),
                "placement_ids": set(),
                "provenance": [dict(reference) for reference in plan_references],
            },
        )
        item["_course_codes"].add(str(row["course_code"]))
        if row["name_th"]:
            item["_names_th"].add(str(row["name_th"]))
        if row["name_en"]:
            item["_names_en"].add(str(row["name_en"]))
        item["placement_ids"].add(placement_id)
        membership_references = _course_provenance(
            connection, placement_id, course_id, group_id, member_id
        )
        item["_membership_provenance"] = _merge_provenance(
            item.get("_membership_provenance", []), membership_references
        )
        item["provenance"] = _merge_provenance(
            item["provenance"],
            membership_references,
        )

    for identity, item in courses.items():
        if not item["_membership_provenance"]:
            return {}, f"course {identity} has no linked membership source evidence"
        item["course_code"] = sorted(item.pop("_course_codes"))[0]
        names_th = sorted(item.pop("_names_th"))
        names_en = sorted(item.pop("_names_en"))
        item["name_th"] = names_th[0] if len(names_th) == 1 else None
        item["name_en"] = names_en[0] if len(names_en) == 1 else None
        if len(names_th) > 1 or len(names_en) > 1:
            item["name_conflict"] = True
        item["placement_ids"] = sorted(item["placement_ids"])
        item["provenance"] = sorted(
            item["provenance"], key=lambda reference: int(reference["provenance_id"])
        )
        item.pop("_membership_provenance")
    return courses, None


def _public_course(course: dict[str, Any], plan: str) -> dict[str, Any]:
    result = {
        "course_code": course["course_code"],
        "course_code_normalized": course["course_code_normalized"],
        "name_th": course["name_th"],
        "name_en": course["name_en"],
        "plan": plan,
        "placement_ids": list(course["placement_ids"]),
        "provenance": [dict(reference) for reference in course["provenance"]],
    }
    if course.get("name_conflict"):
        result["name_conflict"] = True
    return result


def compare_plan_course_sets(
    db_path: str | Path,
    program: str,
    left_plan: str,
    right_plan: str,
) -> dict[str, Any]:
    """Compare two explicitly named plans within one program using read-only SQLite."""
    requested = (program, left_plan, right_plan)
    if any(not isinstance(value, str) or not value.strip() for value in requested):
        return _failure(
            "invalid_scope", program, left_plan, right_plan,
            "program and both plan names must be explicitly provided",
        )
    normalized_program = program.strip().upper()
    normalized_left = left_plan.strip()
    normalized_right = right_plan.strip()
    if normalized_left.casefold() == normalized_right.casefold():
        return _failure(
            "invalid_scope", normalized_program, normalized_left, normalized_right,
            "the two plans must be distinct",
        )

    try:
        uri = f"{Path(db_path).resolve().as_uri()}?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as connection:
            connection.row_factory = sqlite3.Row
            left, left_error = _resolve_plan(
                connection, normalized_program, normalized_left
            )
            right, right_error = _resolve_plan(
                connection, normalized_program, normalized_right
            )
            if left_error or right_error:
                status = (
                    "ambiguous_plan"
                    if "ambiguous_plan" in (left_error, right_error)
                    else "plan_not_found"
                )
                return _failure(
                    status,
                    normalized_program,
                    normalized_left,
                    normalized_right,
                    "each explicit plan must resolve to exactly one plan in the selected program",
                )
            if left["program"].casefold() != right["program"].casefold():
                return _failure(
                    "incompatible_scope",
                    normalized_program,
                    left["plan"],
                    right["plan"],
                    "plans do not belong to the same canonical program scope",
                )

            left_plan_evidence = _provenance_for(
                connection, "curriculum_plan_provenance", "plan_id", left["plan_id"]
            )
            right_plan_evidence = _provenance_for(
                connection, "curriculum_plan_provenance", "plan_id", right["plan_id"]
            )
            if not left_plan_evidence or not right_plan_evidence:
                return _failure(
                    "insufficient_evidence",
                    normalized_program,
                    left["plan"],
                    right["plan"],
                    "one or both plan scopes lack linked source evidence",
                )

            left_courses, left_error = _load_plan_courses(connection, left)
            right_courses, right_error = _load_plan_courses(connection, right)
            if left_error or right_error:
                return _failure(
                    "insufficient_evidence",
                    normalized_program,
                    left["plan"],
                    right["plan"],
                    left_error or right_error,
                )
    except (OSError, sqlite3.Error, ValueError) as error:
        # Keep runtime errors controlled; do not return SQL, paths, or DB details.
        return _failure(
            "database_error", normalized_program, normalized_left, normalized_right,
            f"read-only curriculum evidence is unavailable ({type(error).__name__})",
        )

    left_ids = set(left_courses)
    right_ids = set(right_courses)
    shared_ids = sorted(left_ids & right_ids)
    left_only_ids = sorted(left_ids - right_ids)
    right_only_ids = sorted(right_ids - left_ids)
    warnings = []
    for item in (*left_courses.values(), *right_courses.values()):
        if item.get("name_conflict"):
            warnings.append("conflicting course names for a normalized course code")
        if any(
            reference.get("document_page") is None for reference in item["provenance"]
        ):
            warnings.append("document_page is unavailable for some linked source evidence")
        if any(
            reference.get("source_filename") is None
            or reference.get("source_page") is None
            for reference in item["provenance"]
        ):
            warnings.append("source filename or source page is unavailable for some evidence")
    warnings = list(dict.fromkeys(warnings))

    return {
        "status": "complete",
        "program": left["program"],
        "left_plan": left["plan"],
        "right_plan": right["plan"],
        "left_course_count": len(left_ids),
        "right_course_count": len(right_ids),
        "shared_course_count": len(shared_ids),
        "left_only_courses": [
            _public_course(left_courses[key], left["plan"]) for key in left_only_ids
        ],
        "right_only_courses": [
            _public_course(right_courses[key], right["plan"]) for key in right_only_ids
        ],
        "shared_courses": [
            {
                "course_code": left_courses[key]["course_code"],
                "course_code_normalized": key,
                "name_th": left_courses[key]["name_th"],
                "name_en": left_courses[key]["name_en"],
                "left_provenance": [
                    dict(ref) for ref in left_courses[key]["provenance"]
                ],
                "right_provenance": [
                    dict(ref) for ref in right_courses[key]["provenance"]
                ],
            }
            for key in shared_ids
        ],
        "left_courses": [
            _public_course(left_courses[key], left["plan"]) for key in sorted(left_ids)
        ],
        "right_courses": [
            _public_course(right_courses[key], right["plan"])
            for key in sorted(right_ids)
        ],
        "left_plan_evidence": left_plan_evidence,
        "right_plan_evidence": right_plan_evidence,
        "limitations": [],
        "warnings": warnings,
    }
