"""Deterministic, provenance-carrying comparison of explicit curriculum plans."""

from __future__ import annotations

import sqlite3
import re
from contextlib import closing
from pathlib import Path
from typing import Any
import unicodedata

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
    connection: sqlite3.Connection,
    program: str,
    plan_key: str,
    catalog_key: str | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    rows = connection.execute(
        """SELECT cp.plan_id, cp.catalog_id, cp.program_id,
                  cp.program_code, cp.plan_key, c.catalog_key
           FROM curriculum_plans AS cp
           JOIN programs AS p ON p.program_id = cp.program_id
           JOIN catalogs AS c ON c.catalog_id = cp.catalog_id
           WHERE UPPER(cp.program_code) = ?
             AND UPPER(p.program_code_normalized) = ?
             AND p.catalog_id = cp.catalog_id
             AND LOWER(cp.plan_key) = ?
             AND (? IS NULL OR LOWER(TRIM(c.catalog_key)) = LOWER(TRIM(?)))
           ORDER BY cp.plan_id""",
        (
            program.upper(), program.upper(), plan_key.casefold(),
            catalog_key, catalog_key,
        ),
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
        "catalog_key": str(row["catalog_key"]),
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
    catalog_key: str | None = None,
) -> dict[str, Any]:
    """Compare two explicitly named plans within one program using read-only SQLite."""
    requested = (program, left_plan, right_plan)
    if any(not isinstance(value, str) or not value.strip() for value in requested):
        return _failure(
            "invalid_scope", program, left_plan, right_plan,
            "program and both plan names must be explicitly provided",
        )
    if catalog_key is not None and (
        not isinstance(catalog_key, str) or not catalog_key.strip()
    ):
        return _failure(
            "invalid_scope", program, left_plan, right_plan,
            "catalog_key must be a non-empty string when provided",
        )
    catalog_key = catalog_key.strip() if catalog_key is not None else None
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
                connection, normalized_program, normalized_left, catalog_key
            )
            right, right_error = _resolve_plan(
                connection, normalized_program, normalized_right, catalog_key
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
        "catalog_key": left["catalog_key"],
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


def _edition_failure(status: str, program: Any, reason: str) -> dict[str, Any]:
    return {
        "status": status,
        "program": program,
        "older": None,
        "newer": None,
        "categories": {
            "shared_same_code": [],
            "old_only_by_code": [],
            "new_only_by_code": [],
            "same_name_changed_code_candidates": [],
            "unresolved_non_concrete": [],
        },
        "counts": {},
        "provenance": [],
        "limitations": [reason],
    }


def _normalized_course_name(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = unicodedata.normalize("NFKC", value)
    return " ".join(normalized.split()).casefold()


def _courses_in_edition(
    connection: sqlite3.Connection, catalog_id: int, program: str
) -> tuple[dict[str, list[dict[str, Any]]], bool]:
    rows = connection.execute(
        """SELECT cp.plan_key, pl.placement_id, c.course_id,
                  c.course_code, c.course_code_normalized, c.name_th, c.name_en
           FROM curriculum_plans cp
           JOIN programs p ON p.program_id=cp.program_id AND p.catalog_id=cp.catalog_id
           JOIN plan_placements pl ON pl.plan_id=cp.plan_id
           JOIN courses c ON c.course_id=pl.course_id
           WHERE cp.catalog_id=? AND UPPER(p.program_code_normalized)=?
           UNION ALL
           SELECT cp.plan_key, pl.placement_id, c.course_id,
                  c.course_code, c.course_code_normalized, c.name_th, c.name_en
           FROM curriculum_plans cp
           JOIN programs p ON p.program_id=cp.program_id AND p.catalog_id=cp.catalog_id
           JOIN plan_placements pl ON pl.plan_id=cp.plan_id
           JOIN alternative_course_group_members m
             ON m.alternative_group_id=pl.alternative_group_id
           JOIN courses c ON c.course_id=m.course_id
           WHERE cp.catalog_id=? AND UPPER(p.program_code_normalized)=?
           ORDER BY 5, 3, 1, 2""",
        (catalog_id, program.upper(), catalog_id, program.upper()),
    ).fetchall()
    by_course_id: dict[int, dict[str, Any]] = {}
    course_provenance: dict[int, list[dict[str, Any]]] = {}
    placement_provenance: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        course_id = int(row["course_id"])
        placement_id = int(row["placement_id"])
        course = by_course_id.setdefault(
            course_id,
            {
                "course_id": course_id,
                "course_code": str(row["course_code"]),
                "course_code_normalized": str(row["course_code_normalized"]),
                "name_th": row["name_th"],
                "name_en": row["name_en"],
                "plans": set(),
                "provenance": [],
            },
        )
        course["plans"].add(str(row["plan_key"]))
        course_provenance.setdefault(
            course_id,
            _provenance_for(connection, "course_provenance", "course_id", course_id),
        )
        placement_provenance.setdefault(
            placement_id,
            _provenance_for(
                connection, "plan_placement_provenance", "placement_id", placement_id
            ),
        )
        course["provenance"] = _merge_provenance(
            course["provenance"],
            course_provenance[course_id],
            placement_provenance[placement_id],
        )

    by_code: dict[str, list[dict[str, Any]]] = {}
    complete = True
    for course in by_course_id.values():
        course["plans"] = sorted(course["plans"], key=str.casefold)
        if not course["provenance"]:
            complete = False
        by_code.setdefault(course["course_code_normalized"], []).append(course)
    for courses in by_code.values():
        courses.sort(key=lambda item: (item["course_code"], item["course_id"]))
    return by_code, complete


def compare_curriculum_editions(
    db_path: str | Path, program: str
) -> dict[str, Any]:
    """Compare course-code sets in the two canonical editions of one program."""
    if not isinstance(program, str) or not program.strip():
        return _edition_failure("invalid_scope", program, "program must be explicit")
    normalized_program = program.strip().upper()
    try:
        uri = f"{Path(db_path).resolve().as_uri()}?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as connection:
            connection.row_factory = sqlite3.Row
            catalog_rows = connection.execute(
                """SELECT DISTINCT c.catalog_id, c.catalog_key, c.academic_year
                   FROM catalogs c
                   JOIN programs p ON p.catalog_id=c.catalog_id
                   WHERE UPPER(p.program_code_normalized)=?
                   ORDER BY c.academic_year, c.catalog_key, c.catalog_id""",
                (normalized_program,),
            ).fetchall()
            if len(catalog_rows) < 2:
                return _edition_failure(
                    "insufficient_editions", normalized_program,
                    "two curriculum editions are required for comparison",
                )
            if len(catalog_rows) > 2:
                return _edition_failure(
                    "ambiguous_edition", normalized_program,
                    "more than two curriculum editions exist; select the editions explicitly",
                )
            editions = []
            for row in catalog_rows:
                key = row["catalog_key"]
                year = row["academic_year"]
                if not isinstance(key, str) or not key.strip():
                    return _edition_failure(
                        "ambiguous_edition", normalized_program,
                        "a curriculum edition has no canonical catalog_key",
                    )
                try:
                    year_number = int(str(year).strip())
                except (TypeError, ValueError):
                    return _edition_failure(
                        "ambiguous_edition", normalized_program,
                        "a curriculum edition has no numeric academic_year",
                    )
                editions.append(
                    {
                        "catalog_id": int(row["catalog_id"]),
                        "catalog_key": key.strip(),
                        "academic_year": str(year).strip(),
                        "year_number": year_number,
                    }
                )
            if editions[0]["year_number"] == editions[1]["year_number"]:
                return _edition_failure(
                    "ambiguous_edition", normalized_program,
                    "curriculum editions do not have distinct academic_year values",
                )
            older, newer = sorted(editions, key=lambda item: item["year_number"])
            old_courses, old_complete = _courses_in_edition(
                connection, older["catalog_id"], normalized_program
            )
            new_courses, new_complete = _courses_in_edition(
                connection, newer["catalog_id"], normalized_program
            )
    except (OSError, sqlite3.Error, ValueError) as error:
        return _edition_failure(
            "database_error", normalized_program,
            f"read-only curriculum evidence is unavailable ({type(error).__name__})",
        )

    concrete_code = re.compile(r"\d{8}\Z")
    old_non_concrete = {
        code: old_courses[code] for code in sorted(old_courses)
        if concrete_code.fullmatch(code) is None
    }
    new_non_concrete = {
        code: new_courses[code] for code in sorted(new_courses)
        if concrete_code.fullmatch(code) is None
    }
    categories: dict[str, list[dict[str, Any]]] = {
        "shared_same_code": [],
        "old_only_by_code": [],
        "new_only_by_code": [],
        "same_name_changed_code_candidates": [],
        "unresolved_non_concrete": [
            {
                "course_code": code,
                "older": old_non_concrete.get(code, []),
                "newer": new_non_concrete.get(code, []),
            }
            for code in sorted(set(old_non_concrete) | set(new_non_concrete))
        ],
    }
    old_courses = {
        code: courses for code, courses in old_courses.items()
        if concrete_code.fullmatch(code) is not None
    }
    new_courses = {
        code: courses for code, courses in new_courses.items()
        if concrete_code.fullmatch(code) is not None
    }
    old_codes, new_codes = set(old_courses), set(new_courses)
    shared_codes = sorted(old_codes & new_codes)
    old_only_codes = sorted(old_codes - new_codes)
    new_only_codes = sorted(new_codes - old_codes)
    categories["shared_same_code"] = [
            {
                "course_code_normalized": code,
                "older": old_courses[code],
                "newer": new_courses[code],
            }
            for code in shared_codes
        ]
    categories["old_only_by_code"] = [
            {"course_code_normalized": code, "courses": old_courses[code]}
            for code in old_only_codes
        ]
    categories["new_only_by_code"] = [
            {"course_code_normalized": code, "courses": new_courses[code]}
            for code in new_only_codes
        ]
    for old_code in old_only_codes:
        for new_code in new_only_codes:
            for old_course in old_courses[old_code]:
                for new_course in new_courses[new_code]:
                    matched_fields = [
                        field
                        for field in ("name_th", "name_en")
                        if _normalized_course_name(old_course.get(field)) is not None
                        and _normalized_course_name(old_course.get(field))
                        == _normalized_course_name(new_course.get(field))
                    ]
                    if matched_fields:
                        categories["same_name_changed_code_candidates"].append(
                            {
                                "older": old_course,
                                "newer": new_course,
                                "matched_name_fields": matched_fields,
                                "equivalence_proven": False,
                            }
                        )
    categories["same_name_changed_code_candidates"].sort(
        key=lambda item: (
            item["older"]["course_code_normalized"],
            item["newer"]["course_code_normalized"],
        )
    )
    all_courses = [
        course
        for mapping in (old_courses, new_courses)
        for code in sorted(mapping)
        for course in mapping[code]
    ]
    provenance = _merge_provenance(*(course["provenance"] for course in all_courses))
    complete = old_complete and new_complete and bool(provenance)
    return {
        "status": "complete" if complete else "incomplete_evidence",
        "program": normalized_program,
        "older": {key: older[key] for key in ("catalog_key", "academic_year")},
        "newer": {key: newer[key] for key in ("catalog_key", "academic_year")},
        "categories": categories,
        "counts": {key: len(value) for key, value in categories.items()},
        "provenance": provenance,
        "limitations": [] if complete else [
            "one or more compared courses lack linked source provenance"
        ],
        "warnings": (
            ["non-concrete course-code placeholders are reported separately"]
            if categories["unresolved_non_concrete"] else []
        ),
    }
