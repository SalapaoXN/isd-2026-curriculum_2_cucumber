"""Deterministic structural checks for an explicitly named curriculum plan."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from rag.structured.queries import _merge_provenance, _provenance_for


_MANDATORY = "บังคับ"
_ELECTIVE = "เลือก"
_GRADUATION_CATEGORY = "เกณฑ์การสำเร็จการศึกษา"
_EDITION_SCOPED_PROGRAM_REQUIREMENTS = frozenset(
    {("DSBA", "dsba-2565", "2565")}
)


def _failure(status: str, program: Any, plan: Any, limitation: str) -> dict[str, Any]:
    return {
        "status": status,
        "assessment_scope": "curriculum_plan_structure_only",
        "program": program,
        "plan": plan,
        "checks": [],
        "mandatory_courses": {"status": "incomplete_evidence", "count": 0, "courses": []},
        "alternative_groups": [],
        "credit_requirements": [],
        "placement_quality": {},
        "unassessable_requirements": [],
        "provenance": [],
        "warnings": [limitation],
        "limitations": [limitation],
    }


def _resolve_plan(
    connection: sqlite3.Connection,
    program: str,
    plan_key: str,
    catalog_key: str | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    rows = connection.execute(
        """SELECT cp.plan_id, cp.catalog_id, cp.program_code, cp.plan_key,
                  cp.program_id, p.program_code_normalized, c.catalog_key
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


def _merge_evidence(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for group in groups:
        for item in group:
            if item.get("provenance_id") is not None:
                identity = ("provenance_id", int(item["provenance_id"]))
            else:
                identity = tuple(
                    (key, json.dumps(item.get(key), sort_keys=True, ensure_ascii=False))
                    for key in sorted(item)
                )
            if identity not in seen:
                seen.add(identity)
                merged.append(dict(item))
    return merged


def _course_evidence(
    connection: sqlite3.Connection,
    course_id: int,
    placement_ids: list[int],
    *,
    group_id: int | None = None,
    member_id: int | None = None,
) -> list[dict[str, Any]]:
    groups = [
        _provenance_for(connection, "course_provenance", "course_id", course_id)
    ]
    groups.extend(
        _provenance_for(
            connection, "plan_placement_provenance", "placement_id", placement_id
        )
        for placement_id in placement_ids
    )
    if group_id is not None:
        groups.append(
            _provenance_for(
                connection, "alternative_group_provenance", "alternative_group_id", group_id
            )
        )
    if member_id is not None:
        groups.append(
            _provenance_for(
                connection,
                "alternative_group_member_provenance",
                "alternative_group_member_id",
                member_id,
            )
        )
    return _merge_evidence(*groups)


def _placement_rows(
    connection: sqlite3.Connection, plan_id: int
) -> list[sqlite3.Row]:
    return connection.execute(
        """SELECT pl.placement_id, pl.course_id, pl.alternative_group_id,
                  pl.year_number, pl.semester_number,
                  pl.flexible_year_number, pl.flexible_semester_number,
                  pl.flexible_year_semester_raw, pl.category, pl.requirement_type,
                  pl.raw_text, c.catalog_id AS course_catalog_id,
                  c.course_code, c.course_code_normalized, c.name_th, c.name_en,
                  ag.catalog_id AS group_catalog_id, ag.plan_id AS group_plan_id
           FROM plan_placements AS pl
           LEFT JOIN courses AS c ON c.course_id = pl.course_id
           LEFT JOIN alternative_course_groups AS ag
             ON ag.alternative_group_id = pl.alternative_group_id
           WHERE pl.plan_id = ?
           ORDER BY pl.placement_id""",
        (plan_id,),
    ).fetchall()


def _unique_name(values: set[str]) -> str | None:
    return next(iter(values)) if len(values) == 1 else None


def _mandatory_courses(
    connection: sqlite3.Connection,
    plan: dict[str, Any],
    rows: list[sqlite3.Row],
) -> tuple[dict[str, Any], list[dict[str, Any]], bool]:
    identities: dict[str, dict[str, Any]] = {}
    labels_by_identity: dict[str, set[str]] = {}
    unresolved: list[dict[str, Any]] = []
    complete = True
    for row in rows:
        if row["alternative_group_id"] is not None:
            continue
        placement_id = int(row["placement_id"])
        code = row["course_code_normalized"]
        req_type = row["requirement_type"]
        references = _course_evidence(
            connection,
            int(row["course_id"]) if row["course_id"] is not None else -1,
            [placement_id],
        ) if row["course_id"] is not None else []
        if not isinstance(code, str) or not code.strip():
            complete = False
            unresolved.append({
                "placement_id": placement_id,
                "reason": "missing canonical normalized course identity",
                "requirement_type": req_type,
                "raw_text": row["raw_text"],
                "provenance": references,
            })
            continue
        code = str(code)
        if row["course_catalog_id"] is None or int(row["course_catalog_id"]) != plan["catalog_id"]:
            complete = False
            unresolved.append({
                "placement_id": placement_id,
                "course_code_normalized": code,
                "reason": "course is missing or outside the resolved catalog",
                "requirement_type": req_type,
                "raw_text": row["raw_text"],
                "provenance": references,
            })
            continue
        if not references:
            complete = False
            unresolved.append({
                "placement_id": placement_id,
                "course_code_normalized": code,
                "reason": "placement lacks linked source evidence",
                "requirement_type": req_type,
                "raw_text": row["raw_text"],
                "provenance": [],
            })
        if req_type not in (_MANDATORY, _ELECTIVE):
            complete = False
            unresolved.append({
                "placement_id": placement_id,
                "course_code_normalized": code,
                "reason": "placement requirement type is not recognized",
                "requirement_type": req_type,
                "raw_text": row["raw_text"],
                "provenance": references,
            })
            continue
        labels_by_identity.setdefault(code, set()).add(str(req_type))
        if req_type != _MANDATORY:
            continue
        item = identities.setdefault(
            code,
            {
                "course_code_normalized": code,
                "_codes": set(),
                "_names_th": set(),
                "_names_en": set(),
                "placement_ids": set(),
                "provenance": [],
            },
        )
        item["_codes"].add(str(row["course_code"]))
        if row["name_th"]:
            item["_names_th"].add(str(row["name_th"]))
        if row["name_en"]:
            item["_names_en"].add(str(row["name_en"]))
        item["placement_ids"].add(placement_id)
        item["provenance"] = _merge_evidence(item["provenance"], references)

    for code, labels in labels_by_identity.items():
        if _MANDATORY in labels and _ELECTIVE in labels:
            complete = False
            unresolved.append({
                "course_code_normalized": code,
                "reason": "conflicting mandatory and elective placement labels",
                "requirement_type": sorted(labels),
                "provenance": [
                    dict(reference)
                    for reference in identities.get(code, {}).get("provenance", [])
                ],
            })
            identities.pop(code, None)

    courses = []
    for code in sorted(identities):
        item = identities[code]
        codes = sorted(item.pop("_codes"))
        names_th = item.pop("_names_th")
        names_en = item.pop("_names_en")
        item["course_code"] = codes[0] if len(codes) == 1 else None
        item["name_th"] = _unique_name(names_th)
        item["name_en"] = _unique_name(names_en)
        if len(codes) > 1 or len(names_th) > 1 or len(names_en) > 1:
            item["identity_or_name_conflict"] = True
            complete = False
        item["placement_ids"] = sorted(item["placement_ids"])
        item["provenance"] = item["provenance"]
        courses.append(item)

    return {
        "status": "complete" if complete else "incomplete_evidence",
        "count": len(courses),
        "courses": courses,
        "unresolved_placements": unresolved,
    }, unresolved, complete


def _alternative_groups(
    connection: sqlite3.Connection,
    plan: dict[str, Any],
) -> tuple[list[dict[str, Any]], bool]:
    groups = connection.execute(
        """SELECT DISTINCT ag.alternative_group_id, ag.catalog_id, ag.plan_id,
                  ag.group_key, ag.label, ag.minimum_choices, ag.maximum_choices,
                  pl.placement_id, pl.requirement_type
           FROM plan_placements AS pl
           JOIN alternative_course_groups AS ag
             ON ag.alternative_group_id = pl.alternative_group_id
           WHERE pl.plan_id = ?
           ORDER BY ag.alternative_group_id, pl.placement_id""",
        (plan["plan_id"],),
    ).fetchall()
    grouped: dict[int, dict[str, Any]] = {}
    complete = True
    for row in groups:
        group_id = int(row["alternative_group_id"])
        group = grouped.setdefault(
            group_id,
            {
                "alternative_group_id": group_id,
                "group_key": row["group_key"],
                "label": row["label"],
                "minimum_choices": row["minimum_choices"],
                "maximum_choices": row["maximum_choices"],
                "placement_ids": set(),
                "_placement_types": set(),
                "_limitations": [],
                "provenance": _provenance_for(
                    connection,
                    "alternative_group_provenance",
                    "alternative_group_id",
                    group_id,
                ),
                "_candidates": {},
                "structurally_complete": True,
            },
        )
        group["placement_ids"].add(int(row["placement_id"]))
        group["_placement_types"].add(row["requirement_type"])
        group["provenance"] = _merge_evidence(
            group["provenance"],
            _provenance_for(
                connection,
                "plan_placement_provenance",
                "placement_id",
                int(row["placement_id"]),
            ),
        )
        if (
            int(row["catalog_id"]) != plan["catalog_id"]
            or (row["plan_id"] is not None and int(row["plan_id"]) != plan["plan_id"])
        ):
            group["structurally_complete"] = False
            group["_limitations"].append("alternative group scope conflicts with the resolved plan")
        if row["requirement_type"] != _MANDATORY:
            group["structurally_complete"] = False
            group["_limitations"].append(
                "the group placement is not explicitly represented as mandatory"
            )

    for group_id, group in grouped.items():
        members = connection.execute(
            """SELECT gm.alternative_group_member_id, c.course_id, c.catalog_id,
                      c.course_code, c.course_code_normalized, c.name_th, c.name_en
               FROM alternative_course_group_members AS gm
               LEFT JOIN courses AS c ON c.course_id = gm.course_id
               WHERE gm.alternative_group_id = ?
               ORDER BY gm.member_order, gm.alternative_group_member_id""",
            (group_id,),
        ).fetchall()
        candidates: dict[str, dict[str, Any]] = {}
        for member in members:
            code = member["course_code_normalized"]
            if not isinstance(code, str) or not code.strip() or member["course_id"] is None:
                group["structurally_complete"] = False
                group["_limitations"].append("a group member lacks canonical course identity")
                continue
            if int(member["catalog_id"]) != plan["catalog_id"]:
                group["structurally_complete"] = False
                group["_limitations"].append("a group member belongs to another catalog")
                continue
            member_id = int(member["alternative_group_member_id"])
            refs = _course_evidence(
                connection,
                int(member["course_id"]),
                sorted(group["placement_ids"]),
                group_id=group_id,
                member_id=member_id,
            )
            if not refs:
                group["structurally_complete"] = False
                group["_limitations"].append("a group candidate lacks linked source evidence")
            candidate = candidates.setdefault(
                str(code),
                {
                    "course_code_normalized": str(code),
                    "course_code": member["course_code"],
                    "name_th": member["name_th"],
                    "name_en": member["name_en"],
                    "member_ids": [],
                    "provenance": [],
                },
            )
            candidate["member_ids"].append(member_id)
            candidate["provenance"] = _merge_evidence(candidate["provenance"], refs)
        group["_candidates"] = candidates
        min_choices, max_choices = group["minimum_choices"], group["maximum_choices"]
        if not group["provenance"]:
            group["structurally_complete"] = False
            group["_limitations"].append("alternative group has no linked source evidence")
        if (
            not isinstance(min_choices, int)
            or not isinstance(max_choices, int)
            or min_choices < 1
            or max_choices < min_choices
            or max_choices > len(candidates)
        ):
            group["structurally_complete"] = False
            group["_limitations"].append(
                "choice bounds are invalid or exceed the logical candidate count"
            )
        if not group["structurally_complete"]:
            complete = False

    result = []
    for group_id in sorted(grouped):
        group = grouped[group_id]
        candidates = [group["_candidates"][key] for key in sorted(group["_candidates"])]
        group["placement_ids"] = sorted(group["placement_ids"])
        group["candidates"] = candidates
        group["status"] = (
            "complete" if group["structurally_complete"] else "incomplete_evidence"
        )
        group["limitations"] = list(dict.fromkeys(group.pop("_limitations")))
        group.pop("_placement_types")
        group.pop("_candidates")
        result.append(group)
    broken = connection.execute(
        """SELECT pl.placement_id, pl.alternative_group_id
           FROM plan_placements AS pl
           LEFT JOIN alternative_course_groups AS ag
             ON ag.alternative_group_id = pl.alternative_group_id
           WHERE pl.plan_id = ? AND pl.alternative_group_id IS NOT NULL
             AND ag.alternative_group_id IS NULL
           ORDER BY pl.placement_id""",
        (plan["plan_id"],),
    ).fetchall()
    broken_by_id: dict[int, list[int]] = {}
    for row in broken:
        broken_by_id.setdefault(int(row["alternative_group_id"]), []).append(
            int(row["placement_id"])
        )
    for group_id, placement_ids in sorted(broken_by_id.items()):
        complete = False
        result.append({
            "alternative_group_id": group_id,
            "group_key": None,
            "label": None,
            "minimum_choices": None,
            "maximum_choices": None,
            "placement_ids": placement_ids,
            "candidates": [],
            "structurally_complete": False,
            "status": "incomplete_evidence",
            "limitations": ["alternative group record is missing from the canonical database"],
            "provenance": _merge_evidence(*[
                _provenance_for(
                    connection,
                    "plan_placement_provenance",
                    "placement_id",
                    placement_id,
                )
                for placement_id in placement_ids
            ]),
        })
    return result, complete


def _placement_quality(
    connection: sqlite3.Connection,
    rows: list[sqlite3.Row],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    fixed = flexible = untimed = unresolved_raw = 0
    fixed_ids: list[int] = []
    flexible_ids: list[int] = []
    untimed_ids: list[int] = []
    placement_evidence: list[dict[str, Any]] = []
    direct_code_placements: dict[str, list[int]] = {}
    unresolved: list[dict[str, Any]] = []
    for row in rows:
        placement_id = int(row["placement_id"])
        year, semester = row["year_number"], row["semester_number"]
        flex_year, flex_semester = row["flexible_year_number"], row["flexible_semester_number"]
        raw_flex = row["flexible_year_semester_raw"]
        fixed_any = year is not None or semester is not None
        flex_any = flex_year is not None or flex_semester is not None
        fixed_valid = (
            isinstance(year, int) and 1 <= year <= 5
            and isinstance(semester, int) and 1 <= semester <= 2
        )
        flex_valid = (
            isinstance(flex_year, int) and 1 <= flex_year <= 5
            and isinstance(flex_semester, int) and 1 <= flex_semester <= 2
        )
        if fixed_valid and not flex_any and not raw_flex:
            fixed += 1
            fixed_ids.append(placement_id)
        elif flex_valid and not fixed_any:
            flexible += 1
            flexible_ids.append(placement_id)
        elif not fixed_any and not flex_any and not (isinstance(raw_flex, str) and raw_flex.strip()):
            untimed += 1
            untimed_ids.append(placement_id)
        else:
            unresolved_raw += 1
            unresolved.append({
                "placement_id": placement_id,
                "year_number": year,
                "semester_number": semester,
                "flexible_year_number": flex_year,
                "flexible_semester_number": flex_semester,
                "flexible_year_semester_raw": raw_flex,
                "raw_text": row["raw_text"],
                "provenance": _provenance_for(
                    connection, "plan_placement_provenance", "placement_id", placement_id
                ),
            })
        placement_evidence = _merge_evidence(
            placement_evidence,
            _provenance_for(
                connection, "plan_placement_provenance", "placement_id", placement_id
            ),
        )
        if row["alternative_group_id"] is None and row["course_code_normalized"]:
            code = str(row["course_code_normalized"])
            direct_code_placements.setdefault(code, []).append(placement_id)
    duplicates = {
        code: {"count": len(ids), "placement_ids": sorted(ids)}
        for code, ids in sorted(direct_code_placements.items())
        if len(ids) > 1
    }
    return {
        "fixed_count": fixed,
        "flexible_count": flexible,
        "untimed_count": untimed,
        "unresolved_raw_count": unresolved_raw,
        "fixed_placement_ids": fixed_ids,
        "flexible_placement_ids": flexible_ids,
        "untimed_placement_ids": untimed_ids,
        "placement_provenance": placement_evidence,
        "duplicate_logical_placements": duplicates,
        "unresolved_placements": unresolved,
    }, unresolved


def _program_requirements(
    connection: sqlite3.Connection,
    program: str,
    catalog_key: str | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    catalog_rows = connection.execute(
        """SELECT DISTINCT c.catalog_id, c.catalog_key, c.academic_year
           FROM catalogs c
           JOIN programs p ON p.catalog_id=c.catalog_id
           WHERE UPPER(p.program_code_normalized)=?
           ORDER BY c.catalog_id""",
        (program.strip().casefold().upper(),),
    ).fetchall()
    if catalog_key is not None:
        matching_catalogs = [
            row for row in catalog_rows
            if isinstance(row["catalog_key"], str)
            and row["catalog_key"].strip().casefold() == catalog_key.strip().casefold()
        ]
        if len(matching_catalogs) != 1:
            return [], False
        selected = matching_catalogs[0]
    else:
        if len(catalog_rows) > 1:
            return [], False
        selected = catalog_rows[0] if catalog_rows else None

    if len(catalog_rows) > 1:
        identity = (
            program.strip().upper(),
            str(selected["catalog_key"]).strip() if selected is not None else "",
            str(selected["academic_year"]).strip() if selected is not None else "",
        )
        if identity not in _EDITION_SCOPED_PROGRAM_REQUIREMENTS:
            return [], False

    rows = connection.execute(
        """SELECT requirement_id, program_code, requirement_type, operator, value, unit
           FROM program_requirements
           WHERE UPPER(program_code) = ?
           ORDER BY requirement_type, requirement_id""",
        (program.upper(),),
    ).fetchall()
    result = []
    complete = True
    for row in rows:
        requirement_id = int(row["requirement_id"])
        references = _provenance_for(
            connection,
            "program_requirement_provenance",
            "requirement_id",
            requirement_id,
        )
        if not references or any(
            reference.get("document_category") != "program_requirement"
            for reference in references
        ):
            complete = False
        result.append({
            "requirement_id": requirement_id,
            "requirement_type": row["requirement_type"],
            "operator": row["operator"],
            "required_value": row["value"],
            "unit": row["unit"],
            "measured_value": None,
            "status": "incomplete_evidence",
            "limitation": (
                "A required total cannot be measured safely: elective candidate rows and "
                "alternative choices are not all required credits, and category-specific "
                "minimum-credit semantics are not comprehensively represented. No rows were summed."
            ),
            "provenance": references,
        })
    return result, complete


def _rule_references(references_json: Any) -> list[dict[str, Any]]:
    if not isinstance(references_json, str):
        return []
    try:
        payload = json.loads(references_json)
    except (TypeError, json.JSONDecodeError):
        return []
    if not isinstance(payload, dict) or not isinstance(payload.get("source_provenance"), list):
        return []
    refs = []
    for item in payload["source_provenance"]:
        if not isinstance(item, dict):
            continue
        refs.append({
            key: item[key]
            for key in (
                "program", "source_filename", "source_page", "document_page",
                "document_category", "source_locator",
            )
            if key in item
        })
    return refs


def _unassessable_requirements(
    connection: sqlite3.Connection,
) -> tuple[list[dict[str, Any]], bool]:
    result: list[dict[str, Any]] = [
        {
            "kind": "validator_limitation",
            "requirement_type": "student_completed_courses",
            "status": "unassessable",
            "reason": "No student transcript or completion record is supplied to this plan-only operation.",
            "provenance": [],
        },
        {
            "kind": "validator_limitation",
            "requirement_type": "actual_enrollment_history",
            "status": "unassessable",
            "reason": "No student enrollment history is supplied to this plan-only operation.",
            "provenance": [],
        },
        {
            "kind": "validator_limitation",
            "requirement_type": "category_specific_minimum_credits",
            "status": "unassessable",
            "reason": "The database does not comprehensively encode category-specific minimum-credit requirements.",
            "provenance": [],
        },
    ]
    complete = True
    facts = connection.execute(
        """SELECT fact_id, category, fact_key, condition, value, unit, context,
                  source_rule_id
           FROM policy_facts
           WHERE category = ?
           ORDER BY fact_id""",
        (_GRADUATION_CATEGORY,),
    ).fetchall()
    fact_evidence: list[tuple[str, list[dict[str, Any]]]] = []
    for fact in facts:
        fact_id = int(fact["fact_id"])
        references = _provenance_for(
            connection, "policy_fact_provenance", "fact_id", fact_id
        )
        if not references or any(
            reference.get("document_category") != "rule" for reference in references
        ):
            complete = False
        rule = connection.execute(
            "SELECT section_number, category, references_json FROM regulation_rules WHERE rule_id = ?",
            (fact["source_rule_id"],),
        ).fetchone()
        rule_refs = _rule_references(rule["references_json"]) if rule else []
        if not rule or not rule_refs:
            complete = False
        combined_references = _merge_evidence(references, rule_refs)
        fact_evidence.append((str(fact["fact_key"] or ""), combined_references))
        result.append({
            "kind": "policy_fact",
            "fact_id": fact_id,
            "requirement_type": fact["fact_key"],
            "fact_key": fact["fact_key"],
            "operator": fact["condition"],
            "required_value": fact["value"],
            "unit": fact["unit"],
            "context": fact["context"],
            "source_rule_id": fact["source_rule_id"],
            "rule_section_number": rule["section_number"] if rule else None,
            "status": "unassessable",
            "reason": "This canonical condition requires student-specific records, not just a curriculum plan.",
            "provenance": combined_references,
        })

    gap_categories = (
        ("student_completed_courses", lambda key: "สำเร็จโครงสร้างหลักสูตร" in key),
        ("student_grades_or_gpa", lambda key: "GPA" in key.upper()),
        ("english_exit_exam_result", lambda key: "English Exit Exam" in key),
        ("debt_status", lambda key: "หนี้" in key),
    )
    for requirement_type, matches in gap_categories:
        matching = [references for key, references in fact_evidence if matches(key)]
        result.append({
            "kind": "student_assessment_gap",
            "requirement_type": requirement_type,
            "status": "unassessable",
            "reason": "No student-specific record is supplied to this curriculum-plan operation.",
            "provenance": _merge_evidence(*matching),
        })

    rules = connection.execute(
        """SELECT rule_id, section_number, category, rule_text, references_json
           FROM regulation_rules
           WHERE category = ?
           ORDER BY section_number, rule_id""",
        (_GRADUATION_CATEGORY,),
    ).fetchall()
    for rule in rules:
        references = _rule_references(rule["references_json"])
        if not references:
            complete = False
        result.append({
            "kind": "regulation_rule",
            "rule_id": rule["rule_id"],
            "section_number": rule["section_number"],
            "requirement_type": "graduation_condition",
            "rule_text": rule["rule_text"],
            "status": "unassessable",
            "reason": "A policy rule cannot be evaluated for an individual without student-specific records.",
            "provenance": references,
        })
    return result, complete


def validate_curriculum_plan_structure(
    db_path: str | Path,
    program: str,
    plan: str,
    catalog_key: str | None = None,
) -> dict[str, Any]:
    """Report structural checks supported by canonical evidence for one explicit plan."""
    if (
        not isinstance(program, str) or not program.strip()
        or not isinstance(plan, str) or not plan.strip()
    ):
        return _failure(
            "invalid_scope", program, plan,
            "program and plan must be explicitly provided",
        )
    if catalog_key is not None and (
        not isinstance(catalog_key, str) or not catalog_key.strip()
    ):
        return _failure(
            "invalid_scope", program, plan,
            "catalog_key must be a non-empty string when provided",
        )
    catalog_key = catalog_key.strip() if catalog_key is not None else None
    normalized_program = program.strip().upper()
    normalized_plan = plan.strip()
    try:
        uri = f"{Path(db_path).resolve().as_uri()}?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as connection:
            connection.row_factory = sqlite3.Row
            resolved, scope_error = _resolve_plan(
                connection, normalized_program, normalized_plan, catalog_key
            )
            if scope_error:
                return _failure(
                    scope_error, normalized_program, normalized_plan,
                    "the explicit program and plan must resolve to exactly one canonical plan",
                )
            plan_refs = _provenance_for(
                connection,
                "curriculum_plan_provenance",
                "plan_id",
                resolved["plan_id"],
            )
            if not plan_refs:
                return _failure(
                    "incomplete_evidence", resolved["program"], resolved["plan"],
                    "the resolved plan has no linked source provenance",
                )

            placements = _placement_rows(connection, resolved["plan_id"])
            mandatory, unresolved_mandatory, _mandatory_complete = _mandatory_courses(
                connection, resolved, placements
            )
            groups, groups_complete = _alternative_groups(connection, resolved)
            quality, unresolved_timing = _placement_quality(connection, placements)
            requirements, requirements_complete = _program_requirements(
                connection, resolved["program"], resolved["catalog_key"]
            )
            unassessable, policy_complete = _unassessable_requirements(connection)

            elective_evidence = []
            elective_count = 0
            for row in placements:
                if row["alternative_group_id"] is not None or row["requirement_type"] != _ELECTIVE:
                    continue
                elective_count += 1
                if row["course_id"] is not None:
                    elective_evidence = _merge_evidence(
                        elective_evidence,
                        _course_evidence(
                            connection,
                            int(row["course_id"]),
                            [int(row["placement_id"])],
                        ),
                    )
            if elective_count:
                unassessable.append({
                    "kind": "plan_semantics",
                    "requirement_type": "elective_selection_semantics",
                    "status": "unassessable",
                    "candidate_placement_count": elective_count,
                    "reason": "Elective candidate placements do not specify how many courses or credits are required.",
                    "provenance": elective_evidence,
                })

            credit_check_status = "incomplete_evidence"
            checks = [
                {
                    "check": "mandatory_placements",
                    "status": mandatory["status"],
                    "course_count": mandatory["count"],
                    "unresolved_count": len(unresolved_mandatory),
                },
                {
                    "check": "alternative_groups",
                    "status": "complete" if groups_complete else "incomplete_evidence",
                    "group_count": len(groups),
                },
                {
                    "check": "program_credit_requirements",
                    "status": credit_check_status,
                    "requirement_count": len(requirements),
                },
                {
                    "check": "placement_quality",
                    "status": "incomplete_evidence" if unresolved_timing else "complete",
                    "placement_count": len(placements),
                },
                {
                    "check": "student_specific_requirements",
                    "status": "unassessable",
                    "requirement_count": len(unassessable),
                },
            ]
            warnings = []
            if elective_count:
                warnings.append(
                    "Elective candidates were excluded from mandatory course and required-credit totals."
                )
            if requirements:
                warnings.append(
                    "Program total-credit requirement is recorded, but required credits cannot be safely measured from this plan representation."
                )
            if not requirements:
                warnings.append(
                    "No applicable program credit requirement is represented for this program."
                )
            if unresolved_mandatory:
                warnings.append("Some direct placements have unresolved requirement semantics or evidence.")
            if unresolved_timing:
                warnings.append("Some placements have incomplete or unparsed term fields.")
            if not requirements_complete or not policy_complete:
                warnings.append("Some applicable requirement evidence lacks complete provenance.")
            if any(
                reference.get("document_page") is None
                for group in (
                    plan_refs,
                    quality["placement_provenance"],
                    *(item["provenance"] for item in requirements),
                    *(item["provenance"] for item in unassessable),
                )
                for reference in group
                if "provenance_id" in reference
            ):
                warnings.append("document_page is absent from some stored provenance records.")

            evidence_groups = [plan_refs]
            evidence_groups.extend(course["provenance"] for course in mandatory["courses"])
            evidence_groups.extend(group["provenance"] for group in groups)
            evidence_groups.extend(
                candidate["provenance"]
                for group in groups
                for candidate in group["candidates"]
            )
            evidence_groups.extend(req["provenance"] for req in requirements)
            evidence_groups.extend(item["provenance"] for item in unassessable)
            evidence_groups.extend(item["provenance"] for item in unresolved_mandatory)
            evidence_groups.extend(item["provenance"] for item in unresolved_timing)
            evidence_groups.append(quality["placement_provenance"])
            provenance = _merge_evidence(*evidence_groups)
            if any(
                reference.get("source_filename") is None
                or reference.get("source_page") is None
                for reference in provenance
            ):
                warnings.append("Some linked evidence lacks a source filename or source page.")

            return {
                "status": "incomplete_evidence",
                "assessment_scope": "curriculum_plan_structure_only",
                "program": resolved["program"],
                "plan": resolved["plan"],
                "catalog_key": resolved["catalog_key"],
                "checks": checks,
                "mandatory_courses": mandatory,
                "alternative_groups": groups,
                "credit_requirements": requirements,
                "placement_quality": quality,
                "unassessable_requirements": unassessable,
                "provenance": provenance,
                "warnings": warnings,
                "limitations": list(warnings),
            }
    except (OSError, sqlite3.Error, ValueError, TypeError) as error:
        return _failure(
            "database_error", normalized_program, normalized_plan,
            f"read-only curriculum evidence is unavailable ({type(error).__name__})",
        )
