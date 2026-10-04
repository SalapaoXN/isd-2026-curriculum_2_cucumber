"""Deterministic prerequisite-order validation for a named curriculum plan."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from rag.structured.queries import _merge_provenance, _provenance_for


def _failure(status: str, program: Any, plan: Any, reason: str) -> dict[str, Any]:
    return {
        "status": status,
        "program": program,
        "plan": plan,
        "relationships": [],
        "transitive_paths": [],
        "violations": [],
        "incomplete": [],
        "summary": {"satisfied": 0, "violations": 0, "incomplete": 0},
        "provenance": [],
        "limitations": [reason],
    }


def _connect_read_only(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _resolve_plan(
    connection: sqlite3.Connection,
    program: str,
    plan: str,
    catalog_key: str | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    rows = connection.execute(
        """SELECT cp.plan_id, cp.catalog_id, cp.program_id, cp.program_code,
                  cp.plan_key, c.catalog_key
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
            program.strip().upper(), program.strip().upper(), plan.strip().casefold(),
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


def _course_evidence(
    connection: sqlite3.Connection,
    course_id: int,
    placement_rows: list[sqlite3.Row],
    *,
    group_id: int | None = None,
    member_id: int | None = None,
) -> list[dict[str, Any]]:
    refs = [_provenance_for(connection, "course_provenance", "course_id", course_id)]
    refs.extend(
        _provenance_for(
            connection, "plan_placement_provenance", "placement_id", int(row["placement_id"])
        )
        for row in placement_rows
    )
    if group_id is not None:
        refs.append(_provenance_for(connection, "alternative_group_provenance", "alternative_group_id", group_id))
    if member_id is not None:
        refs.append(_provenance_for(connection, "alternative_group_member_provenance", "alternative_group_member_id", member_id))
    return _merge_provenance(*refs)


def _placement_ordinal(row: sqlite3.Row) -> tuple[int | None, str | None]:
    year, semester = row["year_number"], row["semester_number"]
    flex_year = row["flexible_year_number"]
    flex_semester = row["flexible_semester_number"]
    flex_raw = row["flexible_year_semester_raw"]
    any_flexible = flex_year is not None or flex_semester is not None or bool(
        isinstance(flex_raw, str) and flex_raw.strip()
    )
    if any_flexible:
        return None, "flexible_or_conflicting_placement"
    if year is None and semester is None:
        return None, "untimed_placement"
    if (
        isinstance(year, int) and not isinstance(year, bool) and 1 <= year <= 5
        and isinstance(semester, int) and not isinstance(semester, bool) and 1 <= semester <= 2
    ):
        return (year - 1) * 2 + semester - 1, None
    return None, "incomplete_term_placement"


def _load_plan_courses(
    connection: sqlite3.Connection, scope: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    rows = connection.execute(
        """SELECT pl.placement_id, pl.course_id, pl.alternative_group_id,
                  pl.year_number, pl.semester_number, pl.flexible_year_number,
                  pl.flexible_semester_number, pl.flexible_year_semester_raw,
                  c.catalog_id, c.course_code, c.course_code_normalized,
                  c.name_th, c.name_en
           FROM plan_placements AS pl
           LEFT JOIN courses AS c ON c.course_id = pl.course_id
           WHERE pl.plan_id = ?
           ORDER BY pl.placement_id""",
        (scope["plan_id"],),
    ).fetchall()
    courses: dict[str, dict[str, Any]] = {}
    limitations: list[str] = []
    for placement in rows:
        group_members: list[sqlite3.Row | dict[str, Any]]
        if placement["alternative_group_id"] is not None:
            group_id = int(placement["alternative_group_id"])
            group = connection.execute(
                "SELECT catalog_id, plan_id FROM alternative_course_groups WHERE alternative_group_id = ?",
                (group_id,),
            ).fetchone()
            if (
                group is None or int(group["catalog_id"]) != scope["catalog_id"]
                or (group["plan_id"] is not None and int(group["plan_id"]) != scope["plan_id"])
            ):
                limitations.append(f"alternative plan placement {placement['placement_id']} has unresolved scope")
                continue
            group_members = connection.execute(
                """SELECT gm.alternative_group_member_id AS member_id,
                          c.course_id, c.catalog_id, c.course_code,
                          c.course_code_normalized, c.name_th, c.name_en
                   FROM alternative_course_group_members gm
                   JOIN courses c ON c.course_id = gm.course_id
                   WHERE gm.alternative_group_id = ?
                   ORDER BY gm.member_order, gm.alternative_group_member_id""",
                (group_id,),
            ).fetchall()
            if not group_members:
                limitations.append(f"alternative plan placement {placement['placement_id']} has no canonical members")
                continue
        elif placement["course_id"] is not None and placement["catalog_id"] is not None:
            group_members = [{
                "course_id": placement["course_id"],
                "catalog_id": placement["catalog_id"],
                "course_code": placement["course_code"],
                "course_code_normalized": placement["course_code_normalized"],
                "name_th": placement["name_th"],
                "name_en": placement["name_en"],
            }]
        else:
            limitations.append(f"plan placement {placement['placement_id']} has no canonical course")
            continue

        ordinal, term_issue = _placement_ordinal(placement)
        for member in group_members:
            code = member["course_code_normalized"]
            if (
                not isinstance(code, str) or not code.strip()
                or int(member["catalog_id"]) != scope["catalog_id"]
            ):
                limitations.append(f"placement {placement['placement_id']} has an invalid course identity or catalog")
                continue
            code = code.strip().upper()
            item = courses.setdefault(code, {
                "course_code": str(member["course_code"]),
                "course_code_normalized": code,
                "name_th": member["name_th"],
                "name_en": member["name_en"],
                "course_ids": set(),
                "placements": [],
                "_ordinals": set(),
                "_term_issues": [],
                "provenance": [],
                "evidence_complete": True,
            })
            course_id = int(member["course_id"])
            if item["course_ids"] and course_id not in item["course_ids"]:
                item["_term_issues"].append("duplicate logical identity maps to multiple canonical courses")
            item["course_ids"].add(course_id)
            item["placements"].append({
                "placement_id": int(placement["placement_id"]),
                "ordinal": ordinal,
                "year": placement["year_number"],
                "semester": placement["semester_number"],
                "issue": term_issue,
            })
            if ordinal is not None:
                item["_ordinals"].add(ordinal)
            if term_issue:
                item["_term_issues"].append(term_issue)
            refs = _course_evidence(
                connection,
                course_id,
                [placement],
                group_id=int(placement["alternative_group_id"]) if placement["alternative_group_id"] is not None else None,
                member_id=int(member["member_id"]) if isinstance(member, sqlite3.Row) and "member_id" in member.keys() else None,
            )
            item["provenance"] = _merge_provenance(item["provenance"], refs)
            course_refs = _provenance_for(connection, "course_provenance", "course_id", course_id)
            placement_refs = _provenance_for(
                connection, "plan_placement_provenance", "placement_id", int(placement["placement_id"])
            )
            if not course_refs or not placement_refs:
                item["evidence_complete"] = False
            if placement["alternative_group_id"] is not None:
                group_refs = _provenance_for(
                    connection, "alternative_group_provenance", "alternative_group_id", int(placement["alternative_group_id"])
                )
                member_refs = _provenance_for(
                    connection, "alternative_group_member_provenance", "alternative_group_member_id", int(member["member_id"])
                ) if isinstance(member, sqlite3.Row) and "member_id" in member.keys() else []
                if not group_refs or not member_refs:
                    item["evidence_complete"] = False

    for item in courses.values():
        if len(item["course_ids"]) != 1:
            item["term_issue"] = "conflicting_logical_course_identity"
            item["term_ordinal"] = None
        elif len(item["_ordinals"]) == 1 and not item["_term_issues"]:
            item["term_ordinal"] = next(iter(item["_ordinals"]))
            item["term_issue"] = None
        else:
            item["term_ordinal"] = None
            item["term_issue"] = (
                "conflicting_course_placements" if len(item["_ordinals"]) > 1
                else (item["_term_issues"][0] if item["_term_issues"] else "untimed_placement")
            )
        item["placements"].sort(key=lambda placement: placement["placement_id"])
        item.pop("_ordinals")
        item.pop("_term_issues")
    return courses, limitations


def _term_view(course: dict[str, Any] | None) -> dict[str, Any] | None:
    if course is None:
        return None
    ordinal = course.get("term_ordinal")
    if "candidate_term_index" in course:
        return {
            "ordinal": ordinal,
            "term_index": course["candidate_term_index"],
            "year": None,
            "semester": None,
            "basis": "candidate_sequence_input",
            "issue": course.get("term_issue"),
            "placements": [],
        }
    return {
        "ordinal": ordinal,
        "year": (ordinal // 2) + 1 if ordinal is not None else None,
        "semester": (ordinal % 2) + 1 if ordinal is not None else None,
        "issue": course.get("term_issue"),
        "placements": [dict(item) for item in course.get("placements", [])],
    }


def _relationship_status(
    requirement_type: str,
    dependent: dict[str, Any] | None,
    prerequisite_candidates: list[dict[str, Any]],
    minimum_choices: int,
) -> tuple[str, str | None]:
    normalized_type = requirement_type.casefold()
    allows_same_term = normalized_type in {"co_requisite", "corequisite"}
    if normalized_type not in {"required", "prerequisite", "co_requisite", "corequisite"}:
        return "incomplete_evidence", "unrecognized prerequisite requirement semantics"
    if dependent is None or dependent.get("term_ordinal") is None:
        return "incomplete_evidence", "dependent course placement is not determinately timed"
    if not dependent.get("provenance"):
        return "incomplete_evidence", "dependent course or placement lacks provenance"
    if not dependent.get("evidence_complete", False):
        return "incomplete_evidence", "dependent course placement lacks linked source evidence"
    if not prerequisite_candidates:
        return "incomplete_evidence", "prerequisite candidates are unresolved"
    earlier = 0
    for candidate in prerequisite_candidates:
        if candidate.get("term_ordinal") is None:
            continue
        if not candidate.get("provenance"):
            continue
        if candidate["term_ordinal"] < dependent["term_ordinal"] or (
            allows_same_term and candidate["term_ordinal"] == dependent["term_ordinal"]
        ):
            earlier += 1
    if earlier >= minimum_choices:
        return "satisfied", None
    if any(
        candidate.get("term_ordinal") is None or not candidate.get("provenance")
        for candidate in prerequisite_candidates
    ):
        return "incomplete_evidence", "one or more prerequisite placements or sources are unresolved"
    if len(prerequisite_candidates) < minimum_choices:
        return "incomplete_evidence", "choice minimum exceeds resolved prerequisite candidates"
    return "violation", (
        "co-requisite course is placed after the dependent course"
        if allows_same_term else "required prerequisite ordering is not strictly earlier"
    )


def _load_relationships(
    connection: sqlite3.Connection,
    scope: dict[str, Any],
    courses: dict[str, dict[str, Any]],
    *,
    candidate_terms: dict[str, int] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    if candidate_terms is not None:
        for code, item in courses.items():
            item["term_ordinal"] = candidate_terms.get(code)
            item["candidate_term_index"] = candidate_terms.get(code)
            item["term_issue"] = None if code in candidate_terms else "absent_from_candidate_sequence"
            item["placements"] = []
    rows = connection.execute(
        """SELECT pr.prerequisite_id, pr.course_id, pr.prerequisite_course_id,
                  pr.alternative_group_id, pr.prerequisite_order,
                  pr.requirement_type, pr.raw_text,
                  dependent.course_code_normalized AS dependent_code,
                  dependent.catalog_id AS dependent_catalog
           FROM prerequisites pr
           JOIN courses dependent ON dependent.course_id = pr.course_id
           WHERE dependent.catalog_id = ?
           ORDER BY dependent.course_code_normalized, pr.prerequisite_order,
                    pr.prerequisite_id""",
        (scope["catalog_id"],),
    ).fetchall()
    relationships: list[dict[str, Any]] = []
    limitations: list[str] = []
    plan_refs = _provenance_for(connection, "curriculum_plan_provenance", "plan_id", scope["plan_id"])
    for row in rows:
        dependent_code = str(row["dependent_code"]).strip().upper()
        if dependent_code not in courses:
            continue
        if candidate_terms is not None and dependent_code not in candidate_terms:
            continue
        prerequisite_id = int(row["prerequisite_id"])
        dependent = courses[dependent_code]
        prereq_edge_refs = _provenance_for(connection, "prerequisite_provenance", "prerequisite_id", prerequisite_id)
        prereq_refs = list(prereq_edge_refs)
        condition: dict[str, Any]
        candidates: list[dict[str, Any]] = []
        minimum = 1
        group_id = row["alternative_group_id"]
        if row["prerequisite_course_id"] is not None:
            prereq_course = connection.execute(
                """SELECT course_id, catalog_id, course_code, course_code_normalized,
                          name_th, name_en FROM courses WHERE course_id = ?""",
                (row["prerequisite_course_id"],),
            ).fetchone()
            code = str(prereq_course["course_code_normalized"]).strip().upper() if prereq_course else ""
            candidate = courses.get(code) if prereq_course and int(prereq_course["catalog_id"]) == scope["catalog_id"] else None
            if candidate is not None:
                candidates.append(candidate)
            condition = {"kind": "direct", "course_code": code or None}
            if prereq_course:
                prereq_course_refs = _provenance_for(
                    connection, "course_provenance", "course_id", int(prereq_course["course_id"])
                )
                prereq_refs = _merge_provenance(
                    prereq_refs,
                    prereq_course_refs,
                )
            else:
                prereq_course_refs = []
        elif group_id is not None:
            group = connection.execute(
                """SELECT alternative_group_id, catalog_id, plan_id, group_key,
                          label, minimum_choices, maximum_choices
                   FROM alternative_course_groups WHERE alternative_group_id = ?""",
                (group_id,),
            ).fetchone()
            if group is None or int(group["catalog_id"]) != scope["catalog_id"] or (
                group["plan_id"] is not None and int(group["plan_id"]) != scope["plan_id"]
            ):
                group = None
                limitations.append(f"prerequisite {prerequisite_id} alternative group scope is unresolved")
            members = connection.execute(
                """SELECT gm.alternative_group_member_id AS member_id,
                          c.course_id, c.catalog_id, c.course_code,
                          c.course_code_normalized, c.name_th, c.name_en
                   FROM alternative_course_group_members gm
                   JOIN courses c ON c.course_id = gm.course_id
                   WHERE gm.alternative_group_id = ?
                   ORDER BY gm.member_order, gm.alternative_group_member_id""",
                (group_id,),
            ).fetchall()
            group_evidence_complete = bool(
                _provenance_for(connection, "alternative_group_provenance", "alternative_group_id", int(group_id))
            )
            minimum = int(group["minimum_choices"]) if group is not None else 1
            maximum = int(group["maximum_choices"]) if group is not None else 0
            for member in members:
                code = str(member["course_code_normalized"]).strip().upper()
                candidate = courses.get(code) if int(member["catalog_id"]) == scope["catalog_id"] else None
                if candidate is not None:
                    candidates.append(candidate)
                prereq_refs = _merge_provenance(
                    prereq_refs,
                    _provenance_for(connection, "course_provenance", "course_id", int(member["course_id"])),
                    _provenance_for(connection, "alternative_group_member_provenance", "alternative_group_member_id", int(member["member_id"])),
                )
                if not _provenance_for(
                    connection,
                    "alternative_group_member_provenance",
                    "alternative_group_member_id",
                    int(member["member_id"]),
                ):
                    group_evidence_complete = False
            prereq_refs = _merge_provenance(
                prereq_refs,
                _provenance_for(connection, "alternative_group_provenance", "alternative_group_id", int(group_id)),
            )
            condition = {
                "kind": "alternative_group",
                "alternative_group_id": int(group_id),
                "group_key": group["group_key"] if group else None,
                "label": group["label"] if group else None,
                "minimum_choices": minimum,
                "candidate_course_codes": [str(member["course_code_normalized"]) for member in members],
            }
            if group is None or len(candidates) != len(members) or not members:
                limitations.append(f"prerequisite {prerequisite_id} has unresolved alternative members")
                group_evidence_complete = False
            if (
                group is None
                or minimum < 1
                or maximum < minimum
                or maximum > len(members)
            ):
                limitations.append(f"prerequisite {prerequisite_id} has invalid alternative choice bounds")
                group_evidence_complete = False
        else:
            condition = {"kind": "raw_only", "raw_text": row["raw_text"]}
            rel = {
                "prerequisite_id": prerequisite_id,
                "dependent_course_code": dependent["course_code"],
                "course_term": _term_view(dependent),
                "prerequisite_condition": condition,
                "prerequisite_terms": [],
                "status": "incomplete_evidence",
                "reason": "raw prerequisite text has no resolved structured edge or group",
                "paths": [[None, dependent["course_code"]]],
                "provenance": _merge_provenance(
                    prereq_refs,
                    dependent["provenance"],
                    plan_refs,
                ),
            }
            relationships.append(rel)
            continue

        status, reason = _relationship_status(
            str(row["requirement_type"]), dependent, candidates, minimum
        )
        if not prereq_edge_refs or not plan_refs or (row["prerequisite_course_id"] is not None and not prereq_course_refs):
            status, reason = "incomplete_evidence", "prerequisite relationship lacks linked source evidence"
        if group_id is not None and not group_evidence_complete:
            status, reason = "incomplete_evidence", "alternative prerequisite group or member lacks linked source evidence"
        if any(not candidate.get("evidence_complete", False) for candidate in candidates):
            status, reason = "incomplete_evidence", "prerequisite course placement lacks linked source evidence"
        if not dependent["provenance"]:
            status, reason = "incomplete_evidence", "dependent placement lacks linked source evidence"
        candidate_term_rows = [
            {
                "course_code": candidate["course_code"],
                "term": _term_view(candidate),
                "before_dependent": (
                    candidate.get("term_ordinal") is not None
                    and dependent.get("term_ordinal") is not None
                    and candidate["term_ordinal"] < dependent["term_ordinal"]
                ),
                "satisfies_order": (
                    candidate.get("term_ordinal") is not None
                    and dependent.get("term_ordinal") is not None
                    and (
                        candidate["term_ordinal"] < dependent["term_ordinal"]
                        or (
                            str(row["requirement_type"]).casefold() in {"co_requisite", "corequisite"}
                            and candidate["term_ordinal"] == dependent["term_ordinal"]
                        )
                    )
                ),
                "provenance": list(candidate["provenance"]),
            }
            for candidate in candidates
        ]
        rel = {
            "prerequisite_id": prerequisite_id,
            "dependent_course_code": dependent["course_code"],
            "course_term": _term_view(dependent),
            "prerequisite_condition": condition,
            "prerequisite_terms": candidate_term_rows,
            "status": status,
            "reason": reason,
            "paths": [[candidate["course_code"], dependent["course_code"]] for candidate in candidates],
            "provenance": _merge_provenance(plan_refs, prereq_refs, dependent["provenance"], *[c["provenance"] for c in candidates]),
        }
        relationships.append(rel)
    return relationships, limitations


def _graph_summary(relationships: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    adjacency: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    nodes: set[str] = set()
    for rel in relationships:
        target = str(rel["dependent_course_code"])
        condition = rel["prerequisite_condition"]
        sources = condition.get("candidate_course_codes", []) if condition.get("kind") == "alternative_group" else [condition.get("course_code")]
        for source in sources:
            if not source:
                continue
            source = str(source)
            nodes.update((source, target))
            adjacency.setdefault(source, []).append((target, rel))

    state: dict[str, int] = {}
    stack: list[str] = []
    cycles: set[tuple[str, ...]] = set()

    def visit(node: str) -> None:
        state[node] = 1
        stack.append(node)
        for target, _ in adjacency.get(node, []):
            if state.get(target, 0) == 0:
                visit(target)
            elif state.get(target) == 1:
                start = stack.index(target)
                cycle = tuple(stack[start:] + [target])
                cycles.add(cycle)
        stack.pop()
        state[node] = 2

    for node in sorted(nodes):
        if state.get(node, 0) == 0:
            visit(node)

    paths: list[dict[str, Any]] = []
    for target in sorted({rel["dependent_course_code"] for rel in relationships}):
        def walk(node: str, reverse_path: list[str], rels: list[dict[str, Any]], seen: set[str]) -> None:
            if node in seen:
                return
            parents = [
                (source, rel)
                for source, outgoing in adjacency.items()
                for edge_target, rel in outgoing
                if edge_target == node
            ]
            if not parents:
                if len(reverse_path) >= 3:
                    forward = list(reversed(reverse_path))
                    path_rels = list(reversed(rels))
                    state_value = "incomplete_evidence" if any(r["status"] == "incomplete_evidence" for r in path_rels) else (
                        "violation" if any(r["status"] == "violation" for r in path_rels) else "satisfied"
                    )
                    paths.append({
                        "course_codes": forward,
                        "prerequisite_ids": [r["prerequisite_id"] for r in path_rels],
                        "status": state_value,
                        "provenance": _merge_provenance(*[r["provenance"] for r in path_rels]),
                    })
                return
            for source, rel in parents:
                walk(source, reverse_path + [source], rels + [rel], seen | {node})

        walk(str(target), [str(target)], [], set())
    cycle_rows = [{"course_codes": list(cycle), "status": "incomplete_evidence"} for cycle in sorted(cycles)]
    return paths, cycle_rows


def _run_validation(
    db_path: str | Path,
    program: Any,
    plan: Any,
    semester_assignments: list[dict[str, Any]] | None,
    catalog_key: str | None = None,
) -> dict[str, Any]:
    if not isinstance(program, str) or not program.strip() or not isinstance(plan, str) or not plan.strip():
        return _failure("invalid_scope", program, plan, "explicit program and plan are required")
    try:
        with closing(_connect_read_only(db_path)) as connection:
            scope, scope_error = _resolve_plan(
                connection, program, plan, catalog_key
            )
            if scope_error:
                return _failure(scope_error, program, plan, "the explicit program/plan scope is unresolved")
            assert scope is not None
            courses, limitations = _load_plan_courses(connection, scope)
            candidate_terms: dict[str, int] | None = None
            if semester_assignments is not None:
                if not isinstance(semester_assignments, list):
                    return _failure("invalid_candidate_sequence", scope["program"], scope["plan"], "assignments must be a list")
                candidate_terms = {}
                for assignment in semester_assignments:
                    if not isinstance(assignment, dict):
                        return _failure("invalid_candidate_sequence", scope["program"], scope["plan"], "each assignment must be an object")
                    code = assignment.get("course_code")
                    term_index = assignment.get("term_index")
                    if not isinstance(code, str) or not code.strip() or isinstance(term_index, bool) or not isinstance(term_index, int) or term_index < 1:
                        return _failure("invalid_candidate_sequence", scope["program"], scope["plan"], "each assignment needs a course_code and positive integer term_index")
                    normalized = code.strip().upper()
                    if normalized in candidate_terms:
                        return _failure("invalid_candidate_sequence", scope["program"], scope["plan"], "duplicate logical course assignment")
                    if normalized not in courses:
                        return _failure("unknown_course", scope["program"], scope["plan"], f"course {code.strip()} is not in the resolved plan")
                    candidate_terms[normalized] = term_index
            if candidate_terms is not None:
                for normalized in candidate_terms:
                    courses[normalized]["term_ordinal"] = candidate_terms[normalized]
                    courses[normalized]["candidate_term_index"] = candidate_terms[normalized]
            plan_refs = _provenance_for(connection, "curriculum_plan_provenance", "plan_id", scope["plan_id"])
            if not plan_refs:
                limitations.append("resolved plan lacks linked source evidence")
            relationships, relation_limitations = _load_relationships(
                connection, scope, courses, candidate_terms=candidate_terms
            )
            limitations.extend(relation_limitations)
            transitive_paths, cycles = _graph_summary(relationships)
            for cycle in cycles:
                cycle_codes = set(cycle["course_codes"])
                for rel in relationships:
                    if rel["dependent_course_code"] in cycle_codes or any(
                        code in cycle_codes for path in rel["paths"] for code in path
                    ):
                        rel["status"] = "incomplete_evidence"
                        rel["reason"] = "prerequisite graph contains a cycle"
            violations = [rel for rel in relationships if rel["status"] == "violation"]
            incomplete = [rel for rel in relationships if rel["status"] == "incomplete_evidence"]
            if cycles:
                limitations.append("prerequisite graph contains a cycle")
            if limitations or incomplete:
                status = "incomplete_evidence"
            elif violations:
                status = "violation"
            else:
                status = "satisfied"
            return {
                "status": status,
                "program": scope["program"],
                "plan": scope["plan"],
                "catalog_key": scope["catalog_key"],
                "relationships": relationships,
                "transitive_paths": transitive_paths,
                "violations": violations,
                "incomplete": incomplete,
                "cycles": cycles,
                "summary": {
                    "satisfied": sum(rel["status"] == "satisfied" for rel in relationships),
                    "violations": sum(rel["status"] == "violation" for rel in relationships),
                    "incomplete": sum(rel["status"] == "incomplete_evidence" for rel in relationships),
                    "transitive_paths": len(transitive_paths),
                    "cycles": len(cycles),
                },
                "provenance": plan_refs,
                "limitations": list(dict.fromkeys(limitations)),
            }
    except (OSError, sqlite3.Error) as exc:
        return _failure("database_error", program, plan, f"canonical database could not be read ({type(exc).__name__})")


def validate_plan_prerequisite_sequence(
    db_path: str | Path,
    program: str,
    plan: str,
    catalog_key: str | None = None,
) -> dict[str, Any]:
    """Validate structured prerequisite order in a canonical plan."""
    return _run_validation(db_path, program, plan, None, catalog_key)


def validate_candidate_sequence(
    db_path: str | Path,
    program: str,
    plan: str,
    semester_assignments: list[dict[str, Any]],
    catalog_key: str | None = None,
) -> dict[str, Any]:
    """Validate an explicitly supplied course-to-term assignment."""
    return _run_validation(
        db_path, program, plan, semester_assignments, catalog_key
    )


__all__ = ["validate_candidate_sequence", "validate_plan_prerequisite_sequence"]
