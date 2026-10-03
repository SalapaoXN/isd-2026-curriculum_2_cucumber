"""Bounded deterministic seven-regular-term curriculum sequence planner."""

from __future__ import annotations

import itertools
import math
import sqlite3
from collections import defaultdict
from contextlib import closing
from pathlib import Path
from typing import Any

from backend.hard_plan_validate import validate_curriculum_plan_structure
from backend.hard_prerequisite_validate import (
    validate_candidate_sequence,
    validate_plan_prerequisite_sequence,
)
from rag.structured.queries import _merge_provenance, _provenance_for


_HORIZON_TERMS = 7
_MAX_CHOICE_VARIANTS = 128


def _failure(status: str, program: Any, plan: Any, horizon_terms: Any, reason: str) -> dict[str, Any]:
    return {
        "status": status,
        "sequence_feasible": None,
        "program": program,
        "plan": plan,
        "horizon_terms": horizon_terms,
        "terms": [],
        "moved_from_baseline": [],
        "prerequisite_validation": {"status": "not_run", "validations": []},
        "represented_credit_total": None,
        "required_program_credits": None,
        "unresolved_requirements": [{"type": "plan_scope", "reason": reason}],
        "limitations": [reason],
        "evidence": [],
        "actual_course_offering_unverified": True,
        "graduation_guaranteed": False,
    }


def _read_only(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _resolve_plan(connection: sqlite3.Connection, program: str, plan: str) -> tuple[dict[str, Any] | None, str | None]:
    rows = connection.execute(
        """SELECT cp.plan_id, cp.catalog_id, cp.program_code, cp.plan_key
           FROM curriculum_plans cp
           JOIN programs p ON p.program_id=cp.program_id
           WHERE UPPER(cp.program_code)=? AND UPPER(p.program_code_normalized)=?
             AND p.catalog_id=cp.catalog_id AND LOWER(cp.plan_key)=?
           ORDER BY cp.plan_id""",
        (program.strip().upper(), program.strip().upper(), plan.strip().casefold()),
    ).fetchall()
    if not rows:
        return None, "plan_not_found"
    if len(rows) != 1:
        return None, "ambiguous_plan"
    return {
        "plan_id": int(rows[0]["plan_id"]),
        "catalog_id": int(rows[0]["catalog_id"]),
        "program": str(rows[0]["program_code"]),
        "plan": str(rows[0]["plan_key"]),
    }, None


def _term_index(year: Any, semester: Any) -> int | None:
    if (
        isinstance(year, int) and not isinstance(year, bool) and 1 <= year <= 5
        and isinstance(semester, int) and not isinstance(semester, bool) and semester in (1, 2)
    ):
        return (year - 1) * 2 + semester
    return None


def _credit_value(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    integer = int(value)
    return integer if integer >= 0 and integer == value else None


def _policy_fact(connection: sqlite3.Connection, fact_key: str, context: str) -> dict[str, Any] | None:
    rows = connection.execute(
        """SELECT fact_id, fact_key, value, unit, context, source_rule_id
           FROM policy_facts WHERE fact_key=? AND context=? ORDER BY fact_id""",
        (fact_key, context),
    ).fetchall()
    if len(rows) != 1:
        return None
    row = rows[0]
    refs = _provenance_for(connection, "policy_fact_provenance", "fact_id", int(row["fact_id"]))
    value = _credit_value(row["value"])
    if value is None or not refs:
        return None
    return {
        "fact_id": int(row["fact_id"]),
        "fact_key": str(row["fact_key"]),
        "value": value,
        "unit": row["unit"],
        "context": str(row["context"]),
        "source_rule_id": str(row["source_rule_id"]),
        "provenance": refs,
    }


def _course_nodes(
    connection: sqlite3.Connection,
    scope: dict[str, Any],
    h2_courses: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    nodes: dict[str, dict[str, Any]] = {}
    issues: list[str] = []
    for required in h2_courses:
        code = str(required.get("course_code_normalized") or "").strip().upper()
        if not code:
            issues.append("a mandatory course lacks normalized logical identity")
            continue
        rows = connection.execute(
            """SELECT course_id, catalog_id, course_code, course_code_normalized,
                      name_th, name_en, credit_units
               FROM courses WHERE catalog_id=? AND UPPER(course_code_normalized)=?
               ORDER BY course_id""",
            (scope["catalog_id"], code),
        ).fetchall()
        if len(rows) != 1:
            issues.append(f"mandatory course {code} has missing or ambiguous canonical identity")
            continue
        course = rows[0]
        placements = connection.execute(
            """SELECT placement_id, year_number, semester_number,
                      flexible_year_number, flexible_semester_number,
                      flexible_year_semester_raw, placement_order
               FROM plan_placements WHERE plan_id=? AND course_id=?
               ORDER BY placement_id""",
            (scope["plan_id"], int(course["course_id"])),
        ).fetchall()
        expected_ids = {int(value) for value in required.get("placement_ids", [])}
        placements = [row for row in placements if int(row["placement_id"]) in expected_ids]
        if not placements or len(placements) != len(expected_ids):
            issues.append(f"mandatory course {code} placement evidence is incomplete")
        indexes = {_term_index(row["year_number"], row["semester_number"]) for row in placements}
        baseline_issue = None
        if (
            len(indexes) == 1 and None not in indexes
            and all(not any(row[key] is not None for key in ("flexible_year_number", "flexible_semester_number"))
                    and not (isinstance(row["flexible_year_semester_raw"], str) and row["flexible_year_semester_raw"].strip())
                    for row in placements)
        ):
            baseline = next(iter(indexes))
        else:
            baseline = None
            baseline_issue = "baseline_placement_not_a_single_fixed_regular_term"
        credits = _credit_value(course["credit_units"])
        nodes[code] = {
            "course_code": str(course["course_code"]),
            "course_code_normalized": code,
            "name_th": required.get("name_th") or course["name_th"],
            "name_en": required.get("name_en") or course["name_en"],
            "credit_units": credits,
            "baseline_term_index": baseline,
            "baseline_placement_ids": sorted(expected_ids),
            "placement_order": min(
                (int(row["placement_order"]) for row in placements if row["placement_order"] is not None),
                default=10**9,
            ),
            "course_id": int(course["course_id"]),
            "provenance": list(required.get("provenance", [])),
            "baseline_issue": baseline_issue,
        }
        if credits is None:
            issues.append(f"mandatory course {code} has no determinable canonical credit_units")
        if baseline_issue:
            issues.append(f"mandatory course {code} has no single fixed baseline term")
    return nodes, issues


def _choice_slots(
    connection: sqlite3.Connection,
    scope: dict[str, Any],
    h2_groups: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    slots: dict[str, dict[str, Any]] = {}
    issues: list[str] = []
    for group in h2_groups:
        group_id = int(group["alternative_group_id"])
        key = f"choice:{group_id}"
        placement_rows = connection.execute(
            """SELECT placement_id, year_number, semester_number, flexible_year_number,
                      flexible_semester_number, flexible_year_semester_raw,
                      placement_order
               FROM plan_placements WHERE plan_id=? AND alternative_group_id=?
               ORDER BY placement_id""",
            (scope["plan_id"], group_id),
        ).fetchall()
        indexes = {_term_index(row["year_number"], row["semester_number"]) for row in placement_rows}
        if (
            len(indexes) == 1 and None not in indexes
            and all(row["flexible_year_number"] is None and row["flexible_semester_number"] is None
                    and not (isinstance(row["flexible_year_semester_raw"], str) and row["flexible_year_semester_raw"].strip())
                    for row in placement_rows)
        ):
            baseline = next(iter(indexes))
        else:
            baseline = None
            issues.append(f"alternative group {group_id} has no single fixed baseline term")
        candidates = []
        for candidate in group.get("candidates", []):
            code = str(candidate.get("course_code_normalized") or "").strip().upper()
            rows = connection.execute(
                """SELECT course_id, course_code, course_code_normalized, name_th,
                          name_en, credit_units FROM courses
                   WHERE catalog_id=? AND UPPER(course_code_normalized)=?
                   ORDER BY course_id""",
                (scope["catalog_id"], code),
            ).fetchall()
            if len(rows) != 1:
                issues.append(f"alternative group {group_id} candidate {code or '?'} has unresolved identity")
                continue
            row = rows[0]
            credits = _credit_value(row["credit_units"])
            if credits is None:
                issues.append(f"alternative group {group_id} candidate {code} has unknown credits")
            candidates.append({
                "course_code": str(row["course_code"]),
                "course_code_normalized": str(row["course_code_normalized"]).strip().upper(),
                "name_th": candidate.get("name_th") or row["name_th"],
                "name_en": candidate.get("name_en") or row["name_en"],
                "credit_units": credits,
                "course_id": int(row["course_id"]),
                "member_ids": list(candidate.get("member_ids", [])),
                "provenance": list(candidate.get("provenance", [])),
            })
        minimum = group.get("minimum_choices")
        maximum = group.get("maximum_choices")
        structurally_complete = (
            group.get("status") == "complete"
            and isinstance(minimum, int) and minimum >= 1
            and isinstance(maximum, int) and maximum >= minimum
            and maximum <= len(candidates)
            and bool(group.get("provenance"))
        )
        if not structurally_complete:
            issues.append(f"alternative group {group_id} has unresolved choice semantics")
        choices = sorted(candidates, key=lambda item: (item["credit_units"] is None, item["credit_units"] or 0, item["course_code_normalized"]))
        if structurally_complete and all(item["credit_units"] is not None for item in choices):
            minimum_credits = sum(item["credit_units"] for item in choices[:minimum])
            maximum_credits = sum(item["credit_units"] for item in choices[-minimum:])
        else:
            minimum_credits = maximum_credits = None
        slots[key] = {
            "slot_id": key,
            "alternative_group_id": group_id,
            "minimum_choices": minimum if isinstance(minimum, int) else None,
            "maximum_choices": maximum if isinstance(maximum, int) else None,
            "candidates": sorted(candidates, key=lambda item: item["course_code_normalized"]),
            "baseline_term_index": baseline,
            "baseline_placement_ids": [int(row["placement_id"]) for row in placement_rows],
            "placement_order": min((int(row["placement_order"]) for row in placement_rows if row["placement_order"] is not None), default=10**9),
            "minimum_required_choice_credits": minimum_credits,
            "maximum_required_choice_credits": maximum_credits,
            "provenance": list(group.get("provenance", [])),
            "structurally_complete": structurally_complete,
        }
    return slots, issues


def _graph_constraints(
    h3_result: dict[str, Any],
    mandatory: dict[str, dict[str, Any]],
    slots: dict[str, dict[str, Any]],
) -> tuple[dict[str, set[str]], list[dict[str, Any]]]:
    graph: dict[str, set[str]] = {code: set() for code in mandatory}
    for slot_id in slots:
        graph.setdefault(slot_id, set())
    unresolved: list[dict[str, Any]] = []
    slot_for_group = {slot["alternative_group_id"]: key for key, slot in slots.items()}
    slot_for_candidate: dict[str, set[str]] = defaultdict(set)
    for key, slot in slots.items():
        for candidate in slot["candidates"]:
            slot_for_candidate[candidate["course_code_normalized"]].add(key)
    for relationship in h3_result.get("relationships", []):
        dependent = str(relationship.get("dependent_course_code") or "").strip().upper()
        target_nodes: list[str] = []
        if dependent in mandatory:
            target_nodes.append(dependent)
        target_nodes.extend(slot_for_candidate.get(dependent, set()))
        if not target_nodes:
            continue
        condition = relationship.get("prerequisite_condition", {})
        kind = condition.get("kind")
        if kind == "direct":
            prerequisite = str(condition.get("course_code") or "").strip().upper()
            sources = [prerequisite] if prerequisite else []
        elif kind == "alternative_group":
            group_id = condition.get("alternative_group_id")
            slot_id = slot_for_group.get(group_id)
            sources = [slot_id] if slot_id else []
            if slot_id is None:
                unresolved.append({
                    "type": "alternative_prerequisite",
                    "dependent_course_code": dependent,
                    "alternative_group_id": group_id,
                    "reason": "prerequisite choice group is not represented by a plan choice slot",
                    "provenance": relationship.get("provenance", []),
                })
        else:
            unresolved.append({
                "type": "prerequisite_semantics",
                "dependent_course_code": dependent,
                "condition": condition,
                "reason": "prerequisite is raw-only or structurally unresolved",
                "provenance": relationship.get("provenance", []),
            })
            continue
        for source in sources:
            if source in mandatory or source in slots:
                for target in target_nodes:
                    if source != target:
                        graph.setdefault(source, set()).add(target)
            elif source in slot_for_candidate:
                mapped_slots = slot_for_candidate[source]
                if len(mapped_slots) == 1:
                    slot_id = next(iter(mapped_slots))
                    for target in target_nodes:
                        if slot_id != target:
                            graph.setdefault(slot_id, set()).add(target)
                else:
                    unresolved.append({
                        "type": "ambiguous_alternative_prerequisite",
                        "course_code": source,
                        "reason": "prerequisite course belongs to multiple choice slots",
                        "provenance": relationship.get("provenance", []),
                    })
            else:
                unresolved.append({
                    "type": "unrepresented_prerequisite_course",
                    "course_code": source,
                    "dependent_course_code": dependent,
                    "reason": "prerequisite is not a represented mandatory course or choice candidate",
                    "provenance": relationship.get("provenance", []),
                })
    return graph, unresolved


def _topological_order(
    graph: dict[str, set[str]],
    nodes: dict[str, dict[str, Any]],
    slots: dict[str, dict[str, Any]],
) -> list[str] | None:
    indegree = {node: 0 for node in graph}
    for targets in graph.values():
        for target in targets:
            indegree[target] = indegree.get(target, 0) + 1

    def sort_key(key: str) -> tuple[int, int, str]:
        row = nodes.get(key) or slots.get(key) or {}
        baseline = row.get("baseline_term_index")
        return (baseline if isinstance(baseline, int) else _HORIZON_TERMS + 1,
                row.get("placement_order", 10**9), key)

    available = sorted((node for node, count in indegree.items() if count == 0), key=sort_key)
    ordered: list[str] = []
    while available:
        node = available.pop(0)
        ordered.append(node)
        for target in sorted(graph.get(node, set()), key=sort_key):
            indegree[target] -= 1
            if indegree[target] == 0:
                available.append(target)
                available.sort(key=sort_key)
    return ordered if len(ordered) == len(indegree) else None


def _assign_terms(
    ordered: list[str],
    graph: dict[str, set[str]],
    nodes: dict[str, dict[str, Any]],
    slots: dict[str, dict[str, Any]],
    cap: int | None,
) -> tuple[dict[str, int] | None, list[str]]:
    predecessors: dict[str, set[str]] = {node: set() for node in graph}
    for source, targets in graph.items():
        for target in targets:
            predecessors[target].add(source)
    baseline_due = {
        key: max(1, min(_HORIZON_TERMS, int((nodes.get(key) or slots.get(key) or {}).get("baseline_term_index") or 1)))
        for key in graph
    }
    latest = {key: _HORIZON_TERMS for key in graph}
    latest.update({key: baseline_due[key] for key in graph if (nodes.get(key) or slots.get(key) or {}).get("baseline_term_index") is not None})
    for node in reversed(ordered):
        for predecessor in predecessors[node]:
            latest[predecessor] = min(latest[predecessor], latest[node] - 1)

    assignments: dict[str, int] = {}
    loads = [0] * (_HORIZON_TERMS + 1)
    unknown_weight: set[str] = set()
    for key in ordered:
        row = nodes.get(key) or slots.get(key) or {}
        earliest = max((assignments[pred] + 1 for pred in predecessors[key]), default=1)
        baseline = row.get("baseline_term_index")
        preferred = max(1, min(_HORIZON_TERMS, baseline)) if isinstance(baseline, int) else earliest
        deadline = max(earliest, min(_HORIZON_TERMS, latest[key]))
        weight = row.get("credit_units") if key in nodes else row.get("minimum_required_choice_credits")
        if weight is None:
            unknown_weight.add(key)
            weight = 0
        options = list(range(earliest, _HORIZON_TERMS + 1))
        preferred = min(preferred, deadline)
        options.sort(key=lambda term: (abs(term - preferred), term))
        chosen = next((term for term in options if cap is None or loads[term] + weight <= cap), None)
        if chosen is None:
            return None, [f"no seven-term placement remains for {key} under the normal credit cap"]
        assignments[key] = chosen
        loads[chosen] += weight
    return assignments, [f"credit load is not known for {key}" for key in sorted(unknown_weight)]


def _variant_sets(slots: dict[str, dict[str, Any]]) -> list[dict[str, tuple[str, ...]]] | None:
    choices: list[tuple[str, list[tuple[str, ...]]]] = []
    for key, slot in sorted(slots.items()):
        minimum = slot.get("minimum_choices")
        candidates = [item["course_code_normalized"] for item in slot["candidates"]]
        if not isinstance(minimum, int) or minimum < 1 or minimum > len(candidates):
            return None
        count = math.comb(len(candidates), minimum)
        if count > _MAX_CHOICE_VARIANTS:
            return None
        choices.append((key, list(itertools.combinations(candidates, minimum))))
    total = math.prod(len(options) for _, options in choices) if choices else 1
    if total > _MAX_CHOICE_VARIANTS:
        return None
    variants: list[dict[str, tuple[str, ...]]] = []
    option_lists = [options for _, options in choices]
    for combo in itertools.product(*option_lists) if option_lists else [()]:
        variants.append({choices[index][0]: tuple(combo[index]) for index in range(len(choices))})
    return variants


def _add_evidence(evidence: list[dict[str, Any]], *groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _merge_provenance(evidence, *groups)


def plan_curriculum_sequence(
    db_path: str | Path,
    program: str,
    plan: str,
    horizon_terms: int = _HORIZON_TERMS,
) -> dict[str, Any]:
    """Schedule represented mandatory courses and choice slots over seven regular terms."""
    if not isinstance(program, str) or not program.strip() or not isinstance(plan, str) or not plan.strip():
        return _failure("invalid_scope", program, plan, horizon_terms, "explicit program and plan are required")
    if isinstance(horizon_terms, bool) or horizon_terms != _HORIZON_TERMS:
        return _failure("unsupported_horizon", program, plan, horizon_terms, "only seven regular terms are supported")
    try:
        structure = validate_curriculum_plan_structure(db_path, program, plan)
        if structure["status"] in {"plan_not_found", "ambiguous_plan", "invalid_scope", "database_error"}:
            return _failure(structure["status"], program, plan, horizon_terms, "H2 could not resolve the explicit canonical plan")
        if structure.get("mandatory_courses", {}).get("status") != "complete":
            return _failure("incomplete_evidence", structure.get("program"), structure.get("plan"), horizon_terms, "H2 could not establish a complete mandatory course set")

        with closing(_read_only(db_path)) as connection:
            scope, scope_error = _resolve_plan(connection, structure["program"], structure["plan"])
            if scope_error or scope is None:
                return _failure(scope_error or "plan_not_found", program, plan, horizon_terms, "canonical plan scope did not resolve uniquely")
            max_policy = _policy_fact(connection, "regular_maximum", "regular_semester")
            minimum_policy = _policy_fact(connection, "regular_minimum", "regular_semester")
            exception_policy = _policy_fact(connection, "graduation_exception_maximum", "graduation_exception")
            nodes, course_issues = _course_nodes(
                connection, scope, structure["mandatory_courses"]["courses"]
            )
            slots, slot_issues = _choice_slots(
                connection, scope, structure.get("alternative_groups", [])
            )
            requirement_rows = structure.get("credit_requirements", [])
            program_credit_row = next((row for row in requirement_rows if row.get("requirement_type") == "total_program_credits"), None)
            required_program_credits = _credit_value(program_credit_row.get("required_value")) if program_credit_row else None
            policy_issue = None
            if max_policy is None:
                policy_issue = "canonical regular-semester maximum or its provenance is unresolved"
            cap = max_policy["value"] if max_policy else None
            h3_plan = validate_plan_prerequisite_sequence(db_path, structure["program"], structure["plan"])
            graph, graph_issues = _graph_constraints(h3_plan, nodes, slots)
            unresolved: list[dict[str, Any]] = []
            limitations: list[str] = ["actual_course_offering_unverified"]
            if minimum_policy is not None:
                limitations.append("regular_minimum_not_enforced_applicability_not_established")
            if policy_issue:
                unresolved.append({"type": "regular_credit_policy", "reason": policy_issue})
                limitations.append(policy_issue)
            if not exception_policy:
                limitations.append("conditional_overload_policy_not_resolved_and_not_applied")
            if structure.get("checks"):
                placement_check = next((item for item in structure["checks"] if item.get("check") == "placement_quality"), None)
                if placement_check and placement_check.get("status") != "complete":
                    limitations.append("some baseline placements are flexible, untimed, or otherwise unresolved; they remain preferences only")
            for issue in course_issues + slot_issues:
                limitations.append(issue)
                unresolved.append({"type": "course_or_choice_evidence", "reason": issue})
            unresolved.extend(graph_issues)
            if h3_plan.get("cycles"):
                unresolved.append({"type": "prerequisite_cycle", "cycles": h3_plan["cycles"]})
            if h3_plan.get("status") == "database_error":
                unresolved.append({"type": "prerequisite_evidence", "reason": "H3 could not read canonical prerequisite evidence"})

            for row in requirement_rows:
                if row.get("status") != "complete":
                    unresolved.append({
                        "type": "program_credit_total_unmeasured",
                        "requirement_type": row.get("requirement_type"),
                        "required_value": row.get("required_value"),
                        "reason": row.get("limitation") or "H2 could not measure canonical required credits",
                        "provenance": row.get("provenance", []),
                    })
            for item in structure.get("unassessable_requirements", []):
                if item.get("kind") == "plan_semantics":
                    unresolved.append({
                        "type": item.get("requirement_type", "unresolved_plan_semantics"),
                        "reason": item.get("reason"),
                        "candidate_placement_count": item.get("candidate_placement_count"),
                        "provenance": item.get("provenance", []),
                    })
            if not requirement_rows:
                unresolved.append({"type": "program_credit_total_missing", "reason": "H2 found no canonical total program-credit requirement"})

            if max_policy:
                nodes_credit_total = sum(node["credit_units"] or 0 for node in nodes.values())
            else:
                nodes_credit_total = sum(node["credit_units"] or 0 for node in nodes.values())
            individually_over_cap = [
                code for code, node in nodes.items()
                if cap is not None and node["credit_units"] is not None and node["credit_units"] > cap
            ]
            if cap is not None and (nodes_credit_total > cap * _HORIZON_TERMS or individually_over_cap):
                terms = _empty_terms()
                evidence = _evidence_bundle(connection, scope, structure, h3_plan, max_policy, minimum_policy, exception_policy, nodes, slots)
                return {
                    "status": "infeasible",
                    "sequence_feasible": False,
                    "program": scope["program"],
                    "plan": scope["plan"],
                    "horizon_terms": _HORIZON_TERMS,
                    "terms": terms,
                    "moved_from_baseline": [],
                    "prerequisite_validation": {"status": "not_run_credit_lower_bound_exceeds_capacity", "validations": []},
                    "represented_credit_total": nodes_credit_total if all(node["credit_units"] is not None for node in nodes.values()) else None,
                    "required_program_credits": required_program_credits,
                    "unresolved_requirements": unresolved,
                    "limitations": list(dict.fromkeys(limitations + (["a mandatory course exceeds the documented normal regular-term maximum" ] if individually_over_cap else []) + (["known mandatory credits exceed seven regular-term capacity"] if nodes_credit_total > cap * _HORIZON_TERMS else []))),
                    "credit_constraints": _credit_constraints(max_policy, minimum_policy, exception_policy),
                    "unknown_credit_courses": sorted(code for code, node in nodes.items() if node["credit_units"] is None),
                    "evidence": evidence,
                    "actual_course_offering_unverified": True,
                    "graduation_guaranteed": False,
                }

            ordered = _topological_order(graph, nodes, slots)
            if ordered is None:
                unresolved.append({"type": "prerequisite_cycle", "reason": "represented sequence constraints contain a cycle"})
                return _failure("incomplete_evidence", scope["program"], scope["plan"], horizon_terms, "prerequisite ordering graph could not be topologically arranged")
            assignments, scheduling_issues = _assign_terms(ordered, graph, nodes, slots, cap)
            if assignments is None:
                return {
                    "status": "incomplete_evidence",
                    "sequence_feasible": None,
                    "program": scope["program"],
                    "plan": scope["plan"],
                    "horizon_terms": _HORIZON_TERMS,
                    "terms": _empty_terms(),
                    "moved_from_baseline": [],
                    "prerequisite_validation": {"status": "not_run_no_assignment", "validations": []},
                    "represented_credit_total": nodes_credit_total if all(node["credit_units"] is not None for node in nodes.values()) else None,
                    "required_program_credits": required_program_credits,
                    "unresolved_requirements": unresolved,
                    "limitations": list(dict.fromkeys(limitations + scheduling_issues + ["bounded deterministic placement did not find a schedule; infeasibility was not proven"])),
                    "credit_constraints": _credit_constraints(max_policy, minimum_policy, exception_policy),
                    "unknown_credit_courses": sorted(code for code, node in nodes.items() if node["credit_units"] is None),
                    "evidence": _evidence_bundle(connection, scope, structure, h3_plan, max_policy, minimum_policy, exception_policy, nodes, slots),
                    "actual_course_offering_unverified": True,
                    "graduation_guaranteed": False,
                }

            variants = _variant_sets(slots)
            if variants is None:
                unresolved.append({"type": "alternative_choice_search_bound", "maximum_variants": _MAX_CHOICE_VARIANTS})
                variants = []
            combination_results: list[dict[str, Any]] = []
            if variants:
                for variant in variants:
                    candidate_terms = [
                        {"course_code": node["course_code"], "term_index": assignments[code]}
                        for code, node in sorted(nodes.items())
                    ]
                    selected_rows = []
                    combination_credit_complete = True
                    for slot_id, selected_codes in variant.items():
                        for code in selected_codes:
                            candidate = next(item for item in slots[slot_id]["candidates"] if item["course_code_normalized"] == code)
                            if code in nodes:
                                if assignments[code] != assignments[slot_id]:
                                    combination_results.append({"choices": variant, "status": "incomplete_evidence", "reason": "one canonical course would be assigned to conflicting required and choice-slot terms", "h3": None})
                                    selected_rows = []
                                    break
                            else:
                                candidate_terms.append({"course_code": candidate["course_code"], "term_index": assignments[slot_id]})
                            if candidate["credit_units"] is None:
                                combination_credit_complete = False
                            else:
                                selected_rows.append((slot_id, candidate["credit_units"]))
                        else:
                            continue
                        break
                    else:
                        h3 = validate_candidate_sequence(db_path, scope["program"], scope["plan"], candidate_terms)
                        loads = [0] * (_HORIZON_TERMS + 1)
                        load_complete = True
                        for code, node in nodes.items():
                            if node["credit_units"] is None:
                                load_complete = False
                            else:
                                loads[assignments[code]] += node["credit_units"]
                        for slot_id, weight in selected_rows:
                            loads[assignments[slot_id]] += weight
                        cap_ok = cap is not None and all(loads[index] <= cap for index in range(1, _HORIZON_TERMS + 1))
                        validation_status = h3.get("status")
                        if validation_status == "violation" or h3.get("violations"):
                            combo_status = "violation"
                        elif validation_status != "satisfied" or not load_complete or not combination_credit_complete:
                            combo_status = "incomplete_evidence"
                        elif not cap_ok:
                            combo_status = "violation"
                        else:
                            combo_status = "feasible"
                        combination_results.append({
                            "choices": {key: list(value) for key, value in variant.items()},
                            "status": combo_status,
                            "credit_cap_satisfied": cap_ok,
                            "h3_status": validation_status,
                            "h3": h3,
                        })

            if not slots:
                candidate_terms = [
                    {"course_code": node["course_code"], "term_index": assignments[code]}
                    for code, node in sorted(nodes.items())
                ]
                h3 = validate_candidate_sequence(db_path, scope["program"], scope["plan"], candidate_terms)
                loads = [0] * (_HORIZON_TERMS + 1)
                load_complete = True
                for code, node in nodes.items():
                    if node["credit_units"] is None:
                        load_complete = False
                    else:
                        loads[assignments[code]] += node["credit_units"]
                cap_ok = cap is not None and all(loads[index] <= cap for index in range(1, 8))
                combination_results.append({
                    "choices": {},
                    "status": "feasible" if h3.get("status") == "satisfied" and load_complete and cap_ok else (
                        "violation" if h3.get("status") == "violation" or h3.get("violations") or (load_complete and not cap_ok)
                        else "incomplete_evidence"
                    ),
                    "credit_cap_satisfied": cap_ok,
                    "h3_status": h3.get("status"),
                    "h3": h3,
                })

            valid_count = sum(item["status"] == "feasible" for item in combination_results)
            unknown_credit_codes = sorted(code for code, node in nodes.items() if node["credit_units"] is None)
            if valid_count:
                sequence_feasible: bool | None = True
            elif any(item["status"] == "incomplete_evidence" for item in combination_results) or unknown_credit_codes or policy_issue or not combination_results:
                sequence_feasible = None
            elif any(item["status"] == "violation" for item in combination_results):
                # A failed candidate is not proof that no seven-term assignment exists.
                sequence_feasible = None
            else:
                sequence_feasible = False
            if sequence_feasible is False:
                overall_status = "infeasible"
            elif sequence_feasible is None or unresolved:
                overall_status = "incomplete_evidence"
            else:
                overall_status = "feasible" if structure.get("status") == "complete" else "incomplete_evidence"

            terms = _build_terms(assignments, nodes, slots, cap)
            moved = _moved_from_baseline(assignments, nodes, slots)
            evidence = _evidence_bundle(connection, scope, structure, h3_plan, max_policy, minimum_policy, exception_policy, nodes, slots)
            for combo in combination_results:
                h3 = combo.get("h3")
                if isinstance(h3, dict):
                    evidence = _add_evidence(evidence, h3.get("provenance", []))
                    for relation in h3.get("relationships", []):
                        evidence = _add_evidence(evidence, relation.get("provenance", []))
            mandatory_credit_total = sum(node["credit_units"] for node in nodes.values()) if all(node["credit_units"] is not None for node in nodes.values()) else None
            choice_credit_min = sum(slot["minimum_required_choice_credits"] for slot in slots.values()) if all(slot["minimum_required_choice_credits"] is not None for slot in slots.values()) else None
            choice_credit_max = sum(slot["maximum_required_choice_credits"] for slot in slots.values()) if all(slot["maximum_required_choice_credits"] is not None for slot in slots.values()) else None
            represented_credit_total = {
                "mandatory_course_credits": mandatory_credit_total,
                "minimum_required_choice_credits": choice_credit_min,
                "maximum_required_choice_credits": choice_credit_max,
                "minimum_possible_total": mandatory_credit_total + choice_credit_min if mandatory_credit_total is not None and choice_credit_min is not None else None,
                "maximum_possible_total": mandatory_credit_total + choice_credit_max if mandatory_credit_total is not None and choice_credit_max is not None else None,
            }
            if unknown_credit_codes:
                unresolved.append({"type": "unknown_course_credits", "course_codes": unknown_credit_codes})
            if any(slot["minimum_required_choice_credits"] is None for slot in slots.values()):
                unresolved.append({"type": "unknown_alternative_credit", "reason": "choice-slot credit range cannot be fully measured"})
            if any(item.get("credit_cap_satisfied") is False for item in combination_results):
                limitations.append("a concrete choice sequence may exceed the normal credit cap")
            if not max_policy:
                sequence_feasible = None
                overall_status = "incomplete_evidence"
            return {
                "status": overall_status,
                "sequence_feasible": sequence_feasible,
                "program": scope["program"],
                "plan": scope["plan"],
                "horizon_terms": _HORIZON_TERMS,
                "terms": terms,
                "moved_from_baseline": moved,
                "prerequisite_validation": {
                    "status": "satisfied" if valid_count else ("incomplete_evidence" if sequence_feasible is None else "violation"),
                    "checked_choice_combinations": len(combination_results),
                    "valid_choice_combinations": valid_count,
                    "validations": combination_results,
                },
                "represented_credit_total": represented_credit_total,
                "required_program_credits": required_program_credits,
                "credit_constraints": _credit_constraints(max_policy, minimum_policy, exception_policy),
                "unknown_credit_courses": unknown_credit_codes,
                "unresolved_requirements": unresolved,
                "limitations": list(dict.fromkeys(limitations + scheduling_issues)),
                "evidence": evidence,
                "actual_course_offering_unverified": True,
                "graduation_guaranteed": False,
            }
    except (OSError, sqlite3.Error, ValueError, TypeError, KeyError) as exc:
        return _failure("database_error", program, plan, horizon_terms, f"canonical sequence evidence could not be read ({type(exc).__name__})")


def _empty_terms() -> list[dict[str, Any]]:
    return [
        {"term_index": index, "year": (index - 1) // 2 + 1, "semester": (index - 1) % 2 + 1,
         "courses": [], "choice_slots": [], "total_known_credits": 0,
         "minimum_possible_credits": 0,
         "maximum_possible_credits": 0}
        for index in range(1, _HORIZON_TERMS + 1)
    ]


def _build_terms(
    assignments: dict[str, int],
    nodes: dict[str, dict[str, Any]],
    slots: dict[str, dict[str, Any]],
    cap: int | None,
) -> list[dict[str, Any]]:
    terms = _empty_terms()
    for code, index in assignments.items():
        term = terms[index - 1]
        if code in nodes:
            node = nodes[code]
            term["courses"].append({
                "course_code": node["course_code"],
                "course_code_normalized": code,
                "name_th": node["name_th"],
                "name_en": node["name_en"],
                "credit_units": node["credit_units"],
                "baseline_term_index": node["baseline_term_index"],
                "evidence_ids": sorted({ref["provenance_id"] for ref in node["provenance"]}),
            })
            if node["credit_units"] is not None:
                term["total_known_credits"] += node["credit_units"]
                term["minimum_possible_credits"] += node["credit_units"]
                term["maximum_possible_credits"] += node["credit_units"]
        elif code in slots:
            slot = slots[code]
            term["choice_slots"].append({
                "type": "alternative_group",
                "slot_id": code,
                "alternative_group_id": slot["alternative_group_id"],
                "minimum_choices": slot["minimum_choices"],
                "maximum_choices": slot["maximum_choices"],
                "candidates": [
                    {key: candidate[key] for key in ("course_code", "course_code_normalized", "name_th", "name_en", "credit_units")}
                    for candidate in slot["candidates"]
                ],
                "required_choice_credit_range": {
                    "minimum": slot["minimum_required_choice_credits"],
                    "maximum": slot["maximum_required_choice_credits"],
                },
                "evidence_ids": sorted({ref["provenance_id"] for ref in slot["provenance"]}),
            })
            if slot["minimum_required_choice_credits"] is not None:
                term["minimum_possible_credits"] += slot["minimum_required_choice_credits"]
            else:
                term["minimum_possible_credits"] = None
            if slot["maximum_required_choice_credits"] is not None:
                term["maximum_possible_credits"] += slot["maximum_required_choice_credits"]
            else:
                term["maximum_possible_credits"] = None
    for term in terms:
        term["courses"].sort(key=lambda item: (item["course_code_normalized"]))
        term["choice_slots"].sort(key=lambda item: item["slot_id"])
        term["credit_cap"] = cap
    return terms


def _moved_from_baseline(
    assignments: dict[str, int],
    nodes: dict[str, dict[str, Any]],
    slots: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    moved = []
    for key, index in sorted(assignments.items()):
        row = nodes.get(key) or slots.get(key) or {}
        baseline = row.get("baseline_term_index")
        if baseline is not None and baseline != index:
            moved.append({
                "course_code": row.get("course_code") if key in nodes else None,
                "choice_slot_id": key if key in slots else None,
                "from_term_index": baseline,
                "to_term_index": index,
                "reason": "prerequisite, horizon, or regular credit-cap scheduling preference",
                "evidence_ids": sorted({ref["provenance_id"] for ref in row.get("provenance", [])}),
            })
    return moved


def _credit_constraints(
    maximum: dict[str, Any] | None,
    minimum: dict[str, Any] | None,
    exception: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "regular_maximum": {key: maximum[key] for key in ("fact_id", "value", "unit", "context", "source_rule_id", "provenance")} if maximum else None,
        "regular_minimum": {
            "fact_id": minimum["fact_id"], "value": minimum["value"], "unit": minimum["unit"],
            "context": minimum["context"], "source_rule_id": minimum["source_rule_id"],
            "provenance": minimum["provenance"], "enforced": False,
            "reason": "registration-specific applicability is not established by plan structure alone",
        } if minimum else None,
        "conditional_overload": {
            "fact_id": exception["fact_id"], "value": exception["value"], "unit": exception["unit"],
            "context": exception["context"], "source_rule_id": exception["source_rule_id"],
            "provenance": exception["provenance"], "applied": False,
            "reason": "conditional eligibility was not provided or established",
        } if exception else {"value": None, "applied": False, "reason": "no conditional overload was applied"},
    }


def _evidence_bundle(
    connection: sqlite3.Connection,
    scope: dict[str, Any],
    structure: dict[str, Any],
    h3: dict[str, Any],
    maximum: dict[str, Any] | None,
    minimum: dict[str, Any] | None,
    exception: dict[str, Any] | None,
    nodes: dict[str, dict[str, Any]],
    slots: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: list[list[dict[str, Any]]] = [
        _provenance_for(connection, "curriculum_plan_provenance", "plan_id", scope["plan_id"]),
    ]
    placement_ids = {placement_id for node in nodes.values() for placement_id in node["baseline_placement_ids"]}
    placement_ids.update(placement_id for slot in slots.values() for placement_id in slot["baseline_placement_ids"])
    for relationship in h3.get("relationships", []):
        dependent = str(relationship.get("dependent_course_code") or "").strip().upper()
        if dependent not in nodes and not any(dependent == c["course_code_normalized"] for s in slots.values() for c in s["candidates"]):
            continue
        groups.append(_provenance_for(connection, "prerequisite_provenance", "prerequisite_id", int(relationship["prerequisite_id"])))
        for term in [relationship.get("course_term", {})] + [item.get("term", {}) for item in relationship.get("prerequisite_terms", [])]:
            placement_ids.update(int(item["placement_id"]) for item in term.get("placements", []) if item.get("placement_id") is not None)
        candidate_codes = [dependent]
        candidate_codes.extend(item.get("course_code") for item in relationship.get("prerequisite_terms", []))
        for code in candidate_codes:
            if code:
                course_rows = connection.execute(
                    "SELECT course_id FROM courses WHERE catalog_id=? AND UPPER(course_code_normalized)=? ORDER BY course_id",
                    (scope["catalog_id"], str(code).strip().upper()),
                ).fetchall()
                if len(course_rows) == 1:
                    groups.append(_provenance_for(connection, "course_provenance", "course_id", int(course_rows[0]["course_id"])))
    for placement_id in sorted(placement_ids):
        groups.append(_provenance_for(connection, "plan_placement_provenance", "placement_id", placement_id))
    for node in nodes.values():
        groups.append(_provenance_for(connection, "course_provenance", "course_id", node["course_id"]))
    for slot in slots.values():
        group_id = slot["alternative_group_id"]
        groups.append(_provenance_for(connection, "alternative_group_provenance", "alternative_group_id", group_id))
        for candidate in slot["candidates"]:
            groups.append(_provenance_for(connection, "course_provenance", "course_id", candidate["course_id"]))
            for member_id in candidate.get("member_ids", []):
                groups.append(_provenance_for(connection, "alternative_group_member_provenance", "alternative_group_member_id", member_id))
    for requirement in structure.get("credit_requirements", []):
        requirement_id = requirement.get("requirement_id")
        if requirement_id is not None:
            groups.append(_provenance_for(connection, "program_requirement_provenance", "requirement_id", int(requirement_id)))
    for policy in (maximum, minimum, exception):
        if policy is not None:
            groups.append(policy.get("provenance", []))
    return _merge_provenance(*groups)


__all__ = ["plan_curriculum_sequence"]
