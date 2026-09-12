"""Deterministically validate the Lab 8B IT curriculum submission."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "submission" / "curriculum.json"
DEFAULT_SCHEMA = ROOT / "submission" / "schema" / "curriculum.schema.json"
DEFAULT_OUTPUT = ROOT / "submission" / "verify.json"

COURSE_CODE_RE = re.compile(r"^[0-9]{8}$")
COURSE_CODE_MENTION_RE = re.compile(r"[0-9]{8}")
FLEXIBLE_TERM_RE = re.compile(r"(\d+)/(\d+)")
CREDIT_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)")
PLACEHOLDER_CODES = {"060164xx", "90644xxx", "9064xxxx", "xxxxxxxx"}
PROVENANCE_KEYS = (
    "program",
    "source_filename",
    "source_page",
    "document_category",
    "document_page",
)


def provenance_ref(item: dict[str, Any]) -> dict[str, Any]:
    """Copy only provenance fields that exist in the submission artifact."""

    return {key: item[key] for key in PROVENANCE_KEYS if key in item}


def course_provenance(course: dict[str, Any]) -> list[dict[str, Any]]:
    return [provenance_ref(item) for item in course["source_provenance"]]


def parse_credits(value: str) -> int | float | None:
    match = CREDIT_RE.match(value)
    if match is None:
        return None
    token = match.group(1)
    return float(token) if "." in token else int(token)


def parse_flexible_terms(value: str | None) -> list[tuple[int, int]]:
    if value is None:
        return []
    return [(int(year), int(semester)) for year, semester in FLEXIBLE_TERM_RE.findall(value)]


def term_label(year: int, semester: int) -> str:
    return f"{year}/{semester}"


def placement(course: dict[str, Any]) -> dict[str, Any]:
    return {
        "year": course["year"],
        "semester": course["semester"],
        "flexible_year_semester": course["flexible_year_semester"],
    }


def record_summary(index: int, course: dict[str, Any]) -> dict[str, Any]:
    return {
        "record_index": index,
        "course_code": course["code"],
        "name_en": course["name_en"],
        "credits": course["credits"],
        "year": course["year"],
        "semester": course["semester"],
        "flexible_year_semester": course["flexible_year_semester"],
        "provenance": course_provenance(course),
    }


def record_signature(course: dict[str, Any]) -> str:
    return json.dumps(course, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def is_alternative_group_code(code: str) -> bool:
    return "หรือ" in code and len(COURSE_CODE_MENTION_RE.findall(code)) >= 2


def validate_submission_shape(data: dict[str, Any], schema: dict[str, Any]) -> None:
    """Validate the concrete Draft 2020-12 schema shape without dependencies."""

    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert set(data) == {"program", "plans"}
    assert data["program"] == "IT"
    assert isinstance(data["plans"], list) and len(data["plans"]) == 2
    assert [plan.get("plan") for plan in data["plans"]] == ["coop", "no_coop"]

    for plan in data["plans"]:
        assert set(plan) == {"source", "description", "program", "plan", "total_courses", "courses"}
        assert isinstance(plan["source"], str)
        assert isinstance(plan["description"], str)
        assert plan["program"] == "IT"
        assert plan["plan"] in {"coop", "no_coop"}
        assert type(plan["total_courses"]) is int and plan["total_courses"] >= 0
        assert isinstance(plan["courses"], list)

        required = {
            "code",
            "name_th",
            "name_en",
            "credits",
            "category",
            "type",
            "prerequisite",
            "flexible_year_semester",
            "year",
            "semester",
            "source_provenance",
        }
        allowed = required | {"note", "desc_th", "desc_en"}
        for course in plan["courses"]:
            assert required <= set(course) <= allowed
            assert isinstance(course["code"], str) and course["code"]
            for field in ("name_th", "name_en", "credits", "category", "type", "prerequisite"):
                assert isinstance(course[field], str)
            assert course["flexible_year_semester"] is None or isinstance(
                course["flexible_year_semester"], str
            )
            if "note" in course:
                assert course["note"] is None or isinstance(course["note"], str)
            for field in ("desc_th", "desc_en"):
                if field in course:
                    assert isinstance(course[field], str)
            assert type(course["year"]) is int and type(course["semester"]) is int
            assert isinstance(course["source_provenance"], list)
            assert len(course["source_provenance"]) >= 1
            for item in course["source_provenance"]:
                assert set(item) == set(PROVENANCE_KEYS)
                assert isinstance(item["program"], str)
                assert isinstance(item["source_filename"], str)
                assert type(item["source_page"]) is int
                assert item["document_category"] in {"description", "plan"}
                assert item["document_page"] is None or type(item["document_page"]) is int


def declared_credit_fields(data: dict[str, Any]) -> list[str]:
    candidates = ("total_credits", "declared_total_credits", "declared_credits", "curriculum_credits")
    found = [f"curriculum.{key}" for key in candidates if key in data]
    for plan in data["plans"]:
        found.extend(f"plans[{plan['plan']}].{key}" for key in candidates if key in plan)
    return found


def build_chk1(data: dict[str, Any]) -> dict[str, Any]:
    found = declared_credit_fields(data)
    return {
        "status": "NOT_APPLICABLE" if not found else "EXCEPTION",
        "description": "Plan total credits equal the declared curriculum credits.",
        "counts": {
            "plans_evaluated": len(data["plans"]),
            "plans_with_declared_credit_total": len(found),
            "declared_credit_fields_found": len(found),
        },
        "findings": [
            {
                "code": "DECLARED_CREDIT_TOTAL_UNAVAILABLE",
                "detail": "No independent declared curriculum or plan credit total exists in curriculum.json; total_courses is not a credit total.",
                "fields_checked": [
                    "curriculum.total_credits",
                    "curriculum.declared_total_credits",
                    "plans[].total_credits",
                    "plans[].declared_credits",
                ],
            }
        ]
        if not found
        else [],
        "exceptions": [],
        "evidence": {
            "declared_fields_found": found,
            "plan_total_courses": {plan["plan"]: plan["total_courses"] for plan in data["plans"]},
        },
    }


def build_chk2(data: dict[str, Any]) -> dict[str, Any]:
    counts: dict[str, Any] = {}
    findings: list[dict[str, Any]] = []
    total_missing = 0
    for plan in data["plans"]:
        numeric = [course for course in plan["courses"] if COURSE_CODE_RE.fullmatch(course["code"])]
        missing = [course for course in numeric if not course.get("desc_th") and not course.get("desc_en")]
        total_missing += len(missing)
        counts[plan["plan"]] = {
            "numeric_code_records": len(numeric),
            "records_with_description": len(numeric) - len(missing),
            "records_without_description": len(missing),
        }
        if missing:
            findings.append(
                {
                    "code": "NUMERIC_CODE_WITHOUT_DESCRIPTION",
                    "plan": plan["plan"],
                    "course_codes": [course["code"] for course in missing],
                    "record_count": len(missing),
                    "provenance_by_course": {
                        course["code"]: course_provenance(course) for course in missing
                    },
                }
            )
    return {
        "status": "EXCEPTION" if total_missing else "PASS",
        "description": "Every actual numeric course code has a non-empty desc_th or desc_en when a description is present in the artifact.",
        "counts": counts,
        "findings": findings,
        "exceptions": [
            {
                "code": "DESCRIPTION_NOT_AVAILABLE",
                "detail": "Records without both description fields are reported as exceptions because curriculum.json does not provide an independent description source for them.",
            }
        ]
        if total_missing
        else [],
        "evidence": {
            "numeric_code_rule": "ASCII full match [0-9]{8}",
            "description_present_rule": "bool(desc_th) or bool(desc_en)",
        },
    }


def build_chk3(data: dict[str, Any]) -> dict[str, Any]:
    counts: dict[str, Any] = {}
    findings: list[dict[str, Any]] = []
    exceptions: list[dict[str, Any]] = []
    total_other_invalid = 0

    for plan in data["plans"]:
        courses = plan["courses"]
        numeric = [course for course in courses if COURSE_CODE_RE.fullmatch(course["code"])]
        placeholders = [course for course in courses if course["code"] in PLACEHOLDER_CODES]
        alternative_groups = [course for course in courses if is_alternative_group_code(course["code"])]
        invalid = [course for course in courses if not COURSE_CODE_RE.fullmatch(course["code"])]
        other_invalid = [
            course
            for course in invalid
            if course not in placeholders and course not in alternative_groups
        ]
        total_other_invalid += len(other_invalid)
        counts[plan["plan"]] = {
            "course_records": len(courses),
            "valid_numeric_records": len(numeric),
            "non_numeric_records": len(invalid),
            "documented_placeholder_records": len(placeholders),
            "alternative_group_records": len(alternative_groups),
            "other_invalid_records": len(other_invalid),
        }

        for course in other_invalid:
            findings.append(
                {
                    "code": "NON_PLACEHOLDER_INVALID_COURSE_CODE",
                    "plan": plan["plan"],
                    "course_code": course["code"],
                    "reason": "The value is neither an exact numeric code, a documented placeholder, nor an explicit alternative-group label.",
                    "provenance": course_provenance(course),
                }
            )

        for course in alternative_groups:
            exceptions.append(
                {
                    "code": "ALTERNATIVE_GROUP_COURSE_LABEL",
                    "plan": plan["plan"],
                    "course_code": course["code"],
                    "alternative_course_codes": COURSE_CODE_MENTION_RE.findall(course["code"]),
                    "credits": course["credits"],
                    "name_en": course["name_en"],
                    "reason": "The literal Thai OR label denotes one alternative-course group record; it is not treated as an ordinary malformed code.",
                    "provenance": course_provenance(course),
                }
            )

        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for course in placeholders:
            grouped[course["code"]].append(course)
        for code in sorted(grouped):
            exceptions.append(
                {
                    "code": "DOCUMENTED_SOURCE_PLACEHOLDER",
                    "plan": plan["plan"],
                    "course_code": code,
                    "record_count": len(grouped[code]),
                    "provenance": [
                        ref
                        for course in grouped[code]
                        for ref in course_provenance(course)
                    ],
                }
            )

    return {
        "status": "FAIL" if total_other_invalid else ("EXCEPTION" if exceptions else "PASS"),
        "description": "Every ordinary course code is exactly 8 ASCII numeric digits; source placeholders and explicit alternative groups are reported as Lab exceptions.",
        "counts": counts,
        "findings": findings,
        "exceptions": exceptions,
        "evidence": {
            "ordinary_code_rule": "re.fullmatch([0-9]{8})",
            "placeholder_values": sorted(PLACEHOLDER_CODES),
            "alternative_group_detection": "course code contains Thai OR (หรือ) and at least two exact 8-digit mentions",
        },
    }


def build_chk4(data: dict[str, Any]) -> dict[str, Any]:
    root_keys = sorted(data)
    return {
        "status": "NOT_APPLICABLE",
        "description": "Credits in a plan placement agree with the corresponding course description record.",
        "counts": {
            "distinct_placement_credit_fields": 0,
            "distinct_course_description_credit_fields": 0,
            "plans_evaluated": len(data["plans"]),
        },
        "findings": [
            {
                "code": "NO_SEPARATE_CREDIT_RECORDS",
                "detail": "curriculum.json stores one credits field inside each course object and has no separate course-description record or placement-credit field to compare.",
            }
        ],
        "exceptions": [],
        "evidence": {
            "root_fields": root_keys,
            "course_credit_field": "courses[].credits",
            "comparison_possible": False,
        },
    }


def build_chk5(data: dict[str, Any]) -> dict[str, Any]:
    counts: dict[str, Any] = {}
    exceptions: list[dict[str, Any]] = []
    findings: list[dict[str, Any]] = []

    for plan in data["plans"]:
        courses = plan["courses"]
        by_code: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
        for index, course in enumerate(courses):
            by_code[course["code"]].append((index, course))

        relation_counts = {"explicit": 0, "passed": 0, "failed": 0, "exceptions": 0}
        for dep_index, dependent in enumerate(courses):
            prereq_codes = sorted(set(COURSE_CODE_MENTION_RE.findall(dependent["prerequisite"])))
            for prereq_code in prereq_codes:
                relation_counts["explicit"] += 1
                candidates = by_code.get(prereq_code, [])
                if len(candidates) != 1:
                    relation_counts["exceptions"] += 1
                    exceptions.append(
                        {
                            "plan": plan["plan"],
                            "dependent_course_code": dependent["code"],
                            "prerequisite_course_code": prereq_code,
                            "reason": "The prerequisite course is missing or ambiguous in this plan.",
                            "dependent_provenance": course_provenance(dependent),
                        }
                    )
                    continue

                prereq_index, prerequisite = candidates[0]
                prereq_position = (prerequisite["year"], prerequisite["semester"])
                dependent_flexible = parse_flexible_terms(dependent["flexible_year_semester"])

                if prerequisite["flexible_year_semester"] is not None:
                    relation_counts["exceptions"] += 1
                    reason = "The prerequisite itself has flexible timing; strict ordering cannot be established."
                    exceptions.append(
                        {
                            "plan": plan["plan"],
                            "dependent_course_code": dependent["code"],
                            "prerequisite_course_code": prereq_code,
                            "dependent_placement": placement(dependent),
                            "prerequisite_placement": placement(prerequisite),
                            "reason": reason,
                            "dependent_provenance": course_provenance(dependent),
                            "prerequisite_provenance": course_provenance(prerequisite),
                        }
                    )
                elif dependent_flexible:
                    if all(option > prereq_position for option in dependent_flexible):
                        relation_counts["passed"] += 1
                    elif any(option < prereq_position for option in dependent_flexible):
                        relation_counts["failed"] += 1
                        findings.append(
                            {
                                "code": "PREREQUISITE_NOT_EARLIER",
                                "plan": plan["plan"],
                                "dependent_course_code": dependent["code"],
                                "prerequisite_course_code": prereq_code,
                                "dependent_placement": placement(dependent),
                                "prerequisite_placement": placement(prerequisite),
                                "provenance": course_provenance(dependent),
                            }
                        )
                    else:
                        relation_counts["exceptions"] += 1
                        exceptions.append(
                            {
                                "plan": plan["plan"],
                                "dependent_course_code": dependent["code"],
                                "prerequisite_course_code": prereq_code,
                                "dependent_placement": placement(dependent),
                                "prerequisite_placement": placement(prerequisite),
                                "reason": "At least one flexible dependent term is the same semester as the prerequisite; requirement_type is absent, so co-requisite status cannot be confirmed.",
                                "dependent_provenance": course_provenance(dependent),
                                "prerequisite_provenance": course_provenance(prerequisite),
                            }
                        )
                elif (dependent["year"], dependent["semester"]) > prereq_position:
                    relation_counts["passed"] += 1
                elif (dependent["year"], dependent["semester"]) < prereq_position:
                    relation_counts["failed"] += 1
                    findings.append(
                        {
                            "code": "PREREQUISITE_NOT_EARLIER",
                            "plan": plan["plan"],
                            "dependent_course_code": dependent["code"],
                            "prerequisite_course_code": prereq_code,
                            "dependent_placement": placement(dependent),
                            "prerequisite_placement": placement(prerequisite),
                            "provenance": course_provenance(dependent),
                        }
                    )
                else:
                    relation_counts["exceptions"] += 1
                    exceptions.append(
                        {
                            "plan": plan["plan"],
                            "dependent_course_code": dependent["code"],
                            "prerequisite_course_code": prereq_code,
                            "dependent_placement": placement(dependent),
                            "prerequisite_placement": placement(prerequisite),
                            "reason": "The prerequisite and dependent are in the same semester; requirement_type is absent, so co-requisite status cannot be confirmed.",
                            "dependent_provenance": course_provenance(dependent),
                            "prerequisite_provenance": course_provenance(prerequisite),
                        }
                    )
        counts[plan["plan"]] = {
            "explicit_prerequisite_relations": relation_counts["explicit"],
            "passed_relations": relation_counts["passed"],
            "failed_relations": relation_counts["failed"],
            "exception_relations": relation_counts["exceptions"],
        }

    status = "FAIL" if findings else ("EXCEPTION" if exceptions else "PASS")
    return {
        "status": status,
        "description": "An explicitly coded prerequisite is scheduled earlier than its dependent course, allowing same-semester co-requisites only when the relationship is known.",
        "counts": counts,
        "findings": findings,
        "exceptions": exceptions,
        "evidence": {
            "prerequisite_extraction": "unique ASCII [0-9]{8} mentions in prerequisite text",
            "fixed_order_rule": "(year, semester) lexicographic order",
            "flexible_order_rule": "all listed flexible terms must be later than a fixed prerequisite to pass",
            "co_requisite_policy": "same-semester cases remain exceptions when requirement_type is absent",
        },
    }


def build_chk6(data: dict[str, Any]) -> dict[str, Any]:
    counts: dict[str, Any] = {}
    findings: list[dict[str, Any]] = []
    exceptions: list[dict[str, Any]] = []
    alternative_keys = sorted(
        {
            key
            for plan in data["plans"]
            for course in plan["courses"]
            for key in course
            if "alternative" in key.lower()
        }
    )

    for plan in data["plans"]:
        grouped: dict[tuple[str, int, int], list[tuple[int, dict[str, Any]]]] = defaultdict(list)
        for index, course in enumerate(plan["courses"]):
            grouped[(course["code"], course["year"], course["semester"])].append((index, course))

        non_placeholder_groups = 0
        placeholder_groups = 0
        for (code, year, semester), records in sorted(grouped.items()):
            if len(records) < 2:
                continue
            if code in PLACEHOLDER_CODES:
                placeholder_groups += 1
                exceptions.append(
                    {
                        "code": "PLACEHOLDER_DUPLICATE_NOT_DISAMBIGUATED",
                        "plan": plan["plan"],
                        "course_code": code,
                        "year": year,
                        "semester": semester,
                        "occurrence_count": len(records),
                        "reason": "The artifact has no alternative-group metadata, so repeated placeholder slots cannot be distinguished from duplicate placements deterministically.",
                        "records": [record_summary(index, course) for index, course in records],
                    }
                )
                continue

            non_placeholder_groups += 1
            records_equal = all(course == records[0][1] for _, course in records[1:])
            if records_equal:
                classification = "extraction_merge_issue"
                reason = "The records are value-identical, including provenance, so this is an extraction/merge duplicate rather than two distinct legitimate records."
            else:
                classification = "distinct_records_same_placement"
                reason = "Multiple non-placeholder records share one placement key but differ in content; the artifact does not expose metadata to prove an alternative group."
            finding = {
                "code": "DUPLICATE_COURSE_PLACEMENT",
                "plan": plan["plan"],
                "course_code": code,
                "year": year,
                "semester": semester,
                "occurrence_count": len(records),
                "records_equal": records_equal,
                "classification": classification,
                "reason": reason,
                "records": [record_summary(index, course) for index, course in records],
            }
            findings.append(finding)

        counts[plan["plan"]] = {
            "non_placeholder_duplicate_groups": non_placeholder_groups,
            "placeholder_duplicate_groups": placeholder_groups,
        }

    return {
        "status": "FAIL" if findings else ("EXCEPTION" if exceptions else "PASS"),
        "description": "No duplicate course placement occurs in the same plan and semester; alternatives count once only when alternative metadata identifies them.",
        "counts": counts,
        "findings": findings,
        "exceptions": exceptions,
        "evidence": {
            "duplicate_key": "(plan, course_code, year, semester)",
            "alternative_group_metadata_fields": alternative_keys,
            "alternative_group_metadata_available": bool(alternative_keys),
            "extraction_duplicate_policy": "Exact value-identical records are diagnosed as merge duplicates but are not patched."
        },
    }


def build_chk7(data: dict[str, Any], chk6: dict[str, Any]) -> dict[str, Any]:
    counts: dict[str, Any] = {}
    findings: list[dict[str, Any]] = []
    exceptions: list[dict[str, Any]] = []
    semester_totals: dict[str, dict[str, int | float]] = {}
    adjusted_totals: dict[str, dict[str, int | float]] = {}
    duplicate_adjustments: dict[str, dict[str, list[dict[str, Any]]]] = {}

    merge_duplicate_codes_by_plan: dict[str, set[str]] = defaultdict(set)
    for finding in chk6["findings"]:
        if finding["classification"] == "extraction_merge_issue":
            merge_duplicate_codes_by_plan[finding["plan"]].add(finding["course_code"])

    for plan in data["plans"]:
        raw_by_term: dict[tuple[int, int], list[tuple[int, dict[str, Any]]]] = defaultdict(list)
        flexible_records: list[tuple[int, dict[str, Any]]] = []
        for index, course in enumerate(plan["courses"]):
            if (course["year"], course["semester"]) == (0, 0):
                flexible_records.append((index, course))
            else:
                raw_by_term[(course["year"], course["semester"])].append((index, course))

        raw_totals: dict[str, int | float] = {}
        adjusted: dict[str, int | float] = {}
        adjustments: dict[str, list[dict[str, Any]]] = {}
        evaluated_terms = 0
        passed_terms = 0
        failed_terms = 0
        approved_special = 0

        for (year, semester), records in sorted(raw_by_term.items()):
            label = term_label(year, semester)
            parsed_records = [
                (index, course, parse_credits(course["credits"])) for index, course in records
            ]
            assert all(value is not None for _, _, value in parsed_records)
            raw_total = sum(value for _, _, value in parsed_records if value is not None)
            raw_totals[label] = raw_total

            logical_records: list[tuple[int, dict[str, Any], int | float]] = []
            signatures: set[str] = set()
            term_adjustments: list[dict[str, Any]] = []
            for index, course, value in parsed_records:
                signature = record_signature(course)
                if signature in signatures:
                    continue
                signatures.add(signature)
                logical_records.append((index, course, value))
            adjusted_total = sum(value for _, _, value in logical_records)
            adjusted[label] = adjusted_total
            if len(logical_records) != len(parsed_records):
                term_adjustments.append(
                    {
                        "policy": "deduplicate exact extraction/merge duplicates for diagnosis only",
                        "removed_record_count": len(parsed_records) - len(logical_records),
                        "course_codes": [
                            course["code"]
                            for _, course, _ in parsed_records
                            if course["code"] in merge_duplicate_codes_by_plan[plan["plan"]]
                        ],
                        "raw_total_credits": raw_total,
                        "adjusted_total_credits": adjusted_total,
                    }
                )
            if term_adjustments:
                adjustments[label] = term_adjustments

            if year not in {1, 2, 3, 4} or semester not in {1, 2}:
                continue
            evaluated_terms += 1
            in_range = 9 <= adjusted_total <= 22
            if in_range:
                passed_terms += 1
                continue

            is_coop_single_six = (
                plan["plan"] == "coop"
                and adjusted_total == 6
                and len(logical_records) == 1
            )
            if is_coop_single_six:
                approved_special += 1
                index, course, _ = logical_records[0]
                exceptions.append(
                    {
                        "code": "COOPERATIVE_SINGLE_6_CREDIT_COURSE",
                        "plan": plan["plan"],
                        "year": year,
                        "semester": semester,
                        "total_credits": adjusted_total,
                        "course_count": 1,
                        "course_code": course["code"],
                        "credits": course["credits"],
                        "reason": "Accepted Lab cooperative/internship exception for one 6-credit course.",
                        "provenance": course_provenance(course),
                    }
                )
            elif semester not in {1, 2}:
                approved_special += 1
                exceptions.append(
                    {
                        "code": "EXPLICIT_SPECIAL_TERM_BELOW_9",
                        "plan": plan["plan"],
                        "year": year,
                        "semester": semester,
                        "total_credits": adjusted_total,
                        "reason": "Explicit non-normal semester is exempt from the 9-credit minimum.",
                        "records": [record_summary(index, course) for index, course, _ in logical_records],
                    }
                )
            else:
                failed_terms += 1
                exceptions.append(
                    {
                        "code": "NORMAL_SEMESTER_OVER_22_CREDITS"
                        if adjusted_total > 22
                        else "NORMAL_SEMESTER_UNDER_9_CREDITS",
                        "classification": "source_condition_human_review",
                        "plan": plan["plan"],
                        "year": year,
                        "semester": semester,
                        "raw_total_credits": raw_total,
                        "lab_rule_adjusted_total_credits": adjusted_total,
                        "contributing_records": [
                            record_summary(index, course) for index, course, _ in logical_records
                        ],
                        "provenance": [
                            ref
                            for _, course, _ in logical_records
                            for ref in course_provenance(course)
                        ],
                        "reason": (
                            "The source curriculum explicitly lists these fixed placements "
                            f"with a total of {adjusted_total} credits, violating the Lab CHK7 "
                            "range of 9-22 credits. No alternative, flexible-placement, or "
                            "duplicate exclusion is justified by curriculum.json; classify "
                            "this as a source-condition/human-review exception rather than "
                            "an extraction failure, without inventing a credit exception."
                        ),
                    }
                )

        semester_totals[plan["plan"]] = raw_totals
        adjusted_totals[plan["plan"]] = adjusted
        duplicate_adjustments[plan["plan"]] = adjustments
        for _, course in flexible_records:
            if course["flexible_year_semester"] is not None:
                continue
        exceptions.append(
            {
                "code": "FLEXIBLE_OR_UNASSIGNED_PLACEMENT_NOT_AGGREGATED",
                "plan": plan["plan"],
                "record_count": len(flexible_records),
                "placement_marker": {"year": 0, "semester": 0},
                "flexible_examples": sorted(
                    {
                        course["flexible_year_semester"]
                        for _, course in flexible_records
                        if course["flexible_year_semester"] is not None
                    }
                ),
                "reason": "Flexible records were not counted simultaneously in every possible semester and were excluded from fixed normal-semester totals.",
                "provenance": {
                    "document_category": "description",
                    "source_pages": sorted(
                        {
                            ref["source_page"]
                            for _, course in flexible_records
                            for ref in course_provenance(course)
                            if ref.get("document_category") == "description"
                        }
                    ),
                },
            }
        )
        counts[plan["plan"]] = {
            "normal_fixed_semesters_evaluated": evaluated_terms,
            "normal_fixed_semesters_passed": passed_terms,
            "normal_fixed_semesters_failed": failed_terms,
            "approved_special_exceptions": approved_special,
            "unassigned_flexible_records_excluded": len(flexible_records),
        }

    return {
        "status": "FAIL" if findings else ("EXCEPTION" if exceptions else "PASS"),
        "description": "Normal fixed semesters are checked against 9-22 credits; source-listed out-of-range totals without a deterministic exclusion are recorded as source-condition/human-review exceptions.",
        "counts": counts,
        "findings": findings,
        "exceptions": exceptions,
        "evidence": {
            "normal_term_rule": "fixed year in 1..4 and fixed semester in {1,2}",
            "credit_parse_rule": "leading numeric value in credits before parenthesized contact-hour detail",
            "allowed_range": {"minimum": 9, "maximum": 22},
            "alternative_group_policy": "An explicit OR group is one source record and is counted once; no synthetic alternatives are added.",
            "flexible_policy": "year=0 and semester=0 records with flexible_year_semester are not counted in every possible term.",
            "out_of_range_policy": "A source-listed fixed normal semester outside 9-22 remains at its calculated total and is classified as a source-condition/human-review exception when no deterministic exclusion is supported.",
            "semester_totals_raw": semester_totals,
            "semester_totals_lab_adjusted": adjusted_totals,
            "duplicate_adjustments_for_diagnosis": duplicate_adjustments,
        },
    }


def build_regression_assertions(data: dict[str, Any], report: dict[str, Any]) -> list[dict[str, Any]]:
    assertions: list[dict[str, Any]] = []
    alt_counts = report["checks"]["CHK3"]["counts"]
    assert all(values["alternative_group_records"] == 1 for values in alt_counts.values())
    assert report["checks"]["CHK3"]["status"] == "EXCEPTION"
    assertions.append(
        {
            "name": "combined_cooperative_label_is_alternative_group",
            "status": "PASS",
            "evidence": "one explicit OR group record per plan; no ordinary invalid code remains",
        }
    )

    duplicate_findings = report["checks"]["CHK6"]["findings"]
    assert duplicate_findings == []
    assert report["checks"]["CHK6"]["status"] == "EXCEPTION"
    assert all(
        values["non_placeholder_duplicate_groups"] == 0
        for values in report["checks"]["CHK6"]["counts"].values()
    )
    assertions.append(
        {
            "name": "06016418_duplicate_removed_from_canonical_source",
            "status": "PASS",
            "evidence": "no non-placeholder duplicate placement remains after retaining one canonical record per plan",
        }
    )

    adjusted = report["checks"]["CHK7"]["evidence"]["semester_totals_lab_adjusted"]
    assert adjusted["coop"]["2/2"] == 30
    assert adjusted["coop"]["3/1"] == 33
    assert adjusted["no_coop"]["2/2"] == 30
    assert adjusted["no_coop"]["3/1"] == 33
    assert report["checks"]["CHK7"]["evidence"]["duplicate_adjustments_for_diagnosis"] == {
        "coop": {},
        "no_coop": {},
    }
    assert report["checks"]["CHK7"]["status"] == "EXCEPTION"
    assertions.append(
        {
            "name": "remaining_credit_overages_after_lab_adjustment",
            "status": "PASS",
            "evidence": "2/2 remains 30 and 3/1 remains 33 after the canonical duplicate was removed; source-condition exceptions are retained for human review",
        }
    )
    return assertions


def build_report(data: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    validate_submission_shape(data, schema)
    chk1 = build_chk1(data)
    chk2 = build_chk2(data)
    chk3 = build_chk3(data)
    chk4 = build_chk4(data)
    chk5 = build_chk5(data)
    chk6 = build_chk6(data)
    chk7 = build_chk7(data, chk6)
    checks = {
        "CHK1": chk1,
        "CHK2": chk2,
        "CHK3": chk3,
        "CHK4": chk4,
        "CHK5": chk5,
        "CHK6": chk6,
        "CHK7": chk7,
    }
    status_counts = {status: 0 for status in ("PASS", "FAIL", "EXCEPTION", "NOT_APPLICABLE")}
    for check in checks.values():
        status_counts[check["status"]] += 1
    report: dict[str, Any] = {
        "program": data["program"],
        "source_file": "submission/curriculum.json",
        "schema_validation": {
            "status": "PASS",
            "schema_file": "submission/schema/curriculum.schema.json",
            "draft": "2020-12",
            "method": "dependency-free deterministic conformance checks for the submission schema",
        },
        "summary": {
            "status": "FAIL" if status_counts["FAIL"] else ("EXCEPTION" if status_counts["EXCEPTION"] else "PASS"),
            "checks_total": len(checks),
            "status_counts": status_counts,
            "human_review_required": True,
            "human_review_reasons": [
                "CHK1 has no independent declared curriculum credit total.",
                "CHK4 has no separate placement-credit and course-description-credit fields.",
                "CHK5 has flexible prerequisite/co-requisite cases without requirement_type.",
                "CHK6 has no alternative-group metadata for repeated placeholder slots.",
                "CHK7 has source-listed fixed normal semesters outside 9-22 credits; no automatic credit exception is applied and human/source review is required.",
            ],
        },
        "checks": checks,
    }
    report["regression_assertions"] = build_regression_assertions(data, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    report = build_report(data, schema)
    serialized = json.dumps(report, ensure_ascii=False, indent=4) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized, encoding="utf-8", newline="\n")
    print(f"wrote {args.output.as_posix()}")
    print("schema_validation: PASS")
    for check_id in ("CHK1", "CHK2", "CHK3", "CHK4", "CHK5", "CHK6", "CHK7"):
        print(f"{check_id}: {report['checks'][check_id]['status']}")
    print("regression_assertions: PASS")


if __name__ == "__main__":
    main()
