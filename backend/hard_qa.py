"""Narrow orchestration and deterministic composition for Hard QA tasks."""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, Callable

from backend.hard_plan_compare import compare_plan_course_sets
from backend.hard_plan_validate import validate_curriculum_plan_structure
from backend.hard_prerequisite_validate import validate_plan_prerequisite_sequence
from backend.hard_sequence_planner import plan_curriculum_sequence
from rag.query_spec import parse_query_spec


_TASK_TYPES = frozenset({
    "none",
    "plan_comparison",
    "plan_structure_validation",
    "prerequisite_sequence",
    "seven_term_plan",
})
_INTERPRETATION_KEYS = frozenset({
    "task_type", "program", "plan", "left_plan", "right_plan",
    "target_course_code", "horizon_terms",
})
HARD_INTERPRETATION_RESPONSE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "task_type": {
            "type": "string",
            "enum": sorted(_TASK_TYPES),
        },
        "program": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "plan": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "left_plan": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "right_plan": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "target_course_code": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "horizon_terms": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
    },
    "required": sorted(_INTERPRETATION_KEYS),
}
_HORIZON_TERMS = 7
_logger = logging.getLogger(__name__)


class _InterpretationParseError(ValueError):
    def __init__(self, reason: str, *, metadata: dict[str, Any] | None = None,
                 exception_class: str | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.metadata = metadata or {}
        self.exception_class = exception_class or type(self).__name__


def _read_context(context: Any) -> dict[str, str]:
    if context is None:
        return {}
    if not isinstance(context, dict) or set(context) - {"program", "plan", "course_code", "catalog_key"}:
        raise ValueError("Hard QA accepts only validated catalog, program, plan, and course_code context")
    normalized = {}
    for key, value in context.items():
        if value is not None:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{key} must be a non-empty string")
            normalized[key] = value.strip()
    if "catalog_key" in normalized and len(normalized["catalog_key"]) > 128:
        raise ValueError("catalog_key must be at most 128 characters")
    return normalized


def _load_scopes(
    db_path: str | Path, catalog_key: str | None = None
) -> list[dict[str, str]]:
    path = Path(db_path).resolve()
    if not path.is_file():
        raise OSError("canonical database is unavailable")
    with closing(sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """SELECT DISTINCT cp.program_code, p.program_code_normalized,
                      cp.plan_key, c.catalog_key
               FROM curriculum_plans cp
               JOIN programs p ON p.program_id=cp.program_id AND p.catalog_id=cp.catalog_id
               JOIN catalogs c ON c.catalog_id=cp.catalog_id
               WHERE (? IS NULL OR LOWER(TRIM(c.catalog_key))=LOWER(TRIM(?)))
               ORDER BY cp.program_code, cp.plan_key""",
            (catalog_key, catalog_key),
        ).fetchall()
    return [
        {
            "program": str(row["program_code"]),
            "program_normalized": str(row["program_code_normalized"]),
            "plan": str(row["plan_key"]),
            "catalog_key": str(row["catalog_key"]),
        }
        for row in rows
    ]


def _same(left: str, right: str) -> bool:
    return left.strip().casefold() == right.strip().casefold()


def _mentioned(question: str, token: str) -> bool:
    return re.search(
        rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])",
        question,
        flags=re.IGNORECASE,
    ) is not None


def _mentioned_values(question: str, scopes: list[dict[str, str]], key: str) -> list[str]:
    values = sorted({item[key] for item in scopes}, key=lambda value: (-len(value), value.casefold()))
    found = []
    for value in values:
        if _mentioned(question, value) and not any(_same(value, prior) for prior in found):
            found.append(value)
    return found


def _old_new_question(question: str) -> bool:
    folded = question.casefold()
    return "หลักสูตร" in folded and any(
        marker in folded for marker in ("เก่า", "ใหม่", "เดิม", "ปัจจุบัน")
    )


def _hard_candidate(question: str) -> bool:
    folded = question.casefold()
    if "prerequisite" in parse_query_spec(question).operations:
        return True
    if _old_new_question(question):
        return True
    if any(marker in folded for marker in ("ต่างกัน", "เปรียบเทียบ", "difference")) and any(
        marker in folded for marker in ("coop", "no_coop", "default", "gened")
    ):
        return True
    if any(marker in folded for marker in ("โครงสร้างครบ", "ครบตามหลักสูตร", "ตรวจสอบโครงสร้าง")):
        return True
    if "prerequisite" in folded or "วิชาบังคับก่อน" in folded or "ต้องเรียนอะไรมาก่อน" in folded:
        return True
    return any(marker in folded for marker in ("3.5", "สามปีครึ่ง", "3 ปีครึ่ง"))


def _interpretation_prompt(
    question: str,
    scopes: list[dict[str, str]],
    context: dict[str, str],
) -> str:
    allowed_scopes = sorted(
        {(item["program"], item["plan"]) for item in scopes},
        key=lambda item: (item[0].casefold(), item[1].casefold()),
    )
    payload = {
        "question": question,
        "trusted_context": context,
        "canonical_program_plan_pairs": [
            {"program": program, "plan": plan} for program, plan in allowed_scopes
        ],
    }
    return (
        "Interpret whether the current question requests one supported Hard curriculum task. "
        "Return exactly one JSON object with keys: task_type, program, plan, left_plan, "
        "right_plan, target_course_code, horizon_terms. task_type must be one of "
        "plan_comparison, plan_structure_validation, prerequisite_sequence, "
        "seven_term_plan, none. Use null for absent values. Extract entities only when "
        "explicitly present in the current question; trusted_context may fill a missing "
        "program, plan, or exact course_code only when applicable. A program never implies "
        "a plan. A plan-comparison requires both explicitly named plans. Never map old/new "
        "curriculum to coop/no_coop. seven_term_plan is only the 3.5-year/seven-regular-term "
        "request, with horizon_terms=7. Do not answer the factual question.\nINPUT JSON:\n"
        + json.dumps(payload, ensure_ascii=False)
    )


def _parse_interpretation(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, str):
        raise _InterpretationParseError(
            "invalid_response_type", metadata={"response_type": type(raw).__name__}
        )
    raw_metadata = {
        "response_length": len(raw),
        "contains_code_fence": "```" in raw,
    }
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise _InterpretationParseError(
            "json_decode_error",
            metadata=raw_metadata,
            exception_class=type(exc).__name__,
        ) from exc
    if not isinstance(data, dict):
        raise _InterpretationParseError(
            "root_not_object",
            metadata={**raw_metadata, "root_type": type(data).__name__},
        )
    observed_keys = set(data)
    missing_keys = sorted(_INTERPRETATION_KEYS - observed_keys)
    unexpected_keys = observed_keys - _INTERPRETATION_KEYS
    if missing_keys or unexpected_keys:
        if missing_keys and unexpected_keys:
            reason = "key_set_mismatch"
        elif missing_keys:
            reason = "missing_keys"
        else:
            reason = "unexpected_keys"
        raise _InterpretationParseError(
            reason,
            metadata={
                **raw_metadata,
                "missing_keys": missing_keys,
                "unexpected_key_count": len(unexpected_keys),
            },
        )
    task_type = data["task_type"]
    if not isinstance(task_type, str) or task_type not in _TASK_TYPES:
        raise _InterpretationParseError(
            "invalid_task_type",
            metadata={**raw_metadata, "task_type_type": type(task_type).__name__},
        )
    for key in sorted(_INTERPRETATION_KEYS - {"task_type", "horizon_terms"}):
        value = data[key]
        if value is not None and not isinstance(value, str):
            raise _InterpretationParseError(
                "invalid_field_type",
                metadata={**raw_metadata, "field": key, "field_type": type(value).__name__},
            )
        if isinstance(value, str) and not value.strip():
            raise _InterpretationParseError(
                "invalid_field_value",
                metadata={**raw_metadata, "field": key, "field_type": "str"},
            )
    horizon = data["horizon_terms"]
    if horizon is not None and (isinstance(horizon, bool) or not isinstance(horizon, int)):
        raise _InterpretationParseError(
            "invalid_horizon_terms",
            metadata={**raw_metadata, "horizon_type": type(horizon).__name__},
        )
    return data


def _canonical_match(value: Any, candidates: list[str]) -> str | None:
    if not isinstance(value, str):
        return None
    return next((candidate for candidate in candidates if _same(value, candidate)), None)


def _normal_provenance(groups: list[Any]) -> list[dict[str, Any]]:
    by_id: dict[int, dict[str, Any]] = {}
    for group in groups:
        if not isinstance(group, list):
            continue
        for reference in group:
            if not isinstance(reference, dict):
                continue
            provenance_id = reference.get("provenance_id")
            if isinstance(provenance_id, bool) or not isinstance(provenance_id, int):
                continue
            by_id.setdefault(provenance_id, {
                "provenance_id": provenance_id,
                "program": reference.get("program"),
                "source_filename": reference.get("source_filename"),
                "source_page": reference.get("source_page"),
                "document_page": reference.get("document_page"),
                "document_category": reference.get("document_category"),
            })
    return [by_id[key] for key in sorted(by_id)]


def _course_lines(courses: Any) -> list[str]:
    if not isinstance(courses, list):
        return []
    result = []
    for course in courses:
        if not isinstance(course, dict):
            continue
        code = course.get("course_code")
        if isinstance(code, str):
            name = course.get("name_th") or course.get("name_en")
            if re.fullmatch(r"(?:\d{4}x{4}|\d{5}x{3}|x{8})", code, flags=re.IGNORECASE):
                result.append(name if isinstance(name, str) and name else "รายการวิชาที่ยังไม่มีรหัสระบุแน่นอน")
            else:
                result.append(f"{code}" + (f" — {name}" if isinstance(name, str) and name else ""))
    return result


def _citation_summary(provenance: list[dict[str, Any]]) -> str:
    groups: dict[tuple[str, str, str], set[int]] = {}
    labels: dict[tuple[str, str, str], str] = {}
    category_labels = {
        "plan": "แผน",
        "description": "รายละเอียดรายวิชา",
        "program_requirement": "ข้อกำหนดหลักสูตร",
        "rule": "ข้อกำหนด",
    }
    for reference in provenance:
        if not isinstance(reference, dict):
            continue
        page = reference.get("source_page")
        if isinstance(page, bool) or not isinstance(page, int) or page < 1:
            continue
        program = str(reference.get("program") or "").strip()
        category = str(reference.get("document_category") or "").strip()
        filename = str(reference.get("source_filename") or "").strip()
        page_file = re.match(r"^(?P<document>.+?)_page_\d+(?:_ocr)?\.[^.]+$", filename, re.IGNORECASE)
        document = page_file.group("document") if page_file else (filename or f"provenance-{reference.get('provenance_id')}")
        key = (program.casefold(), category.casefold(), document.casefold())
        groups.setdefault(key, set()).add(page)
        display_program = "ข้อกำหนด" if program.casefold() == "rule" else program
        if page_file and _same(document, program):
            document_label = display_program or document
        else:
            display_document = "" if document.casefold() == program.casefold() else document
            document_label = " · ".join(item for item in (display_program, display_document) if item)
        category_label = category_labels.get(category.casefold(), "เอกสารอ้างอิง" if category else "")
        labels[key] = " ".join(item for item in (document_label, category_label) if item) or "เอกสารอ้างอิง"

    compacted = []
    for key in sorted(groups):
        pages = sorted(groups[key])
        ranges = []
        start = previous = pages[0]
        for page in pages[1:]:
            if page == previous + 1:
                previous = page
                continue
            ranges.append(str(start) if start == previous else f"{start}–{previous}")
            start = previous = page
        ranges.append(str(start) if start == previous else f"{start}–{previous}")
        compacted.append(f"{labels[key]} หน้า {', '.join(ranges)}")
    return "อ้างอิง: " + " · ".join(compacted) if compacted else ""


def _append_citation_summary(answer: str, provenance: list[dict[str, Any]]) -> str:
    citation = _citation_summary(provenance)
    return f"{answer}\n{citation}" if citation else answer


def _format_h1(result: dict[str, Any]) -> tuple[str, str, list[dict[str, Any]]]:
    status = result.get("status")
    provenance = _normal_provenance([
        result.get("left_plan_evidence"), result.get("right_plan_evidence"),
        *[course.get("provenance", []) for key in ("left_only_courses", "right_only_courses") for course in result.get(key, []) if isinstance(course, dict)],
    ])
    if status != "complete":
        return "incomplete_evidence", "เปรียบเทียบชุดรหัสวิชาไม่ได้อย่างสมบูรณ์จากหลักฐานที่มี", provenance
    left = result.get("left_only_courses", [])
    right = result.get("right_only_courses", [])
    if not left and not right:
        answer = f"ไม่พบความแตกต่างของชุดรหัสวิชาเชิงตรรกะระหว่างแผน {result.get('left_plan')} และ {result.get('right_plan')} ข้อนี้ไม่ได้ยืนยันว่ารายละเอียดด้านอื่นของแผนเหมือนกัน"
    else:
        parts = [f"มีเฉพาะในแผน {result.get('left_plan')}: " + (", ".join(_course_lines(left)) or "ไม่มี")]
        parts.append(f"มีเฉพาะในแผน {result.get('right_plan')}: " + (", ".join(_course_lines(right)) or "ไม่มี"))
        answer = "\n".join(parts)
    return "answer", answer, provenance


def _format_h2(result: dict[str, Any]) -> tuple[str, str, list[dict[str, Any]]]:
    raw_status = result.get("status")
    checks = [item for item in result.get("checks", []) if isinstance(item, dict) and item.get("check") in {
        "mandatory_placements", "alternative_groups", "program_credit_requirements", "placement_quality"
    }]
    has_deficit = any(item.get("status") == "deficit" for item in checks)
    status = "satisfied" if raw_status == "complete" else (
        "deficit" if raw_status == "deficit" or (has_deficit and raw_status != "incomplete_evidence")
        else "incomplete_evidence"
    )
    labels = {
        "mandatory_placements": "วิชาบังคับและตำแหน่งในแผน",
        "alternative_groups": "กลุ่มวิชาเลือกแทน",
        "program_credit_requirements": "หน่วยกิตตามข้อกำหนดหลักสูตร",
        "placement_quality": "ข้อมูลภาคเรียนของรายวิชา",
    }
    lines = ["จากข้อมูลที่ยืนยันได้:"]
    for check in checks:
        state = check.get("status")
        phrase = "ยืนยันได้" if state in {"complete", "satisfied"} else "ข้อมูลยังไม่ครบ"
        lines.append(f"{labels[check['check']]}: {phrase}")
    if not lines:
        lines.append("ไม่มีผลตรวจโครงสร้างที่สรุปได้จากข้อมูล")
    if status == "incomplete_evidence":
        lines.append("ยังมีข้อมูลที่ยืนยันไม่ครบ จึงไม่สรุปว่าผ่านหรือไม่ผ่านทั้งหมด")
    elif status == "deficit":
        lines.append("พบข้อขาดตามผลตรวจโครงสร้างที่ยืนยันได้")
    else:
        lines.append("ผลตรวจโครงสร้างที่แทนได้ครบตามหลักฐาน")
    groups = [
        result.get("provenance"),
        *[course.get("provenance", []) for course in result.get("mandatory_courses", {}).get("courses", []) if isinstance(course, dict)],
        *[group.get("provenance", []) for group in result.get("alternative_groups", []) if isinstance(group, dict)],
        *[row.get("provenance", []) for row in result.get("credit_requirements", []) if isinstance(row, dict)],
    ]
    return status, "\n".join(lines), _normal_provenance(groups)


def _target_relations(result: dict[str, Any], target: str) -> list[dict[str, Any]]:
    relationships = [item for item in result.get("relationships", []) if isinstance(item, dict)]
    relevant_codes = {target.upper()}
    changed = True
    while changed:
        changed = False
        for relationship in relationships:
            dependent = str(relationship.get("dependent_course_code") or "").upper()
            if dependent not in relevant_codes:
                continue
            condition = relationship.get("prerequisite_condition", {})
            candidates = []
            if condition.get("kind") == "direct" and condition.get("course_code"):
                candidates.append(condition["course_code"])
            elif condition.get("kind") == "alternative_group":
                candidates.extend(condition.get("candidate_course_codes", []))
            for term in relationship.get("prerequisite_terms", []):
                if term.get("course_code"):
                    candidates.append(term["course_code"])
            before = len(relevant_codes)
            relevant_codes.update(str(code).upper() for code in candidates if isinstance(code, str))
            changed = changed or len(relevant_codes) != before
    return [
        relationship for relationship in relationships
        if str(relationship.get("dependent_course_code") or "").upper() in relevant_codes
    ]


def _format_h3(result: dict[str, Any], target: str | None) -> tuple[str, str, list[dict[str, Any]]]:
    if target is None:
        status = result.get("status", "incomplete_evidence")
        summary = result.get("summary", {})
        if status == "satisfied":
            answer = f"ความสัมพันธ์ prerequisite ที่ตรวจสอบได้ในแผน {result.get('program')} {result.get('plan')} เรียงตามลำดับที่กำหนด"
            return status, answer, _normal_provenance([result.get("provenance"), *[rel.get("provenance", []) for rel in result.get("relationships", []) if isinstance(rel, dict)]])
        selected = [rel for rel in result.get("relationships", []) if isinstance(rel, dict) and rel.get("status") in {"violation", "incomplete_evidence"}]
        if not selected:
            return "incomplete_evidence", "ข้อมูล prerequisite ของแผนยังไม่ครบพอจะยืนยันการเรียงทั้งหมด", _normal_provenance([result.get("provenance")])
    else:
        selected = _target_relations(result, target)
        if not selected and result.get("status") == "satisfied":
            return "satisfied", f"ไม่พบ prerequisite ที่แทนไว้สำหรับวิชา {target} ในขอบเขตแผนที่ตรวจสอบ", _normal_provenance([result.get("provenance")])
    if target is not None and not selected:
        return "incomplete_evidence", f"ข้อมูล prerequisite ของวิชา {target} ยังยืนยันไม่ได้", []
    relevant_statuses = {item.get("status") for item in selected}
    status = "violation" if "violation" in relevant_statuses else ("incomplete_evidence" if "incomplete_evidence" in relevant_statuses else "satisfied")
    if target is None:
        counts = result.get("summary", {})
        violations = counts.get("violations") if isinstance(counts, dict) else None
        incomplete = counts.get("incomplete") if isinstance(counts, dict) else None
        if not isinstance(violations, int):
            violations = sum(item.get("status") == "violation" for item in selected)
        if not isinstance(incomplete, int):
            incomplete = sum(item.get("status") == "incomplete_evidence" for item in selected)
        lines = [f"แผน {result.get('program')} {result.get('plan')}: ตรวจสอบลำดับ prerequisite ที่มีข้อมูลโครงสร้างได้"]
        if violations:
            lines.append(f"พบ {violations} ความสัมพันธ์ที่ไม่เป็นไปตามลำดับที่กำหนด")
        if incomplete:
            lines.append(f"อีก {incomplete} ความสัมพันธ์มีข้อมูลไม่เพียงพอที่จะยืนยัน")
    elif status == "incomplete_evidence":
        lines = [f"ข้อมูลหลักสูตรที่มีอยู่ยังไม่เพียงพอที่จะยืนยัน prerequisite ของวิชา {target} ได้ครบถ้วน"]
    else:
        lines = []
        for relationship in selected:
            dependent = relationship.get("dependent_course_code")
            condition = relationship.get("prerequisite_condition", {})
            candidates = [item for item in relationship.get("prerequisite_terms", []) if isinstance(item, dict)]
            if condition.get("kind") == "direct":
                source = condition.get("course_code")
                if not source:
                    continue
                lines.append(f"วิชา {dependent} ต้องเรียนวิชา {source} มาก่อน")
                source_term = next((item.get("term") for item in candidates if item.get("course_code") == source), None)
                dependent_term = relationship.get("course_term")
                if source_term and dependent_term:
                    source_position = _term_phrase(source_term)
                    dependent_position = _term_phrase(dependent_term)
                    if source_position and dependent_position:
                        if relationship.get("status") == "satisfied":
                            lines.append(
                                f"ตามแผน {result.get('program')} {result.get('plan')} วิชา {source} อยู่{source_position} "
                                f"ก่อนวิชา {dependent} ซึ่งอยู่{dependent_position}"
                            )
                        else:
                            lines.append(
                                f"ตามแผน {result.get('program')} {result.get('plan')} วิชา {source} อยู่{source_position} "
                                f"และวิชา {dependent} อยู่{dependent_position}"
                            )
            elif condition.get("kind") == "alternative_group":
                minimum = condition.get("minimum_choices", 1)
                codes = condition.get("candidate_course_codes", [])
                if not isinstance(codes, list):
                    codes = []
                lines.append(f"ก่อนเรียนวิชา {dependent} ต้องเลือกอย่างน้อย {minimum} วิชาจาก {' หรือ '.join(codes)}")
            else:
                lines.append(f"ข้อมูล prerequisite ของวิชา {dependent} ยังยืนยันไม่ได้ครบถ้วน")
        if status == "violation":
            lines.append("ผลตรวจพบความสัมพันธ์ที่ไม่เป็นไปตามลำดับ prerequisite ที่กำหนด")
        elif lines:
            lines.append("จึงเป็นไปตามลำดับ prerequisite ตามแผนที่ตรวจสอบ")
        else:
            lines.append(f"ไม่พบ prerequisite ที่แทนไว้สำหรับวิชา {target} ในขอบเขตแผนที่ตรวจสอบ")
    provenance = _normal_provenance([rel.get("provenance", []) for rel in selected])
    return status, "\n".join(lines), provenance


def _term_phrase(term: Any) -> str | None:
    if not isinstance(term, dict):
        return None
    year = term.get("year")
    semester = term.get("semester")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in (year, semester)):
        return None
    return f"ปี {year} เทอม {semester}"


def _format_h4(result: dict[str, Any]) -> tuple[str, str, list[dict[str, Any]]]:
    feasible = result.get("sequence_feasible")
    status = result.get("status", "incomplete_evidence")
    terms = result.get("terms", [])
    lines = []
    if status == "incomplete_evidence" and feasible is False:
        lines.append("จากข้อจำกัดหลักสูตรที่ตรวจสอบได้ ยังจัดรายวิชาลงครบ 7 เทอมไม่ได้ และข้อมูลยังไม่พอยืนยันความครบถ้วนทั้งหมด:")
    elif status == "incomplete_evidence" and not terms:
        lines.append("หลักฐานที่มีไม่เพียงพอสำหรับจัดทำโครงร่าง 7 เทอม จึงยังระบุลำดับรายวิชาไม่ได้:")
    elif status == "incomplete_evidence":
        lines.append("จากข้อมูลหลักสูตรที่มี สามารถจัดลำดับรายวิชาเป็นโครงร่าง 7 เทอมได้ดังนี้ แต่ข้อมูลยังไม่เพียงพอที่จะยืนยันว่าแผนนี้ครบเงื่อนไขจบทั้งหมด:")
    elif feasible is True:
        lines.append("ลำดับรายวิชาที่แทนได้ใน 7 ภาคเรียน:")
    elif feasible is False:
        lines.append("หลักฐานที่แทนได้ชี้ว่าโครงสร้างนี้จัดลง 7 ภาคเรียนภายใต้เพดานปกติไม่ได้")
    else:
        lines.append("จากข้อมูลหลักสูตรที่มี สามารถจัดลำดับรายวิชาเป็นโครงร่าง 7 เทอมได้ดังนี้:")
    for term in terms:
        if not isinstance(term, dict):
            continue
        index = term.get("term_index")
        year = term.get("year")
        semester = term.get("semester")
        label = f"ปี {year} เทอม {semester}" if year and semester else f"เทอม {index}"
        courses = _h4_course_lines(term.get("courses", []))
        slots = []
        for slot in term.get("choice_slots", []):
            if isinstance(slot, dict):
                choices = _course_lines(slot.get("candidates", []))
                if choices:
                    minimum = slot.get("minimum_choices")
                    if isinstance(minimum, int) and not isinstance(minimum, bool) and minimum > 0:
                        slots.append(f"ทางเลือก: เลือก {minimum} วิชาจาก {' หรือ '.join(choices)}")
                    else:
                        slots.append(f"ทางเลือก: {' หรือ '.join(choices)}")
        lines.append(f"{label}:")
        for course in courses + slots:
            lines.append(f"- {course}")
        if not courses and not slots:
            lines.append("- ไม่มีรายวิชาที่แทนได้")
    relevant_limitations = [item.casefold() for item in result.get("limitations", []) if isinstance(item, str)]
    concise_limitations = []
    if status == "incomplete_evidence":
        if any(any(marker in item for marker in ("credit", "placement", "term", "semester")) for item in relevant_limitations):
            concise_limitations.append("บางรายวิชายังไม่มีข้อมูลหน่วยกิตหรือภาคเรียนที่ระบุแน่นอน")
        if any("prerequisite" in item for item in relevant_limitations):
            concise_limitations.append("ข้อมูล prerequisite บางส่วนยังยืนยันได้ไม่ครบ")
    if terms and result.get("actual_course_offering_unverified") is True:
        concise_limitations.append("ยังไม่ได้ตรวจสอบการเปิดสอนจริงในแต่ละภาคเรียน")
    if concise_limitations:
        lines.append("หมายเหตุ: " + "; ".join(concise_limitations))
    return status, "\n".join(lines), _normal_provenance([result.get("evidence")])


def _h4_course_lines(courses: Any) -> list[str]:
    if not isinstance(courses, list):
        return []
    result = []
    for course in courses:
        if not isinstance(course, dict):
            continue
        rendered = _course_lines([course])
        code = course.get("course_code")
        name = course.get("name_th") or course.get("name_en")
        is_placeholder = isinstance(code, str) and bool(
            re.fullmatch(r"(?:\d{4}x{4}|\d{5}x{3}|x{8})", code, flags=re.IGNORECASE)
        )
        if is_placeholder and isinstance(name, str) and name.count("วิชาเลือกกลุ่ม") > 1:
            labels = [part.strip() for part in re.split(r"\s+(?=วิชาเลือกกลุ่ม)", name) if part.strip()]
            result.extend(label.replace("วิชาเลือกกลุ่ม ", "วิชาเลือกกลุ่ม", 1) for label in labels)
        else:
            result.extend(rendered)
    return result


def _response(
    status: str,
    answer: str,
    task_type: str,
    scope: dict[str, Any],
    provenance: list[dict[str, Any]] | None = None,
    *,
    action: str | None = None,
) -> dict[str, Any]:
    next_context = {key: value for key, value in scope.items() if key in {"catalog_key", "program", "plan", "target_course_code"} and value is not None}
    public_scope = {
        key: value for key, value in scope.items()
        if key != "catalog_key" or value is not None
    }
    if "target_course_code" in next_context:
        next_context["course_code"] = next_context.pop("target_course_code")
    return {
        "status": status,
        "answer": answer,
        "action": action,
        "route": "hard",
        "hard_task_type": task_type,
        "scope": public_scope,
        "provenance": provenance or [],
        "next_context": next_context or None,
    }


def answer_hard_question(
    db_path: str | Path,
    question: str,
    conversation_context: dict[str, Any] | None,
    interpretation_model: Callable[[str], str],
) -> dict[str, Any] | None:
    """Route a narrowly eligible Hard request to H1-H4 and compose a grounded answer.

    Returns ``None`` for ordinary Easy/Medium questions so their existing path is
    unchanged. The injected model interprets intent only; all facts and statuses
    come from deterministic operations.
    """
    if not isinstance(question, str) or not question.strip():
        return None
    question = question.strip()
    if not _hard_candidate(question):
        return None
    try:
        context = _read_context(conversation_context)
        catalog_key = context.get("catalog_key")
        if catalog_key is not None:
            with closing(sqlite3.connect(f"{Path(db_path).resolve().as_uri()}?mode=ro", uri=True)) as connection:
                catalogs = connection.execute(
                    "SELECT catalog_key FROM catalogs WHERE LOWER(TRIM(catalog_key))=LOWER(TRIM(?))",
                    (catalog_key,),
                ).fetchall()
            if len(catalogs) != 1 or not catalogs[0][0]:
                return _response(
                    "context_conflict",
                    "catalog_key ไม่ได้ระบุหลักสูตรฉบับที่มีอยู่เพียงหนึ่งรายการ",
                    "scope_resolution",
                    {"catalog_key": catalog_key},
                    action="invalid_catalog_scope",
                )
            catalog_key = str(catalogs[0][0]).strip()
        scopes = _load_scopes(db_path, catalog_key)
    except (OSError, sqlite3.Error, ValueError):
        return _response("incomplete_evidence", "ไม่สามารถอ่านขอบเขตหลักสูตรที่ยืนยันได้", "scope_resolution", {}, action="canonical_scope_unavailable")

    programs = sorted({item["program"] for item in scopes}, key=lambda value: value.casefold())
    context_program = _canonical_match(context.get("program"), programs)
    if catalog_key is not None and context.get("program") is not None and context_program is None:
        return _response(
            "context_conflict",
            "หลักสูตรที่เลือกไม่มีโปรแกรมตามบริบท กรุณาเลือกหลักสูตรฉบับที่ตรงกัน",
            "scope_resolution",
            {"catalog_key": catalog_key},
            action="catalog_program_mismatch",
        )
    context_plan = _canonical_match(
        context.get("plan"),
        [item["plan"] for item in scopes if context_program is not None and _same(item["program"], context_program)],
    )

    if _old_new_question(question):
        return _response(
            "unsupported",
            "ฐานข้อมูลที่มีไม่ระบุปีหรือรุ่นหลักสูตรเพียงพอสำหรับเปรียบเทียบหลักสูตรเก่ากับใหม่",
            "unsupported_old_new",
            {"catalog_key": catalog_key, "program": context_program, "plan": context_plan},
        )

    raw = interpretation_model(_interpretation_prompt(question, scopes, context))
    try:
        intent = _parse_interpretation(raw)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        reason = getattr(exc, "reason", "unclassified_parser_error")
        metadata = getattr(exc, "metadata", {})
        exception_class = getattr(exc, "exception_class", type(exc).__name__)
        _logger.warning(
            "Hard interpreter parse rejected stage=%s reason=%s exception_class=%s metadata=%s",
            "hard_interpreter_parse",
            reason,
            exception_class,
            metadata,
            extra={
                "stage": "hard_interpreter_parse",
                "reason": reason,
                "exception_class": exception_class,
                "metadata": metadata,
            },
        )
        return _response("error", "ยังจัดประเภทคำถาม Hard นี้ไม่ได้อย่างปลอดภัย กรุณาระบุคำถามใหม่ให้ชัดเจน", "interpretation", {}, action="hard_interpretation_failure")
    if intent["task_type"] == "none":
        return None

    explicit_programs = _mentioned_values(question, scopes, "program")
    proposed_program = _canonical_match(intent.get("program"), programs)
    if len(explicit_programs) > 1:
        return _response("clarification_required", "โปรดระบุหลักสูตรที่จะใช้ตรวจสอบเพียงหลักสูตรเดียว", intent["task_type"], {})
    if explicit_programs:
        program = explicit_programs[0]
        if proposed_program is not None and not _same(proposed_program, program):
            return _response("clarification_required", "โปรดตรวจสอบชื่อหลักสูตรที่ต้องการอีกครั้ง", intent["task_type"], {"program": program, "plan": None})
    elif context_program is not None:
        program = context_program
        if proposed_program is not None and not _same(proposed_program, program):
            return _response("clarification_required", "หลักสูตรในคำถามขัดกับหลักสูตรที่เลือกไว้ กรุณายืนยันหลักสูตร", intent["task_type"], {})
    else:
        program = None
    if program is None:
        return _response("clarification_required", "ต้องการตรวจสอบหลักสูตรใด? โปรดระบุรหัสหลักสูตร เช่น IT หรือ DSBA", intent["task_type"], {})

    program_catalogs = {
        item["catalog_key"] for item in scopes if _same(item["program"], program)
    }
    if catalog_key is None and len(program_catalogs) > 1:
        return _response(
            "clarification_required",
            "โปรดเลือกฉบับหลักสูตรหรือ academic_year ก่อนตรวจสอบแผน",
            intent["task_type"],
            {"program": program},
        )

    plans_for_program = sorted({item["plan"] for item in scopes if _same(item["program"], program)}, key=lambda value: value.casefold())
    explicit_plans = _mentioned_values(question, [item for item in scopes if _same(item["program"], program)], "plan")
    task_type = intent["task_type"]
    if task_type == "plan_comparison":
        left_plan = _canonical_match(intent.get("left_plan"), plans_for_program)
        right_plan = _canonical_match(intent.get("right_plan"), plans_for_program)
        if (left_plan is None or right_plan is None or _same(left_plan, right_plan)
                or not _mentioned(question, left_plan) or not _mentioned(question, right_plan)):
            return _response("clarification_required", "โปรดระบุชื่อแผนทั้งสองแผนที่ต้องการเปรียบเทียบ", task_type, {"catalog_key": catalog_key, "program": program, "plan": None})
        context_plan = _canonical_match(context.get("plan"), plans_for_program) if context_program is not None and _same(program, context_program) else None
        scope = {"catalog_key": catalog_key, "program": program, "plan": context_plan, "left_plan": left_plan, "right_plan": right_plan}
        hard_result = compare_plan_course_sets(
            db_path, program, left_plan, right_plan, catalog_key
        )
        status, answer, provenance = _format_h1(hard_result)
    else:
        explicit_program_changed = context_program is not None and not _same(program, context_program)
        context_plan = _canonical_match(context.get("plan"), plans_for_program) if not explicit_program_changed else None
        proposed_plan = _canonical_match(intent.get("plan"), plans_for_program)
        if explicit_plans:
            plan = explicit_plans[0] if len(explicit_plans) == 1 else None
            if len(explicit_plans) > 1:
                return _response("clarification_required", "โปรดเลือกแผนเดียวสำหรับการตรวจสอบนี้", task_type, {"catalog_key": catalog_key, "program": program, "plan": None})
            if proposed_plan is not None and not _same(proposed_plan, plan):
                return _response("clarification_required", "โปรดตรวจสอบชื่อแผนที่ต้องการอีกครั้ง", task_type, {"catalog_key": catalog_key, "program": program, "plan": None})
        elif proposed_plan is not None:
            if context_plan is not None and _same(proposed_plan, context_plan):
                plan = context_plan
            else:
                return _response("clarification_required", "ไม่สามารถใช้แผนที่ไม่ได้ระบุในคำถามหรือบริบทที่เลือกไว้ได้ โปรดระบุแผน", task_type, {"catalog_key": catalog_key, "program": program, "plan": None})
        else:
            plan = context_plan
        if plan is None:
            return _response("clarification_required", "โปรดเลือกแผนหลักสูตรที่ต้องการตรวจสอบ เช่น coop หรือ no_coop", task_type, {"catalog_key": catalog_key, "program": program, "plan": None})
        if task_type == "prerequisite_sequence":
            current_codes = list(dict.fromkeys(
                str(code).strip().upper() for code in parse_query_spec(question).course_codes
            ))
            if len(current_codes) > 1:
                return _response("clarification_required", "โปรดระบุวิชาเป้าหมายเพียงหนึ่งวิชา", task_type, {"program": program, "plan": plan})
            target = current_codes[0] if current_codes else intent.get("target_course_code")
            context_course = context.get("course_code")
            if not current_codes and target is None and context_course is not None and not explicit_program_changed:
                target = context_course
            if target is not None:
                target = target.strip().upper()
                if not _mentioned(question, target) and not _same(target, context_course or ""):
                    return _response("clarification_required", "โปรดระบุรหัสวิชาที่ต้องการตรวจสอบ prerequisite", task_type, {"program": program, "plan": plan})
            hard_result = validate_plan_prerequisite_sequence(
                db_path, program, plan, catalog_key
            )
            if target is not None:
                with closing(sqlite3.connect(f"{Path(db_path).resolve().as_uri()}?mode=ro", uri=True)) as connection:
                    exists = connection.execute(
                        """SELECT 1 FROM curriculum_plans cp
                           JOIN plan_placements pp ON pp.plan_id=cp.plan_id
                           LEFT JOIN courses c ON c.course_id=pp.course_id
                           LEFT JOIN alternative_course_group_members gm ON gm.alternative_group_id=pp.alternative_group_id
                           LEFT JOIN courses mc ON mc.course_id=gm.course_id
                           JOIN catalogs cat ON cat.catalog_id=cp.catalog_id
                           WHERE UPPER(cp.program_code)=? AND LOWER(cp.plan_key)=?
                             AND (? IS NULL OR LOWER(TRIM(cat.catalog_key))=LOWER(TRIM(?)))
                             AND UPPER(COALESCE(c.course_code_normalized,mc.course_code_normalized))=? LIMIT 1""",
                        (program.upper(), plan.casefold(), catalog_key, catalog_key, target),
                    ).fetchone()
                if exists is None:
                    return _response("unsupported", f"ไม่พบวิชา {target} ในแผน {plan} ที่ระบุ", task_type, {"program": program, "plan": plan})
            status, answer, provenance = _format_h3(hard_result, target)
            scope = {"catalog_key": catalog_key, "program": program, "plan": plan, "target_course_code": target}
        elif task_type == "plan_structure_validation":
            hard_result = validate_curriculum_plan_structure(
                db_path, program, plan, catalog_key
            )
            status, answer, provenance = _format_h2(hard_result)
            scope = {"catalog_key": catalog_key, "program": program, "plan": plan}
        elif task_type == "seven_term_plan":
            if intent.get("horizon_terms") != _HORIZON_TERMS:
                return _response("unsupported", "รองรับเฉพาะการจัดลำดับ 7 ภาคเรียนปกติสำหรับกรณี 3.5 ปี", task_type, {"program": program, "plan": plan})
            hard_result = plan_curriculum_sequence(
                db_path, program, plan, horizon_terms=_HORIZON_TERMS,
                catalog_key=catalog_key,
            )
            status, answer, provenance = _format_h4(hard_result)
            scope = {"catalog_key": catalog_key, "program": program, "plan": plan}
        else:
            return _response("unsupported", "ไม่รองรับงาน Hard ประเภทนี้", task_type, {"program": program, "plan": plan})
    return _response(status, _append_citation_summary(answer, provenance), task_type, scope, provenance)


__all__ = ["answer_hard_question"]
