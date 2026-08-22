"""Deterministic semantic policy mapping for institution-wide Academic Rules."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


THAI_DIGIT_TRANSLATION = str.maketrans(
    "๐๑๒๓๔๕๖๗๘๙",
    "0123456789",
)
OCR_ZERO_TRANSLATION = str.maketrans({"O": "0", "o": "0"})

PROBATION_CATEGORY = "เกณฑ์ภาคทัณฑ์ (probation)"
REENTRY_CATEGORY = "การกลับเข้าศึกษา"
STATUS_CATEGORY = "เกณฑ์พ้นสภาพนักศึกษา"
GRADING_CATEGORY = "ระบบเกรด/การคิดคะแนน"
LEAVE_CATEGORY = "การลาพักการศึกษา"
RESIGNATION_CATEGORY = "การลาออก"
EXAM_CATEGORY = "การสอบ/วัดผล"
ACADEMIC_DISHONESTY_CATEGORY = "การทุจริตทางวิชาการ"
CONDUCT_CATEGORY = "ระเบียบความประพฤติ"
DISCIPLINARY_PENALTY_CATEGORY = "บทลงโทษทางวินัย"
APPEAL_CATEGORY = "การอุทธรณ์"

CATEGORY_RULES: dict[str, tuple[str, ...]] = {
    PROBATION_CATEGORY: ("22", "33.11"),
    REENTRY_CATEGORY: ("36",),
    STATUS_CATEGORY: (
        "33",
        "33.1",
        "33.2",
        "33.3",
        "33.4",
        "33.5",
        "33.6",
        "33.7",
        "33.8",
        "33.9",
        "33.10",
        "33.11",
        "33.12",
        "34",
        "35",
    ),
    GRADING_CATEGORY: (
        "19.3",
        "21",
        "21.1",
        "21.2",
        "21.2.1",
        "21.2.2",
        "21.2.3",
    ),
    LEAVE_CATEGORY: ("31", "31.1", "31.2", "31.3", "31.4"),
    RESIGNATION_CATEGORY: ("32",),
    EXAM_CATEGORY: ("19", "19.1", "19.2", "19.4", "19.5", "19.6", "19.7", "20", "23", "24"),
    ACADEMIC_DISHONESTY_CATEGORY: ("20", "33.8", "37.6.5", "41", "45.8", "45.9", "45.10"),
    CONDUCT_CATEGORY: (
        "37",
        "37.1",
        "37.2",
        "37.3",
        "37.4",
        "37.5",
        "37.6",
        "37.6.1",
        "37.6.2",
        "37.6.3",
        "37.6.4",
        "37.6.5",
        "37.6.6",
        "37.6.7",
        "37.6.8",
        "37.6.9",
        "37.6.10",
    ),
    DISCIPLINARY_PENALTY_CATEGORY: (
        "38",
        "38.1",
        "38.2",
        "38.3",
        "39",
        "39.1",
        "39.2",
        "39.3",
        "40",
        "41",
        "42",
    ),
    APPEAL_CATEGORY: ("43", "48", "49", "50", "51", "51.1", "51.2"),
}

_STRUCTURED_NUMBER = r"[0-9๐-๙Oo]+(?:\.[0-9๐-๙Oo]+)?"
_STRUCTURED_NUMBER_END = r"(?![.0-9๐-๙OoDd])"

_PROBATION_VALUE_RE = re.compile(
    r"ค่าระดับคะแนน\s*เฉลี่ย\s*"
    r"(?P<scope>สะสม|ประจำภาคการศึกษาถัดไป)\s*"
    r"(?P<qualifier>ไม่ต่ำกว่า|ต่ำกว่า)\s*"
    rf"(?P<value>{_STRUCTURED_NUMBER}){_STRUCTURED_NUMBER_END}"
)
_REENTRY_PERIOD_RE = re.compile(
    rf"ไม่เกิน\s*(?P<value>{_STRUCTURED_NUMBER})\s*"
    rf"(?P<unit>ปี|ภาคการศึกษา){_STRUCTURED_NUMBER_END}"
)
_STATUS_VALUE_RE = re.compile(
    r"ค่าระดับคะแนน\s*เฉลี่ย\s*สะสม\s*"
    r"(?P<qualifier>ไม่ต่ำกว่า|ต่ำกว่า)\s*"
    rf"(?P<value>{_STRUCTURED_NUMBER}){_STRUCTURED_NUMBER_END}"
)
_GRADE_ROW_RE = re.compile(
    r"(?m)^[ \t]*(?P<grade>A|B\+|C\+|D\+|B|C|D|F)[ \t]*\r?\n"
    r"[ \t]*(?P<value>[^\r\n \t]+)[ \t]*(?:\r?\n|$)"
)
_SUSPENSION_PERIOD_RE = re.compile(
    r"พักการเรียนในภาคการศึกษาปกติถัดไปอีก\s*"
    rf"(?P<value>{_STRUCTURED_NUMBER})\s*ภาคการศึกษา{_STRUCTURED_NUMBER_END}"
)
_DISCIPLINARY_COUNT_RE = re.compile(
    r"โทษทางวินัยอย่าง\s*(?P<severity>ไม่ร้ายแรง|ไมร้ายแรง|ร้ายแรง)\s*มี\s*"
    rf"(?P<value>{_STRUCTURED_NUMBER})\s*สถาน{_STRUCTURED_NUMBER_END}"
)
_APPEAL_DEADLINE_RE = re.compile(
    rf"ภายใน\s*(?P<value>{_STRUCTURED_NUMBER})\s*"
    rf"(?P<unit>วันทำการ|วัน){_STRUCTURED_NUMBER_END}"
)


def normalize_structured_number(value: str) -> str:
    """Normalize only complete numeric values used in structured output."""
    if not re.fullmatch(r"[0-9๐-๙Oo]+(?:\.[0-9๐-๙Oo]+)?", value):
        raise ValueError(f"Invalid structured numeric token: {value!r}")
    if not re.search(r"[0-9๐-๙]", value):
        raise ValueError(f"Ambiguous OCR-only numeric token: {value!r}")
    return value.translate(THAI_DIGIT_TRANSLATION).translate(OCR_ZERO_TRANSLATION)


def _try_normalize_structured_number(value: str) -> str | None:
    try:
        return normalize_structured_number(value)
    except ValueError:
        return None


def _is_valid_grade_point(value: str) -> bool:
    """Accept the table's complete half-point scale without repairing digits."""
    return value == "0" or bool(re.fullmatch(r"[0-4]\.(?:00|50)", value))


def _record_section_number(record: Mapping[str, Any]) -> str | None:
    section_number = record.get("section_number")
    if section_number is not None:
        return str(section_number).strip()

    rule_id = record.get("rule_id")
    if isinstance(rule_id, str) and rule_id.startswith("rule:"):
        return rule_id.removeprefix("rule:").strip()
    return None


def _rule_id(record: Mapping[str, Any], section_number: str) -> str:
    value = record.get("rule_id")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return f"rule:{section_number}"


def _nonempty_text(record: Mapping[str, Any]) -> bool:
    value = record.get("rule_text")
    return isinstance(value, str) and bool(value.strip())


def _has_provenance(record: Mapping[str, Any]) -> bool:
    provenance = record.get("source_provenance")
    return isinstance(provenance, list) and bool(provenance)


def _source_provenance(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for record in records:
        provenance = record.get("source_provenance")
        if not isinstance(provenance, list):
            continue
        for entry in provenance:
            if not isinstance(entry, Mapping):
                continue
            copied = dict(entry)
            identity = (
                str(copied.get("source_filename", "")),
                str(copied.get("source_page", "")),
                str(copied.get("document_category", "")),
            )
            if identity in seen:
                continue
            seen.add(identity)
            result.append(copied)
    return result


def _snippet(text: str, start: int, end: int, radius: int = 36) -> str:
    return text[max(0, start - radius) : min(len(text), end + radius)]


def _probation_values(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    values: list[dict[str, Any]] = []
    snippets: dict[str, list[dict[str, Any]]] = {}
    seen: set[tuple[str, str, str]] = set()

    for record in records:
        rule_id = str(record["rule_id"])
        text = str(record.get("rule_text", ""))
        snippets[rule_id] = []
        for match in _PROBATION_VALUE_RE.finditer(text):
            raw_value = match.group("value")
            qualifier = match.group("qualifier")
            scope = match.group("scope")
            condition = "at_least" if qualifier == "ไม่ต่ำกว่า" else "below"
            if scope == "ประจำภาคการศึกษาถัดไป":
                label = "GPA เฉลี่ยประจำภาคการศึกษาถัดไปขณะภาคทัณฑ์"
            elif condition == "at_least":
                label = "GPA สะสมที่พ้นภาคทัณฑ์"
            else:
                label = "GPA สะสมที่ทำให้ถูกภาคทัณฑ์"

            snippets[rule_id].append(
                {
                    "kind": "probation_threshold",
                    "text": _snippet(text, match.start(), match.end()),
                    "raw_value": raw_value,
                }
            )
            normalized_value = _try_normalize_structured_number(raw_value)
            if normalized_value is None:
                continue
            identity = (scope, condition, normalized_value)
            if identity in seen:
                continue
            seen.add(identity)
            values.append(
                {
                    "value": normalized_value,
                    "label": label,
                    "condition": condition,
                    "raw_value": raw_value,
                    "source_rule_id": rule_id,
                    "source_snippet": _snippet(text, match.start(), match.end()),
                }
            )
    return values, snippets


def _reentry_values(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    values: list[dict[str, Any]] = []
    snippets: dict[str, list[dict[str, Any]]] = {}
    seen: set[tuple[str, str]] = set()

    for record in records:
        rule_id = str(record["rule_id"])
        text = str(record.get("rule_text", ""))
        snippets[rule_id] = []
        for match in _REENTRY_PERIOD_RE.finditer(text):
            raw_value = match.group("value")
            unit = match.group("unit")
            snippets[rule_id].append(
                {
                    "kind": "reentry_period",
                    "text": _snippet(text, match.start(), match.end()),
                    "raw_value": raw_value,
                }
            )
            normalized_value = _try_normalize_structured_number(raw_value)
            if normalized_value is None:
                continue
            identity = (normalized_value, unit)
            if identity in seen:
                continue
            seen.add(identity)
            values.append(
                {
                    "value": normalized_value,
                    "unit": unit,
                    "label": "ระยะเวลาสูงสุดในการกลับเข้าศึกษา",
                    "condition": "at_most",
                    "raw_value": raw_value,
                    "source_rule_id": rule_id,
                    "source_snippet": _snippet(text, match.start(), match.end()),
                }
            )
    return values, snippets


def _text_only_values(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    return [], {str(record["rule_id"]): [] for record in records}


def _status_values(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    values: list[dict[str, Any]] = []
    snippets: dict[str, list[dict[str, Any]]] = {}
    seen: set[tuple[str, str]] = set()

    for record in records:
        rule_id = str(record["rule_id"])
        text = str(record.get("rule_text", ""))
        snippets[rule_id] = []
        for match in _STATUS_VALUE_RE.finditer(text):
            raw_value = match.group("value")
            condition = "at_least" if match.group("qualifier") == "ไม่ต่ำกว่า" else "below"
            snippets[rule_id].append(
                {
                    "kind": "status_threshold",
                    "text": _snippet(text, match.start(), match.end()),
                    "raw_value": raw_value,
                }
            )
            normalized_value = _try_normalize_structured_number(raw_value)
            if normalized_value is None:
                continue
            identity = (condition, normalized_value)
            if identity in seen:
                continue
            seen.add(identity)
            values.append(
                {
                    "value": normalized_value,
                    "label": "GPA ที่เป็นเกณฑ์พ้นสภาพนักศึกษา",
                    "condition": condition,
                    "raw_value": raw_value,
                    "source_rule_id": rule_id,
                    "source_snippet": _snippet(text, match.start(), match.end()),
                }
            )
    return values, snippets


def _grade_values(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    values: list[dict[str, Any]] = []
    snippets: dict[str, list[dict[str, Any]]] = {}
    seen: set[tuple[str, str]] = set()

    for record in records:
        rule_id = str(record["rule_id"])
        text = str(record.get("rule_text", ""))
        snippets[rule_id] = []
        for match in _GRADE_ROW_RE.finditer(text):
            raw_value = match.group("value")
            grade = match.group("grade")
            snippet = {
                "kind": "grade_point",
                "text": _snippet(text, match.start(), match.end()),
                "grade": grade,
                "raw_value": raw_value,
            }
            snippets[rule_id].append(snippet)
            normalized_value = _try_normalize_structured_number(raw_value)
            if normalized_value is None:
                snippet["status"] = "invalid_numeric_token"
                continue
            if not _is_valid_grade_point(normalized_value):
                snippet["status"] = "invalid_grade_point"
                continue
            identity = (grade, normalized_value)
            if identity in seen:
                continue
            seen.add(identity)
            values.append(
                {
                    "value": normalized_value,
                    "label": f"Grade {grade} แต้ม",
                    "condition": "grade_point",
                    "grade": grade,
                    "raw_value": raw_value,
                    "source_rule_id": rule_id,
                    "source_snippet": _snippet(text, match.start(), match.end()),
                }
            )
    return values, snippets


def _suspension_period_values(
    records: Sequence[Mapping[str, Any]],
    label: str,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    values: list[dict[str, Any]] = []
    snippets: dict[str, list[dict[str, Any]]] = {}
    seen: set[tuple[str, str]] = set()

    for record in records:
        rule_id = str(record["rule_id"])
        text = str(record.get("rule_text", ""))
        snippets[rule_id] = []
        if "ทุจริตในการสอบ" not in text:
            continue
        for match in _SUSPENSION_PERIOD_RE.finditer(text):
            raw_value = match.group("value")
            snippets[rule_id].append(
                {
                    "kind": "suspension_period",
                    "text": _snippet(text, match.start(), match.end()),
                    "raw_value": raw_value,
                }
            )
            normalized_value = _try_normalize_structured_number(raw_value)
            if normalized_value is None:
                continue
            identity = (normalized_value, "ภาคการศึกษา")
            if identity in seen:
                continue
            seen.add(identity)
            values.append(
                {
                    "value": normalized_value,
                    "unit": "ภาคการศึกษา",
                    "label": label,
                    "condition": "exam_dishonesty_suspension",
                    "raw_value": raw_value,
                    "source_rule_id": rule_id,
                    "source_snippet": _snippet(text, match.start(), match.end()),
                }
            )
    return values, snippets


def _exam_suspension_values(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    return _suspension_period_values(records, "ระยะพักการเรียนจากการทุจริตในการสอบ")


def _academic_dishonesty_suspension_values(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    return _suspension_period_values(records, "ระยะพักการเรียนจากการทุจริตในการสอบ")


def _disciplinary_penalty_values(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    values: list[dict[str, Any]] = []
    snippets: dict[str, list[dict[str, Any]]] = {}
    records_by_section = {
        _record_section_number(record): record
        for record in records
        if _record_section_number(record) is not None
    }
    child_rules = {
        "38": ("38.1", "38.2", "38.3"),
        "39": ("39.1", "39.2", "39.3"),
    }
    expected_severity = {"38": "ไม่ร้ายแรง", "39": "ร้ายแรง"}
    seen: set[tuple[str, str]] = set()

    for record in records:
        rule_id = str(record["rule_id"])
        section_number = _record_section_number(record)
        text = str(record.get("rule_text", ""))
        snippets[rule_id] = []
        if section_number not in child_rules:
            continue

        child_sections = child_rules[section_number]
        child_records = [records_by_section.get(child) for child in child_sections]
        if any(
            record is None or not _nonempty_text(record) or not _has_provenance(record)
            for record in child_records
        ):
            continue

        match = _DISCIPLINARY_COUNT_RE.search(text)
        raw_value = match.group("value") if match else None
        severity = expected_severity[section_number]
        if match:
            matched_severity = match.group("severity")
            if matched_severity == "ไมร้ายแรง":
                matched_severity = "ไม่ร้ายแรง"
            if matched_severity != severity:
                continue
            snippet_text = _snippet(text, match.start(), match.end())
        else:
            snippet_text = _snippet(text, 0, len(text), radius=0)

        child_count = str(len(child_sections))
        normalized_value = _try_normalize_structured_number(raw_value) if raw_value else None
        if normalized_value is not None and normalized_value != child_count:
            continue
        if normalized_value is None:
            normalized_value = child_count

        snippet = {
            "kind": "disciplinary_penalty_count",
            "text": snippet_text,
            "severity": severity,
            "raw_value": raw_value,
            "child_rule_ids": [f"rule:{child}" for child in child_sections],
            "status": "validated_by_child_rules",
        }
        snippets[rule_id].append(snippet)
        identity = (rule_id, normalized_value)
        if identity in seen:
            continue
        seen.add(identity)
        values.append(
            {
                "value": normalized_value,
                "unit": "สถาน",
                "label": f"จำนวนโทษทางวินัย ({severity})",
                "condition": "count",
                "severity": severity,
                "raw_value": raw_value,
                "source_rule_id": rule_id,
                "source_snippet": snippet_text,
                "supporting_rule_ids": [f"rule:{child}" for child in child_sections],
            }
        )
    return values, snippets


def _appeal_context(record_rule_id: str, text: str) -> tuple[str, str] | None:
    if record_rule_id == "rule:43" and "อุทธรณ์" in text and "ลงโทษ" in text:
        return "student_sanction_appeal", "ยื่นอุทธรณ์คำสั่งลงโทษ"
    if record_rule_id == "rule:48" and "อุทธรณ์" in text and "ปริญญา" in text:
        return "degree_decision_appeal", "ยื่นอุทธรณ์กรณีไม่เสนอชื่อรับปริญญา"
    if record_rule_id == "rule:49" and "คำชี้แจง" in text and "อุทธรณ์" in text:
        return "institution_response", "ส่งคำชี้แจงของส่วนงานวิชาการ"
    if record_rule_id == "rule:51.2" and "คำขอ" in text and "ภายใน" in text:
        return "reconsideration_request", "ยื่นคำขอพิจารณาอุทธรณ์ใหม่"
    return None


def _appeal_values(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    values: list[dict[str, Any]] = []
    snippets: dict[str, list[dict[str, Any]]] = {}
    seen: set[tuple[str, str, str, str]] = set()

    for record in records:
        rule_id = str(record["rule_id"])
        text = str(record.get("rule_text", ""))
        snippets[rule_id] = []
        context = _appeal_context(rule_id, text)
        if context is None:
            continue
        procedure, label = context
        for match in _APPEAL_DEADLINE_RE.finditer(text):
            raw_value = match.group("value")
            unit = match.group("unit")
            snippets[rule_id].append(
                {
                    "kind": "appeal_deadline",
                    "text": _snippet(text, match.start(), match.end()),
                    "raw_value": raw_value,
                    "unit": unit,
                    "procedure": procedure,
                }
            )
            normalized_value = _try_normalize_structured_number(raw_value)
            if normalized_value is None:
                continue
            identity = (rule_id, procedure, normalized_value, unit)
            if identity in seen:
                continue
            seen.add(identity)
            values.append(
                {
                    "value": normalized_value,
                    "unit": unit,
                    "label": label,
                    "condition": "deadline",
                    "procedure": procedure,
                    "raw_value": raw_value,
                    "source_rule_id": rule_id,
                    "source_snippet": _snippet(text, match.start(), match.end()),
                }
            )
    return values, snippets


_VALUE_EXTRACTORS: dict[str, Any] = {
    PROBATION_CATEGORY: _probation_values,
    REENTRY_CATEGORY: _reentry_values,
    STATUS_CATEGORY: _status_values,
    GRADING_CATEGORY: _grade_values,
    LEAVE_CATEGORY: _text_only_values,
    RESIGNATION_CATEGORY: _text_only_values,
    EXAM_CATEGORY: _exam_suspension_values,
    ACADEMIC_DISHONESTY_CATEGORY: _academic_dishonesty_suspension_values,
    CONDUCT_CATEGORY: _text_only_values,
    DISCIPLINARY_PENALTY_CATEGORY: _disciplinary_penalty_values,
    APPEAL_CATEGORY: _appeal_values,
}


def _summary(
    category: str,
    required_rule_ids: Sequence[str],
    values: Sequence[Mapping[str, Any]],
    present: bool | None,
) -> str:
    rules = " และ ".join(f"ข้อ {rule_id}" for rule_id in required_rule_ids)
    if present is None:
        return f"ยังไม่พบหลักฐานครบถ้วนสำหรับ{category}ตาม{rules}"
    if not values:
        return f"{category}ตาม{rules}; ไม่พบค่าตัวเลขที่ยืนยันได้จากข้อความปัจจุบัน"
    rendered_values = ", ".join(
        f"{item['label']}={item['value']}{item.get('unit', '')}"
        for item in values
    )
    return f"{category}ตาม{rules}; ค่าที่ดึงได้: {rendered_values}"


class RulesPolicyMapper:
    """Map extracted institutional Rules into deterministic policy records."""

    def map_data(self, extracted: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if isinstance(extracted, Mapping):
            raw_records = extracted.get("rules", [])
            source = extracted.get("source", "Academic Rules")
            total_rules = extracted.get("total_rules")
        else:
            raw_records = extracted
            source = "Academic Rules"
            total_rules = None

        if not isinstance(raw_records, Sequence) or isinstance(raw_records, (str, bytes)):
            raise ValueError("Extracted Rules input must contain a 'rules' list")

        indexed: dict[str, Mapping[str, Any]] = {}
        for raw_record in raw_records:
            if not isinstance(raw_record, Mapping):
                continue
            section_number = _record_section_number(raw_record)
            if section_number is not None and section_number not in indexed:
                indexed[section_number] = raw_record

        categories = [
            self._map_category(
                category=category,
                required_rule_numbers=CATEGORY_RULES[category],
                indexed=indexed,
                value_extractor=_VALUE_EXTRACTORS[category],
            )
            for category in CATEGORY_RULES
        ]
        return {
            "source": source,
            "input_total_rules": total_rules,
            "categories": categories,
        }

    def map_file(self, input_path: str | Path) -> dict[str, Any]:
        path = Path(input_path)
        with path.open("r", encoding="utf-8") as handle:
            extracted = json.load(handle)
        return self.map_data(extracted)

    @staticmethod
    def _map_category(
        category: str,
        required_rule_numbers: Sequence[str],
        indexed: Mapping[str, Mapping[str, Any]],
        value_extractor: Any,
    ) -> dict[str, Any]:
        found_records = [
            indexed[rule_number]
            for rule_number in required_rule_numbers
            if rule_number in indexed
        ]
        valid_records = [
            record
            for record in found_records
            if _nonempty_text(record) and _has_provenance(record)
        ]
        missing_rule_ids = [
            f"rule:{rule_number}"
            for rule_number in required_rule_numbers
            if rule_number not in indexed
            or not _nonempty_text(indexed[rule_number])
            or not _has_provenance(indexed[rule_number])
        ]
        present: bool | None = True if not missing_rule_ids else None
        values, snippets = value_extractor(valid_records)
        evidence_records = []
        for record in valid_records:
            rule_id = _rule_id(record, _record_section_number(record) or "")
            evidence_records.append(
                {
                    "rule_id": rule_id,
                    "rule_text": record.get("rule_text", ""),
                    "snippets": snippets.get(rule_id, []),
                }
            )

        evidence = {
            "rule_ids": [item["rule_id"] for item in evidence_records],
            "missing_rule_ids": missing_rule_ids,
            "source_provenance": _source_provenance(valid_records),
            "supporting_rule_text": evidence_records,
        }
        return {
            "category": category,
            "present": present,
            "values": values,
            "summary": _summary(category, required_rule_numbers, values, present),
            "evidence": evidence,
        }
