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

CATEGORY_RULES: dict[str, tuple[str, ...]] = {
    PROBATION_CATEGORY: ("22", "33.11"),
    REENTRY_CATEGORY: ("36",),
}

_PROBATION_VALUE_RE = re.compile(
    r"ค่าระดับคะแนน\s*เฉลี่ย\s*"
    r"(?P<scope>สะสม|ประจำภาคการศึกษาถัดไป)\s*"
    r"(?P<qualifier>ไม่ต่ำกว่า|ต่ำกว่า)\s*"
    r"(?P<value>[0-9๐-๙Oo]+(?:\.[0-9๐-๙Oo]+)?)(?![.0-9๐-๙OoDd])"
)
_REENTRY_PERIOD_RE = re.compile(
    r"ไม่เกิน\s*(?P<value>[0-9๐-๙Oo]+)\s*"
    r"(?P<unit>ปี|ภาคการศึกษา)(?![0-9๐-๙OoDd])"
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
    """Map extracted institutional Rules into two deterministic policy records."""

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
                category=PROBATION_CATEGORY,
                required_rule_numbers=CATEGORY_RULES[PROBATION_CATEGORY],
                indexed=indexed,
                value_extractor=_probation_values,
            ),
            self._map_category(
                category=REENTRY_CATEGORY,
                required_rule_numbers=CATEGORY_RULES[REENTRY_CATEGORY],
                indexed=indexed,
                value_extractor=_reentry_values,
            ),
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
