"""Execute Gold Questions through the project RAG pipeline and capture results."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # Keep import checks usable in minimal environments.
    def load_dotenv() -> None:
        return None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.answer import EMPTY_ANSWER, answer_question, is_fallback_like  # noqa: E402
from rag.providers.gemini import make_gemini_callable  # noqa: E402
from rag.qa import ask  # noqa: E402
from rag.router import route_question  # noqa: E402


GOLD_QUESTIONS_PATH = PROJECT_ROOT / "ground_truth" / "rag" / "gold_questions.json"
CURRICULUM_DB_PATH = PROJECT_ROOT / "cucumber_outputs" / "runtime" / "curriculum.db"
TOP_K = 10
ROUTES = {"structured", "semantic", "hybrid"}
DEFAULT_MIN_CALL_INTERVAL = 5.0
MAX_MODEL_RETRIES = 3
FALLBACK_RETRY_DELAY = 30.0
ANSWER_STATUSES = ("PASS", "PARTIAL", "FAIL", "REVIEW")


def _exception_text(exc: Exception) -> str:
    values = [str(exc)]
    for name in ("code", "status", "status_code", "reason"):
        value = getattr(exc, name, None)
        if callable(value):
            try:
                value = value()
            except Exception:  # noqa: BLE001 - diagnostic attribute only
                value = None
        if value is not None:
            values.append(str(value))
    return " ".join(values).upper()


def _write_output_atomically(output_path: Path, payload: Mapping[str, Any]) -> None:
    """Write an evaluation JSON file without truncating the existing output first."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as temporary_file:
            temporary_file.write(serialized)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
            temporary_path = Path(temporary_file.name)
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def _is_gemini_rate_limit(exc: Exception) -> bool:
    text = _exception_text(exc)
    return bool(re.search(r"\b429\b", text)) and "RESOURCE_EXHAUSTED" in text


def _delay_value(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return max(0.0, float(value))
    seconds = getattr(value, "seconds", None)
    nanos = getattr(value, "nanos", 0)
    if seconds is not None:
        try:
            return max(0.0, float(seconds) + float(nanos) / 1_000_000_000)
        except (TypeError, ValueError):
            return None
    match = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)\s*s?\s*", str(value))
    return float(match.group(1)) if match else None


def _retry_delay(exc: Exception) -> float:
    for name in (
        "retry_delay_seconds",
        "retry_after_seconds",
        "retry_delay",
        "retry_after",
    ):
        delay = _delay_value(getattr(exc, name, None))
        if delay is not None:
            return delay

    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is not None:
        for name in ("retry-after", "Retry-After"):
            delay = _delay_value(headers.get(name))
            if delay is not None:
                return delay
    return FALLBACK_RETRY_DELAY


class _SharedGeminiThrottle:
    """Throttle one shared Gemini callable and retry only Gemini 429 exhaustion."""

    def __init__(self, callable_: Any, min_call_interval: float) -> None:
        self._callable = callable_
        self._min_call_interval = min_call_interval
        self._last_call_at: float | None = None
        self.retry_count = 0

    def begin_question(self) -> None:
        self.retry_count = 0

    def _wait_for_interval(self) -> None:
        if self._last_call_at is None:
            return
        remaining = self._min_call_interval - (time.monotonic() - self._last_call_at)
        if remaining > 0:
            time.sleep(remaining)

    def __call__(self, prompt: str) -> str:
        retries = 0
        while True:
            self._wait_for_interval()
            self._last_call_at = time.monotonic()
            try:
                return self._callable(prompt)
            except Exception as exc:  # noqa: BLE001 - retry classification is explicit
                if not _is_gemini_rate_limit(exc) or retries >= MAX_MODEL_RETRIES:
                    raise
                retries += 1
                self.retry_count += 1
                delay = max(
                    _retry_delay(exc),
                    self._min_call_interval
                    - (time.monotonic() - self._last_call_at),
                )
                if delay > 0:
                    time.sleep(delay)


def _json_safe(value: Any) -> Any:
    """Convert pipeline containers, including SQLite row tuples, to JSON data."""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _normalized_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).casefold()).strip()


def _contains_text(haystack: Any, needle: Any) -> bool:
    return _normalized_text(needle) in _normalized_text(haystack)


def _structured_rows_for_grading(result: Any) -> list[dict[str, Any]]:
    if not isinstance(result, Mapping):
        return []
    rows = result.get("rows") or []
    columns = result.get("columns") or []
    normalized: list[dict[str, Any]] = []
    for row in rows if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)) else [rows]:
        if isinstance(row, Mapping):
            normalized.append(dict(row))
        elif isinstance(row, Sequence) and not isinstance(row, (str, bytes)):
            normalized.append(dict(zip(columns, row)))
    return normalized


def _iter_nested_mappings(value: Any):
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _iter_nested_mappings(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            yield from _iter_nested_mappings(child)


def _nested_mapping_value(value: Any, aliases: Sequence[str]) -> Any:
    for mapping in _iter_nested_mappings(value):
        for alias in aliases:
            if alias in mapping:
                return mapping[alias]
    return None


def _structured_sql(result: Any) -> str:
    return str(result.get("sql") or "") if isinstance(result, Mapping) else ""


def _sql_has_field_value(sql: str, field: str, value: Any) -> bool:
    if not sql:
        return False
    value_text = re.escape(str(value))
    field_pattern = rf"(?:[A-Za-z_][A-Za-z0-9_]*\.)?{re.escape(field)}"
    if isinstance(value, bool):
        value_pattern = "1" if value else "0"
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        value_pattern = re.escape(str(value))
    else:
        value_pattern = rf"['\"]{value_text}['\"]"
    return bool(
        re.search(
            rf"\b{field_pattern}\b\s*=\s*{value_pattern}",
            sql,
            flags=re.IGNORECASE,
        )
        or re.search(
            rf"\b{field_pattern}\b\s+IN\s*\([^)]*{value_pattern}",
            sql,
            flags=re.IGNORECASE,
        )
    )


def _structured_field_present(result: Any, field: str, value: Any) -> bool:
    aliases = {
        "program": ("program", "program_code"),
        "plan": ("plan", "plan_key"),
        "course_code": ("course_code", "course"),
        "year": ("year",),
        "semester": ("semester",),
        "total_credits": ("total_credits",),
        "credits_raw": ("credits_raw", "credits"),
        "name_en": ("name_en",),
    }.get(field, (field,))
    for row in _structured_rows_for_grading(result):
        if _nested_mapping_value(row, aliases) == value:
            return True
        if field == "program":
            provenance = _nested_mapping_value(row, ("provenance", "source_provenance"))
            if isinstance(provenance, Sequence) and not isinstance(provenance, (str, bytes)):
                if any(isinstance(item, Mapping) and item.get("program") == value for item in provenance):
                    return True
    if field == "plan":
        return _sql_has_field_value(_structured_sql(result), "plan_key", value)
    if field == "program":
        return _sql_has_field_value(_structured_sql(result), "program", value) or _sql_has_field_value(
            _structured_sql(result), "program_code", value
        )
    return any(_sql_has_field_value(_structured_sql(result), alias, value) for alias in aliases)


def _structured_course_codes(result: Any) -> set[str]:
    values: list[str] = []
    for row in _structured_rows_for_grading(result):
        values.extend(
            str(value)
            for mapping in _iter_nested_mappings(row)
            for value in mapping.values()
            if value is not None
        )
    values.append(_structured_sql(result))
    found: set[str] = set()
    for value in values:
        found.update(re.findall(r"(?<![A-Za-z0-9])[0-9xX]{8}(?![A-Za-z0-9])", value))
    return {value.casefold() for value in found}


def _row_value(row: Mapping[str, Any], field: str, *, plan: Any = None) -> Any:
    aliases = {
        "plan": ("plan", "plan_key"),
        "course_code": (
            "course_code",
            "course",
        ),
        "prerequisite_course_code": (
            "prerequisite_course_code",
            "prerequisite_code",
            "course_code",
            "raw_text",
        ),
        "credits_raw": ("credits_raw", "credits"),
    }.get(field, (field,))
    if plan in {"coop", "no_coop"}:
        scoped_aliases = {
            "year": (f"{plan}_year",),
            "semester": (f"{plan}_semester",),
            "flexible_year_semester_raw": (
                f"{plan}_flexible_year_semester_raw",
                f"{plan}_placement_raw",
            ),
            "credits_raw": (f"{plan}_credits_raw", f"{plan}_credits"),
            "course_code": (f"{plan}_course_code",),
        }.get(field, ())
        aliases = (*scoped_aliases, *aliases)
    value = _nested_mapping_value(row, aliases)
    if value is not None:
        return value
    if field == "plan" and plan in {"coop", "no_coop"}:
        scoped_keys = (
            f"{plan}_year",
            f"{plan}_semester",
            f"{plan}_flexible_year_semester_raw",
            f"{plan}_placement_raw",
            f"{plan}_credits_raw",
            f"{plan}_credits",
            f"{plan}_course_code",
        )
        if any(key in row for key in scoped_keys):
            return plan
    return None


def _row_matches(row: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    expected_plan = expected.get("plan")
    for field, value in expected.items():
        if value is None:
            continue
        actual = _row_value(row, field, plan=expected_plan)
        if actual != value:
            return False
    return True


def _answer_has_value(answer: Any, field: str, value: Any) -> bool:
    if value is None or answer is None:
        return value is None
    text = _normalized_text(answer)
    if field == "plan":
        aliases = {
            "coop": ("coop", "สหกิจ"),
            "no_coop": ("no_coop", "ไม่สหกิจ"),
        }.get(str(value).casefold(), (str(value),))
        return any(_normalized_text(alias) in text for alias in aliases)
    if field == "requirement_type" and str(value).casefold() == "required":
        return "required" in text or "บังคับ" in text or "ต้องผ่าน" in text
    if field == "year":
        return bool(
            re.search(
                rf"(?:ปี|year)\s*(?:ที่\s*)?{re.escape(str(value))}(?!\d)",
                text,
            )
        )
    if field == "semester":
        return bool(
            re.search(
                rf"(?:เทอม|ภาคเรียน|ภาคการศึกษา|semester)\s*(?:ที่\s*)?{re.escape(str(value))}(?!\d)",
                text,
            )
        )
    return _contains_text(text, value)


def _question_relevance(question: str, expected: Mapping[str, Any]) -> dict[str, Any]:
    text = _normalized_text(question)
    placement_intent = bool(
        re.search(
            r"(?:เรียนช่วงไหน|ช่วงไหน|ช่วงใด|อยู่ช่วงไหน|เรียนปีไหน|อยู่ปีไหน|"
            r"เทอมไหน|ภาคเรียนไหน|ภาคการศึกษาไหน|ช่วงเรียน|จัดช่วง|จัดวาง|placement|"
            r"ตามปี|ตามเทอม|ปี/เทอม|เรียงลำดับ|ลำดับปี|when\s+(?:the\s+)?course\s+is\s+taken)",
            text,
        )
    )
    credit_intent = "หน่วยกิต" in text or "เครดิต" in text
    name_intent = bool(
        re.search(r"ชื่อภาษาอังกฤษ|ชื่อภาษาไทย|ชื่อวิชา|ชื่ออังกฤษ|ชื่อไทย|\bname\b", text)
    )
    prerequisite_intent = bool(
        re.search(r"ต้องผ่าน|ก่อนลง|เรียนก่อน|prerequisite", text)
    )
    course_list_intent = bool(
        re.search(r"อะไรบ้าง|วิชาใด|วิชาอะไร|รายวิชา|กลุ่มวิชา|เลือกได้", text)
    )
    flexible_intent = bool(
        re.search(r"ช่วงไหน|ช่วงใด|ช่วงเรียน|จัดช่วง|ยืดหยุ่น|เปิดให้ลง|แต่ละแผน", text)
    )
    alternative_intent = bool(
        re.search(r"เลือกได้|เลือกกี่|กี่วิชา|minimum|maximum|alternative", text)
    )
    explicit_course_code = bool(re.search(r"[0-9]{8}", text))
    placement_fields = {"plan"}
    if placement_intent:
        placement_fields.update({"year", "semester"})
    if flexible_intent:
        placement_fields.add("flexible_year_semester_raw")
    if credit_intent:
        placement_fields.add("credits_raw")

    scalar_fields = set()
    if "program" in expected and re.search(r"\bit\b", text):
        scalar_fields.add("program")
    if "plan" in expected and ("สหกิจ" in text or "coop" in text or "แผน" in text):
        scalar_fields.add("plan")
    if "course_code" in expected and explicit_course_code:
        scalar_fields.add("course_code")
    if "year" in expected and placement_intent:
        scalar_fields.add("year")
    if "semester" in expected and placement_intent:
        scalar_fields.add("semester")
    if "total_credits" in expected and credit_intent:
        scalar_fields.add("total_credits")
    raw_credit_intent = bool(re.search(r"รูปแบบหน่วยกิต|แจกแจงหน่วยกิต|credits[_ ]raw", text))
    if "credits_raw" in expected and raw_credit_intent:
        scalar_fields.add("credits_raw")
    if "name_en" in expected and name_intent:
        scalar_fields.add("name_en")
    if "course_codes" in expected and course_list_intent and not prerequisite_intent:
        scalar_fields.add("course_codes")

    return {
        "scalar_fields": scalar_fields,
        "placement_fields": placement_fields,
        "prerequisite_fields": {"course_code"} if prerequisite_intent else set(),
        "alternative_fields": {
            "minimum_choices",
            "maximum_choices",
        }
        if alternative_intent
        else set(),
    }


def _semantic_texts_for_facts(result: Any) -> list[str]:
    return [
        str(chunk.get("text") or "")
        for chunk in _semantic_chunks_for_grading(result)
        if str(chunk.get("text") or "")
    ]


def _semantic_plan_has_value(text: str, value: Any) -> bool:
    normalized = _normalized_text(text)
    plan = str(value).casefold()
    if plan == "no_coop":
        return "no_coop" in normalized or "ไม่สหกิจ" in normalized
    if plan == "coop":
        if "ไม่สหกิจ" in normalized or "no_coop" in normalized:
            return False
        return bool(re.search(r"(?<![a-z0-9_-])coop(?![a-z0-9_-])", normalized)) or "สหกิจ" in normalized
    return _contains_text(normalized, value)


def _semantic_text_has_value(text: str, field: str, value: Any) -> bool:
    if field == "plan":
        return _semantic_plan_has_value(text, value)
    if field == "year":
        return bool(re.search(rf"(?:ปี|year)\s*(?:ที่\s*)?{re.escape(str(value))}(?!\d)", _normalized_text(text)))
    if field == "semester":
        return bool(
            re.search(
                rf"(?:เทอม|ภาคเรียน|ภาคการศึกษา|semester)\s*(?:ที่\s*)?{re.escape(str(value))}(?!\d)",
                _normalized_text(text),
            )
        )
    return _contains_text(text, value)


def _semantic_field_present(result: Any, field: str, value: Any) -> bool:
    return any(_semantic_text_has_value(text, field, value) for text in _semantic_texts_for_facts(result))


def _semantic_placement_matches(result: Any, expected: Mapping[str, Any]) -> bool:
    fields = {field: value for field, value in expected.items() if value is not None}
    return any(
        all(_semantic_text_has_value(text, field, value) for field, value in fields.items())
        for text in _semantic_texts_for_facts(result)
    )


def _semantic_prerequisite_matches(result: Any, expected: Mapping[str, Any]) -> bool:
    code = expected.get("course_code") or expected.get("prerequisite_course_code")
    if code is None:
        return False
    relation = re.compile(r"ก่อน|ต้องผ่าน|เรียนก่อน|prerequisite|required|บังคับ", re.IGNORECASE)
    return any(
        _contains_text(text, code) and relation.search(text)
        for text in _semantic_texts_for_facts(result)
    )


def _structured_checks(
    question: str,
    expected: Mapping[str, Any],
    result: Any,
    answer: Any,
    semantic_result: Any = None,
) -> list[dict[str, Any]]:
    relevance = _question_relevance(question, expected)
    checks: list[dict[str, Any]] = []
    for field in ("program", "plan", "course_code", "year", "semester", "total_credits", "credits_raw", "name_en"):
        if field in expected:
            value = expected[field]
            checks.append(
                {
                    "label": field,
                    "expected": value,
                    "evidence": _structured_field_present(result, field, value)
                    or _semantic_field_present(semantic_result, field, value),
                    "answer": _answer_has_value(answer, field, value),
                    "evidence_required": False if field in {"program", "plan", "course_code"} else field in relevance["scalar_fields"],
                    "answer_required": field in relevance["scalar_fields"] and field not in {"program", "plan", "course_code"},
                }
            )

    if "course_codes" in expected:
        expected_codes = {str(code).casefold() for code in expected["course_codes"]}
        actual_codes = _structured_course_codes(result)
        for text in _semantic_texts_for_facts(semantic_result):
            actual_codes.update(
                code.casefold()
                for code in re.findall(r"(?<![A-Za-z0-9])[0-9xX]{8}(?![A-Za-z0-9])", text)
            )
        checks.append(
            {
                "label": "course_codes",
                "expected": expected["course_codes"],
                "evidence": expected_codes.issubset(actual_codes),
                "answer": all(_answer_has_value(answer, "course_code", code) for code in expected["course_codes"]),
                "evidence_required": "course_codes" in relevance["scalar_fields"],
                "answer_required": "course_codes" in relevance["scalar_fields"],
            }
        )

    for index, placement in enumerate(expected.get("placements", [])):
        rows = _structured_rows_for_grading(result)
        relevant_placement = {
            field: value
            for field, value in placement.items()
            if field in relevance["placement_fields"]
        }
        evidence = any(_row_matches(row, relevant_placement) for row in rows) or _semantic_placement_matches(
            semantic_result, relevant_placement
        )
        answer_ok = all(
            _answer_has_value(answer, field, value)
            for field, value in relevant_placement.items()
            if value is not None
        )
        checks.append(
            {
                "label": f"placements[{index}]",
                "expected": placement,
                "evidence": evidence,
                "answer": answer_ok,
                "required_fields": sorted(relevant_placement),
                "evidence_required": bool(relevant_placement),
                "answer_required": bool([value for value in relevant_placement.values() if value is not None]),
            }
        )

    if "placement" in expected:
        placement = expected["placement"]
        rows = _structured_rows_for_grading(result)
        relevant_placement = {
            field: value
            for field, value in placement.items()
            if field in relevance["placement_fields"]
        }
        evidence = any(_row_matches(row, relevant_placement) for row in rows) or _semantic_placement_matches(
            semantic_result, relevant_placement
        )
        answer_ok = all(
            _answer_has_value(answer, field, value)
            for field, value in relevant_placement.items()
            if value is not None
        )
        checks.append(
            {
                "label": "placement",
                "expected": placement,
                "evidence": evidence,
                "answer": answer_ok,
                "required_fields": sorted(relevant_placement),
                "evidence_required": bool(relevant_placement),
                "answer_required": bool([value for value in relevant_placement.values() if value is not None]),
            }
        )

    for index, prerequisite in enumerate(expected.get("prerequisites", [])):
        rows = _structured_rows_for_grading(result)
        relevant_prerequisite = {
            "prerequisite_course_code" if field == "course_code" else field: value
            for field, value in prerequisite.items()
            if field in relevance["prerequisite_fields"]
        }
        evidence = any(
            _row_matches(
                row,
                relevant_prerequisite,
            )
            for row in rows
        ) or _semantic_prerequisite_matches(semantic_result, prerequisite)
        answer_ok = all(
            _answer_has_value(answer, field, value)
            for field, value in relevant_prerequisite.items()
            if value is not None
        )
        checks.append(
            {
                "label": f"prerequisites[{index}]",
                "expected": prerequisite,
                "evidence": evidence,
                "answer": answer_ok,
                "required_fields": sorted(relevant_prerequisite),
                "evidence_required": bool(relevant_prerequisite),
                "answer_required": bool([value for value in relevant_prerequisite.values() if value is not None]),
            }
        )

    if "alternative" in expected:
        alternative = expected["alternative"]
        rows = _structured_rows_for_grading(result)
        relevant_alternative = {
            field: value
            for field, value in alternative.items()
            if field in relevance["alternative_fields"]
        }
        evidence = any(
            all(row.get(field) == value for field, value in relevant_alternative.items())
            for row in rows
        )
        answer_ok = all(
            _answer_has_value(answer, field, value)
            for field, value in relevant_alternative.items()
            if value is not None
        )
        checks.append(
            {
                "label": "alternative",
                "expected": alternative,
                "evidence": evidence,
                "answer": answer_ok,
                "required_fields": sorted(relevant_alternative),
                "evidence_required": bool(relevant_alternative),
                "answer_required": bool([value for value in relevant_alternative.values() if value is not None]),
            }
        )
    return checks


def _semantic_chunks_for_grading(result: Any) -> list[Mapping[str, Any]]:
    if not isinstance(result, Sequence) or isinstance(result, (str, bytes)):
        return []
    return [chunk for chunk in result if isinstance(chunk, Mapping)]


def _semantic_relevant_text(result: Any, expected: Mapping[str, Any]) -> str:
    chunks = _semantic_chunks_for_grading(result)
    identifiers = [
        str(expected[field])
        for field in ("course_code", "name_en")
        if expected.get(field)
    ]
    relevant = [
        str(chunk.get("text") or "")
        for chunk in chunks
        if not identifiers or any(_contains_text(chunk.get("text") or "", identifier) for identifier in identifiers)
    ]
    return "\n".join(relevant or [str(chunk.get("text") or "") for chunk in chunks])


def _character_ngrams(value: Any, size: int = 3) -> set[str]:
    text = re.sub(r"\s+", "", str(value).casefold())
    return {text[index : index + size] for index in range(max(0, len(text) - size + 1))}


def _semantic_answer_supports_paraphrase(
    result: Any,
    expected: Mapping[str, Any],
    answer: Any,
) -> tuple[bool, float]:
    """Use conservative evidence-overlap for deterministic Thai paraphrase checks."""
    if not isinstance(answer, str) or is_fallback_like(answer):
        return False, 0.0
    answer_grams = _character_ngrams(answer)
    evidence_grams = _character_ngrams(_semantic_relevant_text(result, expected))
    if not answer_grams or not evidence_grams:
        return False, 0.0
    score = len(answer_grams & evidence_grams) / len(answer_grams)
    return score >= 0.68, score


def _semantic_provenance_items(chunks: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    items: list[Mapping[str, Any]] = []
    for chunk in chunks:
        provenance = chunk.get("provenance")
        if isinstance(provenance, Sequence) and not isinstance(provenance, (str, bytes)):
            items.extend(item for item in provenance if isinstance(item, Mapping))
    return items


def _provenance_matches(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> bool:
    for field, value in expected.items():
        if field == "source_document_key":
            actual_value = actual.get("source_document_key", actual.get("source_filename"))
        else:
            actual_value = actual.get(field)
        if actual_value != value:
            return False
    return True


def _semantic_checks(
    question: str,
    expected: Mapping[str, Any],
    result: Any,
    answer: Any,
) -> dict[str, Any]:
    chunks = _semantic_chunks_for_grading(result)
    texts = "\n".join(str(chunk.get("text") or "") for chunk in chunks)
    relevance = _question_relevance(question, expected)
    description_expected = [str(item) for item in expected.get("description_evidence", [])]
    description_found = [item for item in description_expected if _contains_text(texts, item)]
    provenance_expected = expected.get("provenance", [])
    provenance_items = _semantic_provenance_items(chunks)
    provenance_found = [
        item
        for item in provenance_expected
        if any(_provenance_matches(item, actual) for actual in provenance_items)
    ]
    metadata_checks = []
    for field in ("program", "plan", "course_code", "name_en"):
        if field in expected:
            value = str(expected[field])
            metadata_checks.append(
                {
                    "label": field,
                    "evidence": _contains_text(texts, value),
                    "answer": _answer_has_value(answer, field, value),
                    "evidence_required": field in {"program", "plan", "course_code"},
                    "answer_required": field in relevance["scalar_fields"]
                    and field not in {"program", "plan", "course_code"},
                }
            )
    retrieved_checks = metadata_checks + [
        {
            "label": f"description_evidence[{index}]",
            "evidence": item in description_found,
            "answer": _contains_text(answer, item),
            "evidence_required": True,
            "answer_required": True,
        }
        for index, item in enumerate(description_expected)
    ]
    evidence_required_checks = [
        item for item in retrieved_checks if item.get("evidence_required", True)
    ]
    answer_required_checks = [
        item for item in retrieved_checks if item.get("answer_required", True)
    ]
    evidence_count = sum(bool(item["evidence"]) for item in evidence_required_checks)
    evidence_total = len(evidence_required_checks)
    answer_count = sum(bool(item["answer"]) for item in answer_required_checks)
    answer_total = len(answer_required_checks)
    description_answer_found = [item for item in description_expected if _contains_text(answer, item)]
    paraphrase_supported, paraphrase_score = _semantic_answer_supports_paraphrase(
        result,
        expected,
        answer,
    )
    provenance_ok = len(provenance_found) == len(provenance_expected)
    return {
        "checks": retrieved_checks,
        "evidence_count": evidence_count,
        "evidence_total": evidence_total,
        "answer_count": answer_count,
        "answer_total": answer_total,
        "description_found": description_found,
        "description_missing": [item for item in description_expected if item not in description_found],
        "description_answer_found": description_answer_found,
        "paraphrase_supported": paraphrase_supported,
        "paraphrase_score": paraphrase_score,
        "provenance_found": provenance_found,
        "provenance_missing": [item for item in provenance_expected if item not in provenance_found],
        "provenance_correct": provenance_ok,
    }


def _status_from_checks(
    checks: Sequence[Mapping[str, Any]],
    final_answer: Any,
    *,
    evidence_label: str,
) -> tuple[str, str]:
    if is_fallback_like(final_answer):
        return "FAIL", "non-unknown question returned a fallback-like denial"
    evidence_checks = [
        item for item in checks if item.get("evidence_required", True)
    ]
    answer_checks = [item for item in checks if item.get("answer_required", True)]
    if not evidence_checks and not answer_checks:
        return "PASS", "no additional user-requested facts require grading"
    evidence_count = sum(bool(item.get("evidence")) for item in evidence_checks)
    evidence_total = len(evidence_checks)
    answer_count = sum(bool(item.get("answer")) for item in answer_checks)
    answer_total = len(answer_checks)
    if evidence_total and evidence_count < evidence_total:
        missing = [item.get("label") for item in evidence_checks if not item.get("evidence")]
        return "FAIL", f"{evidence_label} missing: {', '.join(missing)}"
    if answer_total and answer_count == answer_total:
        return "PASS", "all expected facts are present in evidence and final answer"
    if answer_count:
        return "PARTIAL", f"final answer supports {answer_count}/{answer_total} checked facts"
    return "REVIEW", "evidence is present but exact final-answer coverage needs semantic review"


def _grade_answer(gold: Mapping[str, Any], result: Mapping[str, Any]) -> tuple[str, str, dict[str, Any]]:
    gold_type = gold["type"]
    final_answer = result.get("final_answer")
    if gold_type == "unknown":
        if isinstance(final_answer, str) and final_answer == EMPTY_ANSWER:
            return "PASS", "unknown question returned the exact required fallback", {
                "unknown_exact_fallback": True
            }
        return "FAIL", "unknown question did not return the exact required fallback", {
            "unknown_exact_fallback": False
        }

    if is_fallback_like(final_answer):
        return "FAIL", "fallback returned for a non-unknown question", {
            "unknown_exact_fallback": False
        }

    structured_details = None
    semantic_details = None
    if gold_type in {"structured", "hybrid"}:
        structured_details = _structured_checks(
            gold["question"],
            gold["expected"],
            result.get("structured_result"),
            final_answer,
            result.get("semantic_results"),
        )
        structured_status, structured_reason = _status_from_checks(
            structured_details,
            final_answer,
            evidence_label="structured evidence",
        )
    else:
        structured_status, structured_reason = None, None

    if gold_type in {"semantic", "hybrid"}:
        semantic_details = _semantic_checks(
            gold["question"],
            gold["expected"],
            result.get("semantic_results"),
            final_answer,
        )
        semantic_checks = semantic_details["checks"]
        semantic_status, semantic_reason = _status_from_checks(
            semantic_checks,
            final_answer,
            evidence_label="semantic evidence",
        )
        if (
            semantic_status == "REVIEW"
            and semantic_details["evidence_count"] == semantic_details["evidence_total"]
            and semantic_details["paraphrase_supported"]
        ):
            semantic_status = "PASS"
            semantic_reason = "retrieved concepts and conservative evidence-supported paraphrase are complete"
        if semantic_details["provenance_missing"] and semantic_status == "PASS":
            semantic_status = "REVIEW"
            semantic_reason = "retrieved facts match, but expected provenance is incomplete"
        elif (
            semantic_status == "PASS"
            and not semantic_details["paraphrase_supported"]
            and semantic_details["answer_count"] < semantic_details["answer_total"]
        ):
            semantic_status = "REVIEW"
            semantic_reason = "retrieved evidence is complete; final wording needs semantic review"
        elif (
            semantic_status == "PARTIAL"
            and semantic_details["evidence_count"] == semantic_details["evidence_total"]
            and gold["expected"].get("description_evidence")
            and re.search(r"[\u0e00-\u0e7f]", str(final_answer or ""))
        ):
            semantic_status = "REVIEW"
            semantic_reason = "retrieved evidence is complete; Thai paraphrase needs semantic review"
    else:
        semantic_status, semantic_reason = None, None

    if gold_type == "structured":
        return structured_status, structured_reason, {
            "structured_fact_correctness": structured_status,
            "structured_checks": structured_details,
            "semantic_evidence_coverage": None,
            "provenance_correct": None,
        }
    if gold_type == "semantic":
        return semantic_status, semantic_reason, {
            "structured_fact_correctness": None,
            "structured_checks": None,
            "semantic_evidence_coverage": {
                "matched": len(semantic_details["description_found"]),
                "expected": len(gold["expected"].get("description_evidence", [])),
                "description_matched": len(semantic_details["description_found"]),
                "description_expected": len(gold["expected"].get("description_evidence", [])),
                "metadata_matched": semantic_details["evidence_count"] - len(semantic_details["description_found"]),
                "metadata_expected": len(semantic_details["checks"]) - len(gold["expected"].get("description_evidence", [])),
            },
            "provenance_correct": semantic_details["provenance_correct"],
        }

    statuses = [structured_status, semantic_status]
    if "FAIL" in statuses:
        status = "FAIL"
    elif "PARTIAL" in statuses:
        status = "PARTIAL"
    elif "REVIEW" in statuses:
        status = "REVIEW"
    else:
        status = "PASS"
    reason = f"structured: {structured_reason}; semantic: {semantic_reason}"
    return status, reason, {
        "structured_fact_correctness": structured_status,
        "structured_checks": structured_details,
        "semantic_evidence_coverage": {
            "matched": len(semantic_details["description_found"]),
            "expected": len(gold["expected"].get("description_evidence", [])),
            "description_matched": len(semantic_details["description_found"]),
            "description_expected": len(gold["expected"].get("description_evidence", [])),
            "metadata_matched": semantic_details["evidence_count"] - len(semantic_details["description_found"]),
            "metadata_expected": len(semantic_details["checks"]) - len(gold["expected"].get("description_evidence", [])),
        },
        "provenance_correct": semantic_details["provenance_correct"],
    }


def _append_provenance(
    target: list[Mapping[str, Any]],
    value: Any,
) -> None:
    if isinstance(value, Mapping):
        target.append(value)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        target.extend(item for item in value if isinstance(item, Mapping))


def _structured_provenance_items(result: Any) -> list[Mapping[str, Any]]:
    items: list[Mapping[str, Any]] = []
    if not isinstance(result, Mapping):
        return items
    _append_provenance(items, result.get("provenance"))
    _append_provenance(items, result.get("source_provenance"))
    for row in _structured_rows_for_grading(result):
        _append_provenance(items, row.get("provenance"))
        _append_provenance(items, row.get("source_provenance"))
    return items


def _required_provenance(gold: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    value = gold.get("required_provenance")
    if value is None:
        expected = gold.get("expected")
        value = expected.get("provenance", []) if isinstance(expected, Mapping) else []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _provenance_correct_for_result(
    gold: Mapping[str, Any],
    structured_result: Any,
    semantic_results: Any,
) -> bool:
    expected = _required_provenance(gold)
    if not expected:
        return True
    actual = _structured_provenance_items(structured_result)
    actual.extend(_semantic_provenance_items(_semantic_chunks_for_grading(semantic_results)))
    return all(
        any(_provenance_matches(item, candidate) for candidate in actual)
        for item in expected
    )


def _earliest_failure_stage(
    gold: Mapping[str, Any],
    graded: Mapping[str, Any],
) -> str | None:
    if not graded.get("execution_success"):
        return "execution"
    if gold["type"] == "unknown":
        return None if graded.get("not_found_correct") else "final_answer"
    if gold["type"] in {"structured", "hybrid"} and graded.get("evidence_correct") is False:
        return "structured_evidence"
    if gold["type"] in {"semantic", "hybrid"}:
        coverage = graded.get("semantic_evidence_coverage")
        if isinstance(coverage, Mapping) and coverage.get("matched") != coverage.get("expected"):
            return "semantic_evidence"
    if graded.get("provenance_correct") is False:
        return "provenance"
    if graded.get("answer_correctness") != "PASS":
        return "final_answer"
    return None


def _grade_result(gold: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    graded = dict(result)
    answer_correctness, answer_check_reason, details = _grade_answer(gold, result)
    execution_success = bool(result.get("execution_success"))
    structured_checks = details.get("structured_checks")
    semantic_coverage = details.get("semantic_evidence_coverage")
    if not execution_success:
        evidence_correct = None
        provenance_correct = None
    elif isinstance(structured_checks, Sequence) and not isinstance(structured_checks, (str, bytes)):
        structured_evidence_correct = all(
            bool(check.get("evidence"))
            for check in structured_checks
            if check.get("evidence_required")
        )
        if isinstance(semantic_coverage, Mapping):
            semantic_evidence_correct = (
                semantic_coverage.get("matched") == semantic_coverage.get("expected")
            )
            evidence_correct = structured_evidence_correct and semantic_evidence_correct
        else:
            evidence_correct = structured_evidence_correct
        provenance_correct = _provenance_correct_for_result(
            gold,
            result.get("structured_result"),
            result.get("semantic_results"),
        )
    elif isinstance(semantic_coverage, Mapping):
        evidence_correct = semantic_coverage.get("matched") == semantic_coverage.get("expected")
        provenance_correct = _provenance_correct_for_result(
            gold,
            result.get("structured_result"),
            result.get("semantic_results"),
        )
    else:
        evidence_correct = True
        provenance_correct = True
    not_found_correct = (
        answer_correctness == "PASS" if gold["type"] == "unknown" else None
    )
    graded.update(
        {
            "id": result.get("id", gold["id"]),
            "type": gold["type"],
            "question": gold.get("question", result.get("question")),
            **details,
            "difficulty": gold.get("difficulty"),
            "execution_success": execution_success,
            "actual_route": result.get("actual_route"),
            "route_match": result.get("route_match"),
            "answer_correctness": answer_correctness,
            "answer_check_reason": answer_check_reason,
            "evidence_correct": evidence_correct,
            "provenance_correct": provenance_correct,
            "not_found_correct": not_found_correct,
            "latency_sec": result.get("latency_sec"),
            "model_retry_count": result.get("model_retry_count", 0),
            "error": result.get("error"),
        }
    )
    graded["earliest_failure_stage"] = _earliest_failure_stage(
        gold,
        graded,
    )
    return graded


def _merge_gold_with_runtime_results(
    gold_questions: Sequence[Mapping[str, Any]],
    runtime_results: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Pair runtime observations with the current Gold definition by question ID."""

    gold_by_id: dict[str, Mapping[str, Any]] = {}
    gold_ids: list[str] = []
    for item in gold_questions:
        question_id = item.get("id")
        if not isinstance(question_id, str) or not question_id:
            raise ValueError("Gold Questions must contain a non-empty string id")
        if question_id in gold_by_id:
            raise ValueError(f"Gold Questions contain duplicate id: {question_id}")
        gold_by_id[question_id] = item
        gold_ids.append(question_id)

    runtime_by_id: dict[str, Mapping[str, Any]] = {}
    runtime_ids: list[str] = []
    for item in runtime_results:
        if not isinstance(item, Mapping):
            raise ValueError("Existing evaluation results must contain objects")
        question_id = item.get("id")
        if not isinstance(question_id, str) or not question_id:
            raise ValueError("Existing evaluation results must contain a non-empty string id")
        if question_id in runtime_by_id:
            raise ValueError(f"Existing evaluation results contain duplicate id: {question_id}")
        runtime_by_id[question_id] = item
        runtime_ids.append(question_id)

    missing_ids = [question_id for question_id in gold_ids if question_id not in runtime_by_id]
    unexpected_ids = [question_id for question_id in runtime_ids if question_id not in gold_by_id]
    if missing_ids or unexpected_ids:
        problems = []
        if missing_ids:
            problems.append(f"missing Gold ids: {', '.join(missing_ids)}")
        if unexpected_ids:
            problems.append(f"unexpected runtime ids: {', '.join(unexpected_ids)}")
        raise ValueError("Existing evaluation IDs do not match current Gold: " + "; ".join(problems))

    gold_owned_fields = set().union(*(item.keys() for item in gold_questions))
    merged_results: list[dict[str, Any]] = []
    for question_id in gold_ids:
        runtime = runtime_by_id[question_id]
        merged = {
            key: value for key, value in runtime.items() if key not in gold_owned_fields
        }
        merged.update(gold_by_id[question_id])
        merged_results.append(merged)
    return merged_results


def grade_results(
    gold_questions: Sequence[Mapping[str, Any]], results: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    gold_by_id = {item["id"]: item for item in gold_questions}
    graded = [
        _grade_result(gold_by_id[result["id"]], result)
        for result in results
        if result.get("id") in gold_by_id
    ]
    statuses = Counter(item["answer_correctness"] for item in graded)
    executable = [item for item in graded if item.get("execution_success") is True]
    route_applicable = [item for item in graded if item.get("type") != "unknown"]
    unknown = [item for item in graded if item.get("type") == "unknown"]
    semantic_items = [item for item in graded if item.get("type") in {"semantic", "hybrid"}]
    semantic_coverage = [
        item["semantic_evidence_coverage"]
        for item in semantic_items
        if isinstance(item.get("semantic_evidence_coverage"), Mapping)
    ]
    semantic_matched = sum(item["matched"] for item in semantic_coverage)
    semantic_expected = sum(item["expected"] for item in semantic_coverage)
    provenance_items = [
        item for item in graded if item.get("provenance_correct") is not None
    ]
    evidence_items = [
        item for item in graded if item.get("evidence_correct") is not None
    ]
    not_found_items = [
        item for item in graded if item.get("not_found_correct") is not None
    ]
    strict_items = [
        item
        for item in graded
        if (
            item.get("execution_success") is True
            and item.get("answer_correctness") == "PASS"
            and item.get("evidence_correct") is True
            and item.get("provenance_correct") is True
            and (item.get("not_found_correct") is not False)
        )
    ]

    def rate(count: int, total: int) -> float | None:
        return count / total if total else None

    def difficulty_summary(
        difficulty: str,
    ) -> dict[str, Any]:
        subset = [item for item in graded if item.get("difficulty") == difficulty]
        subset_statuses = Counter(item["answer_correctness"] for item in subset)
        subset_strict = sum(item in strict_items for item in subset)
        return {
            "total": len(subset),
            "execution_success": {
                "count": sum(item.get("execution_success") is True for item in subset),
                "rate": rate(
                    sum(item.get("execution_success") is True for item in subset),
                    len(subset),
                ),
            },
            "answer_correctness": {
                status: subset_statuses.get(status, 0)
                for status in ANSWER_STATUSES
            },
            "strict_correct": {
                "count": subset_strict,
                "rate": rate(subset_strict, len(subset)),
            },
        }

    latency_values = sorted(
        float(item["latency_sec"])
        for item in graded
        if isinstance(item.get("latency_sec"), (int, float))
        and math.isfinite(float(item["latency_sec"]))
    )
    p95_rank = math.ceil(0.95 * len(latency_values)) if latency_values else None
    p95_latency = latency_values[p95_rank - 1] if p95_rank else None
    failure_ids_by_stage: dict[str, list[str]] = {}
    for item in graded:
        stage = item.get("earliest_failure_stage")
        if stage:
            failure_ids_by_stage.setdefault(stage, []).append(str(item["id"]))
    return graded, {
        "total_questions": len(gold_questions),
        "execution_success": {
            "count": sum(item.get("execution_success") is True for item in graded),
            "total": len(gold_questions),
            "rate": rate(
                sum(item.get("execution_success") is True for item in graded),
                len(gold_questions),
            ),
        },
        "route_match": {
            "matched": sum(item.get("route_match") is True for item in route_applicable),
            "applicable": len(route_applicable),
            "rate": rate(
                sum(item.get("route_match") is True for item in route_applicable),
                len(route_applicable),
            ),
        },
        "answer_correctness_counts": {status: statuses.get(status, 0) for status in ANSWER_STATUSES},
        "strict_correct": {
            "count": len(strict_items),
            "rate": rate(len(strict_items), len(gold_questions)),
        },
        "provenance_correct": {
            "count": sum(item.get("provenance_correct") is True for item in provenance_items),
            "total": len(provenance_items),
            "rate": rate(
                sum(item.get("provenance_correct") is True for item in provenance_items),
                len(provenance_items),
            ),
        },
        "evidence_correct": {
            "count": sum(item.get("evidence_correct") is True for item in evidence_items),
            "total": len(evidence_items),
            "rate": rate(
                sum(item.get("evidence_correct") is True for item in evidence_items),
                len(evidence_items),
            ),
        },
        "not_found_correct": {
            "count": sum(item.get("not_found_correct") is True for item in not_found_items),
            "total": len(not_found_items),
            "rate": rate(
                sum(item.get("not_found_correct") is True for item in not_found_items),
                len(not_found_items),
            ),
        },
        "unknown_exact_fallback": {
            "count": sum(
                item.get("type") == "unknown"
                and isinstance(item.get("final_answer"), str)
                and item["final_answer"] == EMPTY_ANSWER
                for item in graded
            ),
            "total": len(unknown),
        },
        "semantic_evidence_coverage": {
            "matched": semantic_matched,
            "expected": semantic_expected,
            "ratio": semantic_matched / semantic_expected if semantic_expected else None,
            "questions": len(semantic_items),
        },
        "model_retry_count": sum(int(item.get("model_retry_count") or 0) for item in graded),
        "errors": sum(bool(item.get("error")) for item in graded),
        "error_count": sum(bool(item.get("error")) for item in graded),
        "by_difficulty": {
            difficulty: difficulty_summary(difficulty)
            for difficulty in ("easy", "medium", "hard")
        },
        "latency_avg_sec": (
            sum(latency_values) / len(latency_values) if latency_values else None
        ),
        "latency_p95_sec": p95_latency,
        "latency_p95_method": (
            "nearest-rank: sorted finite latency samples, rank=ceil(0.95*n)"
        ),
        "failure_ids_by_earliest_stage": failure_ids_by_stage,
        "graded_execution_success_count": len(executable),
    }


def _split_pipeline_result(
    route: str,
    result: Any,
) -> tuple[Any, Any]:
    if route == "structured":
        return result, None
    if route == "semantic":
        return None, result
    return result["structured"], result["semantic"]


def _evaluate_question(
    gold: Mapping[str, Any],
    gemini_callable: Any,
    db_path: Path,
) -> dict[str, Any]:
    gemini_callable.begin_question()
    started_at = time.perf_counter()
    actual_route: str | None = route_question(gold["question"])
    structured_result: Any = None
    semantic_results: Any = None
    final_answer: str | None = None
    error: str | None = None
    execution_success = False
    synthesis_call_count = 0

    def capture_answer(prompt: str) -> str:
        nonlocal synthesis_call_count
        synthesis_call_count += 1
        return gemini_callable(prompt)

    try:
        response = ask(
            db_path,
            gold["question"],
            structured_model_callable=gemini_callable,
            top_k=TOP_K,
        )
        actual_route = response["route"]
        structured_result, semantic_results = _split_pipeline_result(
            actual_route,
            response["result"],
        )
        final_answer = answer_question(
            gold["question"],
            actual_route,
            structured_result=structured_result,
            semantic_chunks=semantic_results,
            answer_model_callable=capture_answer,
        )
        execution_success = True
    except Exception as exc:  # noqa: BLE001 - capture per-question failures
        error = f"{type(exc).__name__}: {exc}"

    gold_type = gold["type"]
    route_match = (
        actual_route == gold_type if gold_type in ROUTES and actual_route else None
    )
    record = {
        "id": gold["id"],
        "type": gold_type,
        "difficulty": gold.get("difficulty"),
        "question": gold["question"],
        "expected": gold["expected"],
        "actual_route": actual_route,
        "route_match": route_match,
        "structured_result": _json_safe(structured_result),
        "semantic_results": _json_safe(semantic_results),
        "final_answer": final_answer,
        "execution_success": execution_success,
        "error": error,
        "top_k": TOP_K,
        "model_retry_count": gemini_callable.retry_count + max(synthesis_call_count - 1, 0),
        "latency_sec": time.perf_counter() - started_at,
    }
    return _grade_result(gold, record)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Gold Questions through the actual curriculum RAG pipeline."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="run only the first N Gold Questions",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval_result.json"),
        metavar="PATH",
        help="output JSON path (default: eval_result.json)",
    )
    parser.add_argument(
        "--gold",
        type=Path,
        default=GOLD_QUESTIONS_PATH,
        metavar="PATH",
        help="Gold Questions JSON path",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=CURRICULUM_DB_PATH,
        metavar="PATH",
        help="curriculum SQLite database path",
    )
    parser.add_argument(
        "--min-call-interval",
        type=float,
        default=DEFAULT_MIN_CALL_INTERVAL,
        metavar="SECONDS",
        help="minimum interval between Gemini calls (default: 5)",
    )
    parser.add_argument(
        "--grade-existing",
        type=Path,
        default=None,
        metavar="PATH",
        help="grade an existing evaluation JSON without running the RAG pipeline",
    )
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be non-negative")
    if not math.isfinite(args.min_call_interval) or args.min_call_interval < 0:
        parser.error("--min-call-interval must be a finite non-negative number")
    return args


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = _parse_args(argv)
    gold_questions = json.loads(args.gold.read_text(encoding="utf-8"))
    if args.limit is not None:
        gold_questions = gold_questions[: args.limit]

    if args.grade_existing is not None:
        existing = json.loads(args.grade_existing.read_text(encoding="utf-8"))
        existing_results = (
            existing.get("results", [])
            if isinstance(existing, Mapping)
            else existing
        )
        if not isinstance(existing_results, list):
            raise ValueError("Existing evaluation must contain a results list")
        merged_results = _merge_gold_with_runtime_results(
            gold_questions,
            existing_results,
        )
        results, summary = grade_results(gold_questions, merged_results)
    else:
        gemini_callable = _SharedGeminiThrottle(
            make_gemini_callable(), args.min_call_interval
        )
        results = [
            _evaluate_question(gold, gemini_callable, args.db) for gold in gold_questions
        ]
        summary = {
            **grade_results(gold_questions, results)[1],
        }
    _write_output_atomically(
        args.output,
        {"results": results, "summary": summary},
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
