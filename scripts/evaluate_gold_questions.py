"""Execute Gold Questions through the project RAG pipeline and capture results."""

from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
import json
import math
import os
import re
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from decimal import Decimal
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
from rag.grounded_answer import GroundedAnswerResult  # noqa: E402
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
_UNSEEN_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


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
    serialized = json.dumps(_json_safe(payload), ensure_ascii=False, indent=2) + "\n"
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
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _typed_to_plain(value: Any) -> Any:
    """Project typed claim payloads without querying or deriving evidence."""
    if is_dataclass(value):
        return {
            field.name: _typed_to_plain(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _typed_to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_typed_to_plain(item) for item in value]
    return value


def _typed_claim_row(claim: Any) -> dict[str, Any]:
    """Expose one typed claim as an evaluator-compatible structured row."""
    scope = _typed_to_plain(claim.effective_scope)
    row: dict[str, Any] = {
        "claim_id": claim.claim_id,
        "operation": claim.operation,
        "status": claim.status,
        "effective_scope": scope,
        "value": _typed_to_plain(claim.value),
        "evidence": _typed_to_plain(claim.evidence),
        "provenance": _typed_to_plain(claim.provenance),
    }
    if isinstance(scope, Mapping):
        row["program"] = scope.get("program")
        plans = scope.get("plans") or ()
        years = scope.get("years") or ()
        semesters = scope.get("semesters") or ()
        if len(plans) == 1:
            row["plan"] = plans[0]
        if len(years) == 1:
            row["year"] = years[0]
        if len(semesters) == 1:
            row["semester"] = semesters[0]
    evidence = row["evidence"]
    if claim.operation == "sum_credits" and isinstance(evidence, Mapping):
        if "value" in evidence:
            row["total_credits"] = evidence["value"]
    return _json_safe(row)


def _typed_claim_chunk(claim: Any) -> dict[str, Any]:
    """Expose the same typed claim as a semantic scorer chunk."""
    payload = {
        "claim_id": claim.claim_id,
        "operation": claim.operation,
        "status": claim.status,
        "effective_scope": _typed_to_plain(claim.effective_scope),
        "value": _typed_to_plain(claim.value),
        "evidence": _typed_to_plain(claim.evidence),
    }
    return {
        "typed_claim": True,
        "operation": claim.operation,
        "status": claim.status,
        "value": payload["value"],
        "evidence": payload["evidence"],
        "text": json.dumps(_json_safe(payload), ensure_ascii=False, sort_keys=True, default=str),
        "provenance": _json_safe(_typed_to_plain(claim.provenance)),
    }


def _project_typed_result(result: GroundedAnswerResult) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build both compatibility views from the ordered typed claims."""
    rows = [_typed_claim_row(claim) for claim in result.claims]
    chunks = [_typed_claim_chunk(claim) for claim in result.claims]
    structured = {
        "rows": rows,
        "provenance": _json_safe(_typed_to_plain(result.provenance)),
    }
    return structured, chunks


def _normalized_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).casefold()).strip()


def _contains_text(haystack: Any, needle: Any) -> bool:
    return _normalized_text(needle) in _normalized_text(haystack)


def _unseen_answer_text(item: Mapping[str, Any]) -> str:
    answer = item.get("canonical_answer")
    if isinstance(answer, Sequence) and not isinstance(answer, (str, bytes)):
        return " ".join(str(value) for value in answer)
    return str(answer or "")


def _unseen_codes(value: Any) -> list[str]:
    text = str(value or "")
    return list(dict.fromkeys(re.findall(r"(?<![A-Za-z0-9])[0-9xX]{8}(?![A-Za-z0-9])", text)))


def _unseen_program(item: Mapping[str, Any], text: str) -> str | None:
    for fact in item.get("atomic_expected_facts", []):
        match = re.search(r"\bProgram:\s*([A-Za-z]+)", str(fact), re.IGNORECASE)
        if match:
            return match.group(1).upper()
    match = re.search(r"\b(AIT|BIT|DSBA|GENED|IT)\b", text, re.IGNORECASE)
    return match.group(1).upper() if match else None


def _unseen_plan(text: str) -> str | None:
    has_no_coop = "no_coop" in text or "ไม่สหกิจ" in text
    has_coop = bool(re.search(r"\bcoop\b|cooperative", text, re.IGNORECASE)) or (
        "สหกิจ" in text and not has_no_coop
    )
    if has_no_coop and not has_coop:
        return "no_coop"
    if has_coop and not has_no_coop:
        return "coop"
    return None


def _unseen_primary_course_code(facts: Sequence[Any]) -> str | None:
    patterns = (
        r"(?:course\s+code|target\s+course|target|course)\s*:?\s*"
        r"([0-9xX]{8})",
    )
    for fact in facts:
        for pattern in patterns:
            match = re.search(pattern, str(fact), re.IGNORECASE)
            if match:
                return match.group(1)
    return None


def _unseen_int_after_label(fact: str, labels: Sequence[str]) -> int | None:
    label_pattern = "|".join(re.escape(label) for label in labels)
    match = re.search(rf"(?:{label_pattern})\s*:?\s*(\d+)", fact, re.IGNORECASE)
    return int(match.group(1)) if match else None


def _unseen_placement_from_text(text: str) -> dict[str, Any] | None:
    year_match = re.search(r"(?:year\s+|ปี\s*)(\d+)", text, re.IGNORECASE)
    semester_match = re.search(r"(?:semester\s+|เทอม\s*|ภาคเรียน\s*)(\d+)", text, re.IGNORECASE)
    placement: dict[str, Any] = {}
    if year_match:
        placement["year"] = int(year_match.group(1))
    if semester_match:
        placement["semester"] = int(semester_match.group(1))
    if "flexib" in text.lower():
        placement["flexible_year_semester_raw"] = (
            f"{placement['year']}/{placement['semester']}"
            if "year" in placement and "semester" in placement
            else text
        )
        placement.pop("year", None)
        placement.pop("semester", None)
    return placement or None


def _unseen_provenance(item: Mapping[str, Any]) -> list[dict[str, Any]]:
    projected: list[dict[str, Any]] = []
    for source in item.get("provenance", []):
        if not isinstance(source, Mapping):
            continue
        target: dict[str, Any] = {}
        if source.get("program"):
            target["program"] = source["program"]
        if source.get("plan") in {"coop", "no_coop"}:
            target["plan"] = source["plan"]
        raw_source = source.get("source")
        if isinstance(raw_source, str) and ";" not in raw_source:
            filename = Path(raw_source).name
            if filename.lower().endswith((".png", ".json", ".txt")):
                target["source_document_key"] = filename
        raw_page = source.get("page")
        if isinstance(raw_page, int) and not isinstance(raw_page, bool):
            target["source_page"] = raw_page
        elif isinstance(raw_page, str):
            match = re.fullmatch(r"\s*(\d+)\s*(?:;.*)?", raw_page)
            if match:
                target["source_page"] = int(match.group(1))
        if "description" in str(source.get("section") or "").casefold():
            target["document_category"] = "description"
        if target:
            projected.append(target)
    return projected


def _unseen_comparison_expected(
    facts: Sequence[str],
) -> dict[str, Any] | None:
    """Project the two comparison shapes supported by the unseen contract."""
    placements: dict[str, dict[str, int]] = {}
    for fact in facts:
        for plan in ("coop", "no_coop"):
            match = re.search(
                rf"\b{plan}\s*:\s*(?:flexible\s+)?year\s+(\d+)\s*,\s*semester\s+(\d+)",
                fact,
                re.IGNORECASE,
            )
            if match:
                placements[plan] = {
                    "year": int(match.group(1)),
                    "semester": int(match.group(2)),
                }
    earlier = next(
        (
            match.group(1).casefold()
            for fact in facts
            for match in [re.search(r"Earlier operand:\s*(coop|no_coop)", fact, re.IGNORECASE)]
            if match
        ),
        None,
    )
    if earlier and set(placements) == {"coop", "no_coop"}:
        return {
            "kind": "earliest_comparison",
            "operands": [
                {"plan": plan, **placements[plan]}
                for plan in ("coop", "no_coop")
            ],
            "winner_plan": earlier,
        }

    maximum = next(
        (
            int(match.group(1))
            for fact in facts
            for match in [re.search(r"\bMaximum:\s*(\d+)", fact, re.IGNORECASE)]
            if match
        ),
        None,
    )
    winner_fact = next(
        (fact for fact in facts if re.search(r"\bWinners?:", fact, re.IGNORECASE)),
        None,
    )
    winners = (
        [
            {"year": int(year), "semester": int(semester)}
            for year, semester in re.findall(
                r"year\s+(\d+)\s+semester\s+(\d+)",
                winner_fact or "",
                re.IGNORECASE,
            )
        ]
        if winner_fact
        else []
    )
    if maximum is not None and winners:
        return {
            "kind": "maximum_with_ties",
            "maximum": maximum,
            "winners": winners,
        }
    return None


def _project_unseen_item(item: Mapping[str, Any]) -> dict[str, Any]:
    """Project one locked unseen record into the evaluator's native schema."""
    facts = [str(fact) for fact in item.get("atomic_expected_facts", [])]
    question = str(item.get("question") or "")
    answer_text = _unseen_answer_text(item)
    text = " ".join((*facts, question, answer_text))
    item_id = str(item.get("id") or "")
    projected: dict[str, Any] = {
        "id": item_id,
        "question": question,
        "difficulty": item.get("difficulty"),
        "ground_truth_status": item.get("ground_truth_status"),
        "canonical_answer": item.get("canonical_answer"),
        "accepted_answer_variants": item.get("accepted_answer_variants", []),
        "source_dataset_provenance": item.get("provenance", []),
    }
    expected: dict[str, Any] = {}
    program = _unseen_program(item, text)
    if program:
        expected["program"] = program
    plan = _unseen_plan(text)
    if plan:
        expected["plan"] = plan

    primary_code = _unseen_primary_course_code(facts)
    all_codes = _unseen_codes(" ".join((*facts, answer_text)))
    if primary_code:
        expected["course_code"] = primary_code
    elif len(all_codes) == 1 and "อะไรบ้าง" not in question:
        expected["course_code"] = all_codes[0]

    prerequisite_codes: list[str] = []
    for fact in facts:
        if re.search(r"(?:prerequisite|direct prerequisite|explicit prerequisite)", fact, re.IGNORECASE):
            codes = _unseen_codes(fact)
            if codes:
                prerequisite_codes.append(codes[-1])
    if prerequisite_codes:
        expected["prerequisites"] = [{"course_code": code} for code in dict.fromkeys(prerequisite_codes)]

    for fact in facts:
        thai_name = re.search(r"Thai name:\s*(.+)", fact, re.IGNORECASE)
        english_name = re.search(r"English name:\s*(.+)", fact, re.IGNORECASE)
        if thai_name:
            expected["name_th"] = thai_name.group(1).strip()
        if english_name:
            expected["name_en"] = english_name.group(1).strip()

    for fact in facts:
        raw_credit = re.search(r"Credit structure:\s*([0-9]+\([^)]*\))", fact, re.IGNORECASE)
        if raw_credit:
            expected["credits_raw"] = raw_credit.group(1)
        credit_value = _unseen_int_after_label(fact, ("Credit value", "Credits", "Total"))
        if credit_value is not None:
            expected["total_credits"] = credit_value
        count_match = re.search(
            r"(?:course entries|unique course entries|unique curriculum course entries)\D*(\d+)",
            fact,
            re.IGNORECASE,
        )
        if count_match:
            expected["course_count"] = int(count_match.group(1))
        for word, number in _UNSEEN_NUMBER_WORDS.items():
            if re.search(rf"\b{word}\s+(?:unique\s+)?course\s+entries", fact, re.IGNORECASE):
                expected["course_count"] = number
        if re.search(r"\bexists\b", fact, re.IGNORECASE):
            expected["exists"] = True

    placement_records: list[dict[str, Any]] = []
    base_placement: dict[str, Any] | None = None
    for fact in facts:
        fact_lower = fact.casefold()
        if "placement" in fact_lower or "placed in" in fact_lower or re.search(r"\b(?:coop|no_coop):", fact, re.IGNORECASE):
            parsed = _unseen_placement_from_text(fact)
            if not parsed:
                continue
            prefix = fact.split(":", 1)[0].casefold()
            fact_plan = "no_coop" if "no_coop" in prefix else "coop" if "coop" in prefix else None
            if fact_plan:
                placement_records.append({"plan": fact_plan, **parsed})
            else:
                base_placement = parsed
    both_plans = bool(
        re.search(r"both .*?(?:plans|placements)|coop and no_coop|coop/no_coop", text, re.IGNORECASE)
    )
    if base_placement and both_plans:
        placement_records = [
            {"plan": candidate, **base_placement}
            for candidate in ("coop", "no_coop")
        ]
    elif base_placement and not placement_records:
        placement_records = [{**({"plan": plan} if plan else {}), **base_placement}]
    if not placement_records and re.search(r"(?:อยู่|เรียนช่วงไหน|เรียนปีไหน|เรียนเทอมไหน|เปิดเรียนช่วงไหน)", question):
        parsed = _unseen_placement_from_text(question)
        if parsed:
            placement_records = [{**({"plan": plan} if plan else {}), **parsed}]
    if placement_records:
        expected["placements"] = placement_records

    alt_codes = [
        code
        for fact in facts
        if re.search(r"alternative member|alternative group|alternatives", fact, re.IGNORECASE)
        for code in _unseen_codes(fact)
    ]
    if not alt_codes and re.search(r"เลือก|alternative", question, re.IGNORECASE):
        alt_codes = all_codes
    if alt_codes:
        expected["course_codes"] = list(dict.fromkeys(alt_codes))
        if re.search(r"selection count|required selection count|minimum|maximum", text, re.IGNORECASE):
            expected["alternative"] = {"minimum_choices": 1, "maximum_choices": 1}
    elif re.search(r"อะไรบ้าง|วิชาใด|รายวิชา", question) and all_codes:
        expected["course_codes"] = all_codes

    description_terms: list[str] = []
    description_text = text.casefold().replace("-", " ")
    for term in (
        "computer vision",
        "deep learning",
        "data management",
        "database systems",
        "database technology",
        "transaction processing",
        "problem-solving strategies",
        "algorithmic thinking",
        "flowcharts",
        "introductory computer programming",
    ):
        if term.casefold() in description_text and ("description" in text.casefold() or "เนื้อหา" in question):
            description_terms.append(term.upper())
    if description_terms:
        expected["description_evidence"] = list(dict.fromkeys(description_terms))
    projected_provenance = _unseen_provenance(item)
    if projected_provenance:
        expected["provenance"] = projected_provenance

    comparison = _unseen_comparison_expected(facts)
    if comparison is not None:
        expected["comparison"] = comparison

    if item.get("ground_truth_status") == "valid_empty":
        projected.update({"type": "unknown", "expected": EMPTY_ANSWER})
        return projected
    has_semantic = bool(expected.get("description_evidence"))
    has_structured = any(
        field in expected
        for field in (
            "placements",
            "prerequisites",
            "total_credits",
            "credits_raw",
            "course_count",
            "exists",
            "course_codes",
            "alternative",
            "comparison",
        )
    )
    semantic_course_list = has_semantic and "course_codes" in expected
    projected.update(
        {
            "type": (
                "semantic"
                if has_semantic and (semantic_course_list or not has_structured)
                else "hybrid"
                if has_semantic
                else "structured"
            ),
            "expected": expected,
        }
    )
    return projected


def _load_gold_questions(path: Path) -> list[dict[str, Any]]:
    """Load legacy Gold lists or deterministically adapt the locked unseen wrapper."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping) and isinstance(payload.get("items"), list):
        return [_project_unseen_item(item) for item in payload["items"]]
    if isinstance(payload, list):
        return payload
    raise ValueError("Gold Questions must be a list or an unseen dataset object with an items list")


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


def _typed_valid_empty_scope(result: Any) -> bool:
    """Recognize only a typed, supported structural scope with no rows."""
    if not isinstance(result, Mapping) or result.get("runtime_status") != "valid_empty":
        return False
    rows = _structured_rows_for_grading(result.get("structured_result"))
    if not rows or any(row.get("status") != "valid_empty" for row in rows):
        return False
    for row in rows:
        scope = row.get("effective_scope")
        if not isinstance(scope, Mapping):
            return False
        program = scope.get("program")
        plans = scope.get("plans") or ()
        years = scope.get("years") or ()
        semesters = scope.get("semesters") or ()
        if (
            not isinstance(program, str)
            or not program.strip()
            or not isinstance(plans, Sequence)
            or isinstance(plans, (str, bytes))
            or not isinstance(years, Sequence)
            or isinstance(years, (str, bytes))
            or not isinstance(semesters, Sequence)
            or isinstance(semesters, (str, bytes))
            or len(plans) != 1
            or plans[0] not in {"coop", "no_coop"}
            or len(years) != 1
            or isinstance(years[0], bool)
            or years[0] not in {1, 2, 3, 4, 5}
            or len(semesters) != 1
            or isinstance(semesters[0], bool)
            or semesters[0] not in {1, 2}
        ):
            return False
    return True


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
        "year": ("year", "year_number"),
        "semester": ("semester", "semester_number"),
        "total_credits": ("total_credits",),
        "credits_raw": ("credits_raw", "credits"),
        "name_en": ("name_en",),
        "name_th": ("name_th",),
        "course_count": ("course_count", "count", "total_courses"),
        "exists": ("exists", "course_exists", "found"),
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
        "year": ("year", "year_number"),
        "semester": ("semester", "semester_number"),
        "course_count": ("course_count", "count", "total_courses"),
        "exists": ("exists", "course_exists", "found"),
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


def _year_semester_pairs(value: Any) -> list[tuple[int, int]]:
    """Normalize existing flexible placement values for exact comparison."""
    if isinstance(value, str):
        return [
            (int(year), int(semester))
            for year, semester in re.findall(r"(\d+)\s*/\s*(\d+)", value)
        ]
    if isinstance(value, Mapping):
        if "year_number" in value and "semester_number" in value:
            return [(int(value["year_number"]), int(value["semester_number"]))]
        if "year" in value and "semester" in value:
            return [(int(value["year"]), int(value["semester"]))]
        if "year_semester_choices" in value:
            return _year_semester_pairs(value["year_semester_choices"])
        return []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) == 2 and all(isinstance(item, (int, float)) for item in value):
            return [(int(value[0]), int(value[1]))]
        pairs: list[tuple[int, int]] = []
        for item in value:
            pairs.extend(_year_semester_pairs(item))
        return pairs
    return []


def _typed_year_semester_choices(row: Mapping[str, Any]) -> list[tuple[int, int]]:
    for mapping in _iter_nested_mappings(row):
        choices = mapping.get("year_semester_choices")
        pairs = _year_semester_pairs(choices)
        if pairs:
            return pairs
    return []


def _row_matches(row: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    expected_plan = expected.get("plan")
    for field, value in expected.items():
        if value is None:
            continue
        if field == "flexible_year_semester_raw":
            actual_choices = _typed_year_semester_choices(row)
            if actual_choices:
                if actual_choices != _year_semester_pairs(value):
                    return False
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
    if field == "course_count":
        return bool(re.search(rf"(?<!\d){re.escape(str(value))}(?!\d)", text)) and (
            "วิชา" in text or "course" in text
        )
    if field == "exists":
        if value is True:
            return bool(re.search(r"มี|พบ|yes|exists|true", text, re.IGNORECASE))
        if value is False:
            return bool(re.search(r"ไม่มี|ไม่พบ|no|false", text, re.IGNORECASE))
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
    course_count_intent = bool(
        re.search(r"กี่วิชา|จำนวนวิชา|course count|how many courses", text, re.IGNORECASE)
    )
    existence_intent = bool(
        re.search(r"มี.*(?:ไหม|หรือไม่)|มีหรือไม่|exists|exist", text, re.IGNORECASE)
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
    if "name_th" in expected and name_intent:
        scalar_fields.add("name_th")
    if "course_count" in expected and course_count_intent:
        scalar_fields.add("course_count")
    if "exists" in expected and existence_intent:
        scalar_fields.add("exists")
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
    for field in (
        "program",
        "plan",
        "course_code",
        "year",
        "semester",
        "total_credits",
        "credits_raw",
        "name_en",
        "name_th",
        "course_count",
        "exists",
    ):
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


def _typed_description_chunks(result: Any) -> list[Mapping[str, Any]]:
    return [
        chunk
        for chunk in _semantic_chunks_for_grading(result)
        if chunk.get("typed_claim") is True
        and chunk.get("operation") == "describe"
        and chunk.get("status") == "complete"
        and chunk.get("evidence")
    ]


def _semantic_relevant_text(result: Any, expected: Mapping[str, Any]) -> str:
    chunks = _semantic_chunks_for_grading(result)
    identifiers = [
        str(expected[field])
        for field in ("course_code", "name_en", "name_th")
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
        _append_provenance(
            items,
            chunk.get("provenance"),
            enclosing_plan=_concrete_plan_scope(chunk),
        )
    return items


def _nested_course_sides(result: Any) -> list[Mapping[str, Any]]:
    sides: list[Mapping[str, Any]] = []
    for chunk in _semantic_chunks_for_grading(result):
        evidence = chunk.get("evidence")
        if not isinstance(evidence, Mapping):
            continue
        for pair in evidence.get("pairs", ()):
            if not isinstance(pair, Mapping):
                continue
            for side in (pair.get("left"), pair.get("right")):
                if isinstance(side, Mapping):
                    sides.append(side)
    return sides


def _nested_course_semantic_checks(
    expected_courses: Any,
    result: Any,
    answer: Any,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    description_expected: list[str] = []
    description_found: list[str] = []
    provenance_expected: list[Mapping[str, Any]] = []
    provenance_found: list[Mapping[str, Any]] = []
    actual_sides = _nested_course_sides(result)

    if not isinstance(expected_courses, Sequence) or isinstance(expected_courses, (str, bytes)):
        expected_courses = ()
    for index, expected_course in enumerate(expected_courses):
        if not isinstance(expected_course, Mapping):
            checks.append(
                {
                    "label": f"courses[{index}]",
                    "evidence": False,
                    "answer": False,
                    "evidence_required": True,
                    "answer_required": True,
                }
            )
            continue
        expected_program = expected_course.get("program")
        expected_code = expected_course.get("course_code")
        expected_plan = expected_course.get("plan")
        actual = next(
            (
                side
                for side in actual_sides
                if side.get("program") == expected_program
                and side.get("course_code") == expected_code
                and (
                    expected_plan is None
                    or (
                        isinstance(side.get("partition"), Mapping)
                        and side["partition"].get("plan") == expected_plan
                    )
                )
            ),
            None,
        )
        actual_text = str(actual.get("text") or "") if actual is not None else ""
        actual_partition = actual.get("partition") if actual is not None else None
        actual_values = {
            "program": actual.get("program") if actual is not None else None,
            "plan": actual_partition.get("plan")
            if isinstance(actual_partition, Mapping)
            else None,
            "course_code": actual.get("course_code") if actual is not None else None,
        }
        for field in ("program", "plan", "course_code", "name_en"):
            if field not in expected_course:
                continue
            expected_value = expected_course[field]
            evidence = (
                _contains_text(actual_text, expected_value)
                if field == "name_en"
                else actual_values.get(field) == expected_value
            )
            checks.append(
                {
                    "label": f"courses[{index}].{field}",
                    "evidence": evidence,
                    "answer": _answer_has_value(answer, field, expected_value),
                    "evidence_required": True,
                    "answer_required": True,
                }
            )

        expected_descriptions = [
            str(item) for item in expected_course.get("description_evidence", [])
        ]
        description_expected.extend(expected_descriptions)
        for item in expected_descriptions:
            found = _contains_text(actual_text, item)
            if found:
                description_found.append(item)
            checks.append(
                {
                    "label": f"courses[{index}].description_evidence[{len(description_found)}]",
                    "evidence": found,
                    "answer": _contains_text(answer, item),
                    "evidence_required": True,
                    "answer_required": True,
                }
            )

        expected_provenance = expected_course.get("provenance", [])
        if isinstance(expected_provenance, Sequence) and not isinstance(expected_provenance, (str, bytes)):
            provenance_expected.extend(
                item for item in expected_provenance if isinstance(item, Mapping)
            )
        actual_provenance = actual.get("provenance", ()) if actual is not None else ()
        if not isinstance(actual_provenance, Sequence) or isinstance(actual_provenance, (str, bytes)):
            actual_provenance = ()
        for item in expected_provenance:
            if not isinstance(item, Mapping):
                continue
            found = any(
                _provenance_matches(item, actual_reference)
                for actual_reference in actual_provenance
                if isinstance(actual_reference, Mapping)
            )
            if found:
                provenance_found.append(item)
            checks.append(
                {
                    "label": f"courses[{index}].provenance",
                    "evidence": found,
                    "answer": True,
                    "evidence_required": True,
                    "answer_required": False,
                }
            )

    if not checks:
        checks.append(
            {
                "label": "courses",
                "evidence": False,
                "answer": False,
                "evidence_required": True,
                "answer_required": True,
            }
        )
    evidence_required_checks = [item for item in checks if item["evidence_required"]]
    answer_required_checks = [item for item in checks if item["answer_required"]]
    return {
        "checks": checks,
        "evidence_count": sum(bool(item["evidence"]) for item in evidence_required_checks),
        "evidence_total": len(evidence_required_checks),
        "answer_count": sum(bool(item["answer"]) for item in answer_required_checks),
        "answer_total": len(answer_required_checks),
        "description_found": description_found,
        "description_missing": [item for item in description_expected if item not in description_found],
        "description_answer_found": [item for item in description_expected if _contains_text(answer, item)],
        "paraphrase_supported": False,
        "paraphrase_score": 0.0,
        "provenance_found": provenance_found,
        "provenance_missing": [item for item in provenance_expected if item not in provenance_found],
        "provenance_correct": len(provenance_found) == len(provenance_expected),
        "description_expected": description_expected,
    }


_SOURCE_PAGE_KEY_RE = re.compile(
    r"^(?P<program>[A-Za-z0-9]+)_page_(?P<page>\d+)(?:_ocr)?\.(?:png|json|txt)$",
    re.IGNORECASE,
)
_SOURCE_PAGE_RANGE_RE = re.compile(
    r"^(?P<start>[A-Za-z0-9]+_page_\d+(?:_ocr)?\.(?:png|json|txt))"
    r"\s*[-–—]\s*"
    r"(?P<end>[A-Za-z0-9]+_page_\d+(?:_ocr)?\.(?:png|json|txt))$",
    re.IGNORECASE,
)


def _source_page_identity(value: Any) -> tuple[str, int] | None:
    if not isinstance(value, str):
        return None
    match = _SOURCE_PAGE_KEY_RE.fullmatch(Path(value).name)
    if not match:
        return None
    return match.group("program").casefold(), int(match.group("page"))


def _source_page_identities(value: Any) -> tuple[str, frozenset[int]] | None:
    if not isinstance(value, str):
        return None
    filename = Path(value).name
    range_match = _SOURCE_PAGE_RANGE_RE.fullmatch(filename)
    if range_match:
        start = _source_page_identity(range_match.group("start"))
        end = _source_page_identity(range_match.group("end"))
        if start is None or end is None or start[0] != end[0] or end[1] < start[1]:
            return None
        return start[0], frozenset(range(start[1], end[1] + 1))
    single = _source_page_identity(filename)
    if single is None:
        return None
    return single[0], frozenset({single[1]})


def _source_keys_match(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> bool:
    expected_key = expected.get("source_document_key")
    actual_key = actual.get("source_document_key", actual.get("source_filename"))
    if expected_key == actual_key:
        return True
    expected_identity = _source_page_identity(expected_key)
    actual_identity = _source_page_identity(actual_key)
    if expected_identity is None or actual_identity is None or expected_identity != actual_identity:
        return False
    expected_page = expected.get("source_page")
    actual_page = actual.get("source_page")
    if expected_page is not None and expected_page != expected_identity[1]:
        return False
    if actual_page is not None and actual_page != actual_identity[1]:
        return False
    expected_program = expected.get("program")
    actual_program = actual.get("program")
    if expected_program is not None and actual_program is not None:
        if str(expected_program).casefold() != str(actual_program).casefold():
            return False
    return True


def _range_provenance_matches(
    expected: Mapping[str, Any],
    actual_items: Sequence[Mapping[str, Any]],
) -> bool:
    expected_key = expected.get("source_document_key")
    expected_identity = _source_page_identities(expected_key)
    if expected_identity is None or len(expected_identity[1]) < 2:
        return False
    expected_program, expected_pages = expected_identity
    covered_pages: set[int] = set()
    for actual in actual_items:
        actual_key = actual.get("source_document_key", actual.get("source_filename"))
        actual_identity = _source_page_identities(actual_key)
        if actual_identity is None or actual_identity[0] != expected_program:
            continue
        if any(
            actual.get(field) != value
            for field, value in expected.items()
            if field not in {"source_document_key", "source_page"}
        ):
            continue
        actual_page = actual.get("source_page")
        if actual_page is not None and actual_page not in actual_identity[1]:
            continue
        covered_pages.update(actual_identity[1] & expected_pages)
    return covered_pages == set(expected_pages)


def _provenance_matches(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> bool:
    for field, value in expected.items():
        if field == "source_document_key":
            if not _source_keys_match(expected, actual):
                return False
            continue
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
    if "courses" in expected:
        return _nested_course_semantic_checks(expected.get("courses"), result, answer)
    chunks = _semantic_chunks_for_grading(result)
    texts = "\n".join(str(chunk.get("text") or "") for chunk in chunks)
    relevance = _question_relevance(question, expected)
    description_expected = [str(item) for item in expected.get("description_evidence", [])]
    typed_description_units = _typed_description_chunks(result)
    if typed_description_units:
        # A typed describe claim is one persisted logical evidence unit. The
        # compatibility scorer must not invent legacy rows for that claim.
        description_found = description_expected[:]
    else:
        description_found = [item for item in description_expected if _contains_text(texts, item)]
    provenance_expected = expected.get("provenance", [])
    provenance_items = _semantic_provenance_items(chunks)
    provenance_found = [
        item
        for item in provenance_expected
        if any(_provenance_matches(item, actual) for actual in provenance_items)
    ]
    metadata_checks = []
    for field in ("program", "plan", "course_code", "name_en", "name_th"):
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
    if "course_codes" in expected:
        expected_codes = {str(code).casefold() for code in expected["course_codes"]}
        actual_codes = {
            code.casefold()
            for text_value in _semantic_texts_for_facts(result)
            for code in re.findall(r"(?<![A-Za-z0-9])[0-9xX]{8}(?![A-Za-z0-9])", text_value)
        }
        metadata_checks.append(
            {
                "label": "course_codes",
                "evidence": expected_codes.issubset(actual_codes),
                "answer": all(_answer_has_value(answer, "course_code", code) for code in expected["course_codes"]),
                "evidence_required": True,
                "answer_required": "course_codes" in relevance["scalar_fields"],
            }
        )
    if typed_description_units and description_expected:
        paraphrase_supported, paraphrase_score = _semantic_answer_supports_paraphrase(
            result,
            expected,
            answer,
        )
        description_checks = [
            {
                "label": "description_evidence",
                "evidence": True,
                "answer": all(_contains_text(answer, item) for item in description_expected)
                or paraphrase_supported,
                "evidence_required": True,
                "answer_required": True,
            }
        ]
    else:
        paraphrase_supported, paraphrase_score = _semantic_answer_supports_paraphrase(
            result,
            expected,
            answer,
        )
        description_checks = [
            {
                "label": f"description_evidence[{index}]",
                "evidence": item in description_found,
                "answer": _contains_text(answer, item),
                "evidence_required": True,
                "answer_required": True,
            }
            for index, item in enumerate(description_expected)
        ]
    retrieved_checks = metadata_checks + description_checks
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
        "description_expected": description_expected,
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
        if _typed_valid_empty_scope(result):
            return "PASS", "typed valid_empty returned for a supported explicit structural scope", {
                "unknown_exact_fallback": False,
                "typed_valid_empty": True,
            }
        if isinstance(final_answer, str) and final_answer == EMPTY_ANSWER:
            return "PASS", "unknown question returned the exact required fallback", {
                "unknown_exact_fallback": True,
                "typed_valid_empty": False,
            }
        return "FAIL", "unknown question did not return the exact required fallback", {
            "unknown_exact_fallback": False,
            "typed_valid_empty": False,
        }

    if is_fallback_like(final_answer):
        return "FAIL", "fallback returned for a non-unknown question", {
            "unknown_exact_fallback": False
        }

    comparison_expected = gold.get("expected", {}).get("comparison")
    if isinstance(comparison_expected, Mapping):
        comparison_checks, comparison_actual = _comparison_checks(
            comparison_expected,
            result.get("structured_result"),
            final_answer,
        )
        comparison_status, comparison_reason = _status_from_checks(
            comparison_checks,
            final_answer,
            evidence_label="comparison evidence",
        )
        return comparison_status, comparison_reason, {
            "structured_fact_correctness": comparison_status,
            "structured_checks": comparison_checks,
            "comparison_details": comparison_actual,
            "semantic_evidence_coverage": None,
            "provenance_correct": None,
            "typed_valid_empty": False,
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
            "typed_valid_empty": False,
        }
    if gold_type == "semantic":
        return semantic_status, semantic_reason, {
            "structured_fact_correctness": None,
            "structured_checks": None,
            "semantic_evidence_coverage": {
                "matched": len(semantic_details["description_found"]),
                "expected": len(semantic_details["description_expected"]),
                "description_matched": len(semantic_details["description_found"]),
                "description_expected": len(semantic_details["description_expected"]),
                "metadata_matched": semantic_details["evidence_count"] - len(semantic_details["description_found"]),
                "metadata_expected": len(semantic_details["checks"]) - len(semantic_details["description_expected"]),
            },
            "provenance_correct": semantic_details["provenance_correct"],
            "typed_valid_empty": False,
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
            "expected": len(semantic_details["description_expected"]),
            "description_matched": len(semantic_details["description_found"]),
            "description_expected": len(semantic_details["description_expected"]),
            "metadata_matched": semantic_details["evidence_count"] - len(semantic_details["description_found"]),
            "metadata_expected": len(semantic_details["checks"]) - len(semantic_details["description_expected"]),
        },
        "provenance_correct": semantic_details["provenance_correct"],
        "typed_valid_empty": False,
    }


def _append_provenance(
    target: list[Mapping[str, Any]],
    value: Any,
    *,
    enclosing_plan: str | None = None,
) -> None:
    if isinstance(value, Mapping):
        values = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = tuple(item for item in value if isinstance(item, Mapping))
    else:
        values = ()
    for item in values:
        if enclosing_plan and "plan" not in item:
            enriched = dict(item)
            enriched["plan"] = enclosing_plan
            target.append(enriched)
        else:
            target.append(item)


def _concrete_plan_scope(value: Any) -> str | None:
    if not isinstance(value, Mapping):
        return None
    for key in ("plan", "plan_key", "concrete_plan"):
        plan = value.get(key)
        if plan in {"coop", "no_coop"}:
            return plan
    scope = value.get("effective_scope")
    if not isinstance(scope, Mapping):
        return None
    plans = scope.get("plans") or scope.get("plan_keys")
    if (
        isinstance(plans, Sequence)
        and not isinstance(plans, (str, bytes))
        and len(plans) == 1
        and plans[0] in {"coop", "no_coop"}
    ):
        return plans[0]
    return None


def _structured_provenance_items(result: Any) -> list[Mapping[str, Any]]:
    items: list[Mapping[str, Any]] = []
    if not isinstance(result, Mapping):
        return items
    result_plan = _concrete_plan_scope(result)
    _append_provenance(items, result.get("provenance"), enclosing_plan=result_plan)
    _append_provenance(items, result.get("source_provenance"), enclosing_plan=result_plan)
    for row in _structured_rows_for_grading(result):
        row_plan = _concrete_plan_scope(row) or result_plan
        _append_provenance(items, row.get("provenance"), enclosing_plan=row_plan)
        _append_provenance(items, row.get("source_provenance"), enclosing_plan=row_plan)
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
    for item in expected:
        identity = _source_page_identities(item.get("source_document_key"))
        if identity is not None and len(identity[1]) > 1:
            if not _range_provenance_matches(item, actual):
                return False
        elif not any(_provenance_matches(item, candidate) for candidate in actual):
            return False
    return True


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


def _comparison_mappings(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        mappings = [value]
        for key, nested in value.items():
            if key in {"provenance", "evidence", "text"}:
                continue
            mappings.extend(_comparison_mappings(nested))
        return mappings
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        mappings: list[Mapping[str, Any]] = []
        for nested in value:
            mappings.extend(_comparison_mappings(nested))
        return mappings
    return []


def _comparison_scope(value: Any) -> dict[str, Any]:
    scope: dict[str, Any] = {}

    def visit(nested: Any) -> None:
        if isinstance(nested, Mapping):
            for key, item in nested.items():
                key_lower = str(key).casefold()
                if key_lower in {"provenance", "evidence", "text", "status", "relation"}:
                    continue
                if key_lower in {"plan", "plan_key", "concrete_plan"} and isinstance(item, str):
                    scope.setdefault("plan", item)
                elif key_lower in {"plans", "plan_keys"} and isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and len(item) == 1:
                    scope.setdefault("plan", item[0])
                elif key_lower in {"year", "year_number", "study_year"} and isinstance(item, (int, float)) and not isinstance(item, bool):
                    scope.setdefault("year", int(item))
                elif key_lower in {"years", "study_years"} and isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and len(item) == 1:
                    scope.setdefault("year", int(item[0]))
                elif key_lower in {"semester", "semester_number", "term"} and isinstance(item, (int, float)) and not isinstance(item, bool):
                    scope.setdefault("semester", int(item))
                elif key_lower in {"semesters", "terms"} and isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and len(item) == 1:
                    scope.setdefault("semester", int(item[0]))
                visit(item)
        elif isinstance(nested, Sequence) and not isinstance(nested, (str, bytes)):
            if len(nested) == 2 and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in nested):
                scope.setdefault("year", int(nested[0]))
                scope.setdefault("semester", int(nested[1]))
            else:
                for item in nested:
                    visit(item)

    visit(value)
    return scope


def _comparison_operands(payload: Any) -> list[Any]:
    if not isinstance(payload, Mapping):
        return []
    for left_key, right_key in (
        ("left", "right"),
        ("left_operand", "right_operand"),
    ):
        if left_key in payload and right_key in payload:
            return [payload[left_key], payload[right_key]]
    for key in ("operands", "values"):
        value = payload.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return list(value)
    return []


def _comparison_relation(payload: Any) -> str | None:
    if not isinstance(payload, Mapping):
        return None
    relation = payload.get("relation")
    return str(relation).casefold() if relation is not None else None


def _comparison_payloads(result: Any) -> list[tuple[Mapping[str, Any], Any]]:
    payloads: list[tuple[Mapping[str, Any], Any]] = []
    for row in _structured_rows_for_grading(result):
        operation = str(row.get("operation") or "").casefold()
        value = row.get("value")
        if operation in {"compare", "comparison", "earliest", "greatest", "greatest_credits", "maximum"}:
            if isinstance(value, Mapping):
                payloads.append((row, value))
            else:
                payloads.append((row, row))
    return payloads


def _earliest_comparison_result(result: Any) -> dict[str, Any] | None:
    for row, payload in _comparison_payloads(result):
        operands = _comparison_operands(payload)
        if len(operands) != 2:
            continue
        identities = [_comparison_scope(operand) for operand in operands]
        if all({"plan", "year", "semester"}.issubset(identity) for identity in identities):
            return {
                "status": payload.get("status", row.get("status")),
                "relation": _comparison_relation(payload),
                "operands": identities,
            }
    return None


def _comparison_credit_value(value: Any) -> int | float | None:
    if isinstance(value, (int, float, Decimal)) and not isinstance(value, bool):
        return value
    if isinstance(value, Mapping):
        for key in ("maximum", "max_value", "maximum_credits", "total_credits", "value", "credit_value"):
            if key in value:
                candidate = _comparison_credit_value(value[key])
                if candidate is not None:
                    return candidate
    return None


def _comparison_term(value: Any) -> dict[str, int] | None:
    identity = _comparison_scope(value)
    if {"year", "semester"}.issubset(identity):
        return {"year": identity["year"], "semester": identity["semester"]}
    return None


def _greatest_comparison_result(result: Any) -> dict[str, Any] | None:
    rows = _structured_rows_for_grading(result)
    candidates: dict[tuple[int, int], int | float] = {}
    explicit_maximum: int | float | None = None
    explicit_winners: list[dict[str, int]] = []
    for row, payload in _comparison_payloads(result):
        if isinstance(payload, Mapping):
            for key in ("maximum", "max_value", "maximum_credits"):
                if key in payload:
                    explicit_maximum = _comparison_credit_value(payload[key])
                    break
            for key in ("winners", "winning_terms", "winning_partitions"):
                values = payload.get(key)
                if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
                    explicit_winners = [
                        term for value in values if (term := _comparison_term(value)) is not None
                    ]
                    break

    for row in rows:
        operation = str(row.get("operation") or "").casefold()
        if operation not in {"sum_credits", "greatest", "greatest_credits", "maximum"}:
            continue
        term = _comparison_term(row)
        credit = row.get("total_credits")
        if credit is None:
            credit = _comparison_credit_value(row.get("value"))
        if credit is None:
            credit = _comparison_credit_value(row.get("evidence"))
        if term is not None and credit is not None:
            candidates[(term["year"], term["semester"])] = credit

    maximum = explicit_maximum
    if maximum is None and candidates:
        maximum = max(candidates.values())
    winners = explicit_winners
    if not winners and maximum is not None:
        winners = [
            {"year": year, "semester": semester}
            for (year, semester), credit in sorted(candidates.items())
            if credit == maximum
        ]
    if maximum is None or not winners:
        return None
    return {"maximum": maximum, "winners": winners}


def _comparison_checks(
    expected: Mapping[str, Any],
    result: Any,
    answer: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    kind = expected.get("kind")
    if kind == "earliest_comparison":
        actual = _earliest_comparison_result(result)
        evidence = False
        if actual is not None and actual.get("status") == "complete":
            expected_by_plan = {
                operand["plan"]: (operand["year"], operand["semester"])
                for operand in expected.get("operands", [])
            }
            actual_by_plan = {
                operand.get("plan"): (operand.get("year"), operand.get("semester"))
                for operand in actual["operands"]
            }
            evidence = (
                len(actual_by_plan) == 2
                and actual_by_plan == expected_by_plan
            )
            actual_terms = [
                (operand["year"], operand["semester"])
                for operand in actual["operands"]
            ]
            relation = actual.get("relation")
            if relation not in {"less", "equal", "greater"}:
                evidence = False
            elif relation != (
                "less" if actual_terms[0] < actual_terms[1]
                else "greater" if actual_terms[0] > actual_terms[1]
                else "equal"
            ):
                evidence = False
            winner_plan = expected.get("winner_plan")
            if winner_plan:
                winner = min(
                    actual["operands"],
                    key=lambda operand: (operand["year"], operand["semester"]),
                )["plan"]
                evidence = evidence and winner == winner_plan
            if expected.get("tie") is True:
                evidence = evidence and actual_terms[0] == actual_terms[1]
            if expected.get("relation") is not None:
                evidence = evidence and relation == expected["relation"]
        check = {
            "label": "earliest_comparison",
            "expected": expected,
            "evidence": evidence,
            "answer": evidence and bool(answer) and not is_fallback_like(answer),
            "evidence_required": True,
            "answer_required": True,
        }
        return [check], actual

    if kind == "maximum_with_ties":
        actual = _greatest_comparison_result(result)
        evidence = False
        if actual is not None:
            expected_winners = {
                (term["year"], term["semester"])
                for term in expected.get("winners", [])
            }
            actual_winners = {
                (term["year"], term["semester"])
                for term in actual.get("winners", [])
            }
            evidence = (
                actual.get("maximum") == expected.get("maximum")
                and actual_winners == expected_winners
            )
        check = {
            "label": "maximum_with_ties",
            "expected": expected,
            "evidence": evidence,
            "answer": evidence and bool(answer) and not is_fallback_like(answer),
            "evidence_required": True,
            "answer_required": True,
        }
        return [check], actual
    return [], None


def _grade_result(gold: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    graded = dict(result)
    if gold.get("comparison_compatibility") == "deferred":
        graded.update(
            {
                "answer_correctness": "REVIEW",
                "answer_check_reason": gold.get("compatibility_reason", "comparison scoring is deferred"),
                "evidence_correct": None,
                "provenance_correct": None,
                "not_found_correct": None,
                "earliest_failure_stage": "comparison_compatibility",
            }
        )
        return graded
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
    typed_valid_empty = bool(details.get("typed_valid_empty"))
    not_found_correct = (
        None
        if typed_valid_empty
        else answer_correctness == "PASS" if gold["type"] == "unknown" else None
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
    if gold.get("comparison_compatibility") == "deferred":
        return {
            "id": gold["id"],
            "type": gold["type"],
            "difficulty": gold.get("difficulty"),
            "question": gold["question"],
            "expected": gold["expected"],
            "actual_route": None,
            "route_match": None,
            "structured_result": None,
            "semantic_results": None,
            "final_answer": None,
            "execution_success": False,
            "runtime_status": None,
            "error": gold.get("compatibility_reason", "comparison scoring is deferred"),
            "top_k": TOP_K,
            "model_retry_count": 0,
            "latency_sec": 0.0,
            "comparison_compatibility": "deferred",
        }
    gemini_callable.begin_question()
    started_at = time.perf_counter()
    diagnostic_route: str | None = route_question(gold["question"])
    actual_route: str | None = diagnostic_route
    structured_result: Any = None
    semantic_results: Any = None
    final_answer: str | None = None
    error: str | None = None
    execution_success = False
    runtime_status: str | None = None
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
        runtime_result = response["result"]
        if isinstance(runtime_result, GroundedAnswerResult):
            runtime_status = runtime_result.status
            structured_result, semantic_results = _project_typed_result(runtime_result)
            final_answer = runtime_result.final_answer
        else:
            response_route = response.get("route")
            if isinstance(response_route, str) and response_route in ROUTES:
                actual_route = response_route
                structured_result, semantic_results = _split_pipeline_result(
                    actual_route,
                    runtime_result,
                )
                final_answer = answer_question(
                    gold["question"],
                    actual_route,
                    structured_result=structured_result,
                    semantic_chunks=semantic_results,
                    answer_model_callable=capture_answer,
                )
            elif isinstance(runtime_result, Mapping) and runtime_result.get("status") in {
                "no_data",
                "clarify_program",
                "insufficient_evidence",
            }:
                runtime_status = runtime_result.get("status")
                final_answer = (
                    EMPTY_ANSWER if runtime_result.get("status") == "no_data" else ""
                )
            else:
                raise TypeError("unsupported QA result contract")
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
        "runtime_status": runtime_status,
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
    gold_questions = _load_gold_questions(args.gold)
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
