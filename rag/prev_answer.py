"""Bounded previous-answer follow-up grammar and reference validation.

A follow-up (explain / source / rationale) refers only to the immediately
previous grounded answer. Retained state holds canonical references, never
assistant prose or factual values; every follow-up re-grounds from the
canonical authority before answering. No chat history, no sessions.
"""

from __future__ import annotations

from typing import Any

from rag.answer import is_fallback_like
from rag.policy.routing import POLICY_ROUTE_ALLOWLIST
from rag.query_spec import _COURSE_CODE_PATTERN

EXPLAIN_PREVIOUS = "explain"
SOURCE_PREVIOUS = "source"
RATIONALE_PREVIOUS = "rationale"

# Bounded follow-up-intent grammar only: finite substring anchors matched
# against whitespace-compacted text. No sentence patterns, no regex.
_EXPLAIN_ANCHORS = (
    "ขยายความ",
    "ขยายอีก",
    "อธิบายเพิ่ม",
    "อธิบายอีกครั้ง",
    "อธิบายอีกที",
    "หมายความว่า",
    "หมายถึง",
    "ความหมายคือ",
    "รายละเอียดเพิ่ม",
    "รายละเอียดอีก",
    "ลงรายละเอียด",
)
_SOURCE_ANCHORS = (
    "อ้างอิงจาก",
    "มาจากกฎ",
    "มาจากไหน",
    "ที่มา",
    "กฎข้อไหน",
    "กฎข้อใด",
    "แหล่งอ้างอิง",
    "อ้างอิงข้อไหน",
    "ข้อไหน",
    "ข้อใด",
)
_RATIONALE_ANCHORS = (
    "ทำไม",
    "เพราะเหตุใด",
    "เพราะอะไร",
    "เหตุผลคือ",
)
_FOLLOWUP_ANCHORS = tuple(
    sorted(
        set(_EXPLAIN_ANCHORS + _SOURCE_ANCHORS + _RATIONALE_ANCHORS),
        key=len,
    )
)
# Residues accepted by the bounded follow-up grammar after its intent anchor
# is removed. These are request particles, anaphoric references, or rationale
# continuations; any other text is an independent target and must use normal
# routing instead of inheriting the previous answer.
_NON_SUBSTANTIVE_FOLLOWUP_RESIDUES = frozenset(
    {
        "",
        "หน่อย",
        "ช่วย",
        "ขอ",
        "ช่วยหน่อย",
        "ขอหน่อย",
        "ข้อมูลนี้",
        "เรื่องนี้",
        "ที่บอกมาอะไร",
        "ที่กล่าวมาอะไร",
        "อันนี้อะไร",
        "ตรงนี้อะไร",
        "แบบนี้อะไร",
        "สรุปว่าอะไร",
        "อะไร",
        "นิด",
        "อีกนิด",
        "เอา",
        "มีไหม",
        "มีมั้ย",
        "ขอดู",
        "อยู่ใน",
        "ยังไง",
        "ถึงกำหนดแบบนี้",
        "กำหนดแบบนี้",
    }
)

_POLICY_KINDS = frozenset(
    set(POLICY_ROUTE_ALLOWLIST) | {"program_total_credits", "registration_compare"}
)
_PLAN_KEYS = frozenset({"coop", "no_coop", "default", "gened"})
_COURSE_OPERATIONS = frozenset(
    {"sum_credits", "placement", "describe", "prerequisite", "existence"}
)
_POLICY_SCOPE_KEYS = frozenset(
    {"route", "policy_kind", "program", "catalog_key", "plan", "amount", "evidence_ids"}
)
_COURSE_SCOPE_KEYS = frozenset(
    {"route", "course_code", "operations", "program", "catalog_key"}
)
_MAX_EVIDENCE_IDS = 50


def _compact(question: str) -> str:
    return "".join(str(question).casefold().split())


def _has_substance(spec: Any) -> bool:
    """Whether the current turn carries its own factual scope or target."""
    return bool(
        tuple(getattr(spec, "course_codes", ()) or ())
        or getattr(spec, "course_name", None) is not None
        or tuple(getattr(spec, "operations", ()) or ())
        or getattr(spec, "topic", None) is not None
        or tuple(getattr(spec, "plans", ()) or ())
        or tuple(getattr(spec, "years", ()) or ())
        or tuple(getattr(spec, "semesters", ()) or ())
        or getattr(spec, "program", None) is not None
        or getattr(spec, "category", None) is not None
        or getattr(spec, "credit_units", None) is not None
        or tuple(getattr(spec, "group_by", ()) or ())
        or bool(getattr(spec, "references_previous_result_set", False))
        or getattr(spec, "result_ordinal", None) is not None
    )


def _non_substantive_followup_residue(value: str) -> bool:
    compacted = "".join(value.split()).rstrip("?？!！")
    if compacted in _NON_SUBSTANTIVE_FOLLOWUP_RESIDUES:
        return True
    # Preserve the existing bounded rationale form with a stated duration,
    # without treating arbitrary numbers or subjects as follow-up residue.
    prefix = "ถึงกำหนด"
    if not compacted.startswith(prefix):
        return False
    duration = compacted[len(prefix) :]
    digits = len(duration) - len(duration.lstrip("0123456789"))
    return digits > 0 and duration[digits:] in {"วัน", "วันทำการ"}


def classify_prev_followup(question: str, spec: Any) -> str | None:
    """Classify a substance-free turn as explain/source/rationale, else None."""
    if not isinstance(question, str) or not question.strip():
        return None
    if _has_substance(spec):
        return None
    compacted = _compact(question)
    if any(anchor in compacted for anchor in _EXPLAIN_ANCHORS):
        kind = EXPLAIN_PREVIOUS
    elif any(anchor in compacted for anchor in _SOURCE_ANCHORS):
        kind = SOURCE_PREVIOUS
    elif any(anchor in compacted for anchor in _RATIONALE_ANCHORS):
        kind = RATIONALE_PREVIOUS
    else:
        return None

    residue = compacted
    for anchor in _FOLLOWUP_ANCHORS:
        residue = residue.replace(anchor, " ")
    if not _non_substantive_followup_residue(residue):
        return None
    return kind


def _is_rule_id(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    head, separator, tail = value.partition(":")
    if head != "rule" or not separator or not tail:
        return False
    return all(part.isdigit() and part for part in tail.split("."))


def _optional_scope_text(value: Any, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"last_answer.{field} must be a non-empty string or null")
    return value.strip()


def _check_active_scope(
    field: str, value: str | None, active: str | None
) -> None:
    if value is None or active is None:
        return
    if value.casefold() != active.strip().casefold():
        raise _StaleScope(f"last_answer.{field} is outside the active scope")


class _StaleScope(ValueError):
    """A well-formed reference that no longer matches the active scope."""


def _parse_evidence_ids(value: Any) -> list[str]:
    if not isinstance(value, list):
        raise ValueError("last_answer.evidence_ids must be a list")
    if len(value) > _MAX_EVIDENCE_IDS:
        raise ValueError("last_answer.evidence_ids exceeds the reference bound")
    for item in value:
        if not _is_rule_id(item):
            raise ValueError("last_answer.evidence_ids must contain rule references")
    seen: list[str] = []
    for item in value:
        if item not in seen:
            seen.append(item)
    return seen


def parse_last_answer(
    value: Any, *, program: str | None, catalog_key: str | None
) -> dict[str, Any] | None:
    """Validate client-held previous-answer references.

    Returns a normalized reference, None when a well-formed reference is
    stale against the active scope, and raises on malformed input. Never
    retains prose or factual values.
    """
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("last_answer must be an object")
    route = value.get("route")
    if route == "policy":
        if set(value) - _POLICY_SCOPE_KEYS:
            raise ValueError("last_answer has unsupported fields")
        kind = value.get("policy_kind")
        if not isinstance(kind, str) or kind not in _POLICY_KINDS:
            raise ValueError("last_answer.policy_kind is not a supported policy kind")
        ref_program = _optional_scope_text(value.get("program"), "program")
        ref_catalog = _optional_scope_text(value.get("catalog_key"), "catalog_key")
        if ref_catalog is not None and len(ref_catalog) > 128:
            raise ValueError("last_answer.catalog_key is too long")
        plan = value.get("plan")
        if plan is not None and (not isinstance(plan, str) or plan not in _PLAN_KEYS):
            raise ValueError("last_answer.plan must be a canonical plan or null")
        amount = value.get("amount")
        if amount is not None and (
            not isinstance(amount, int) or isinstance(amount, bool) or amount < 0
        ):
            raise ValueError("last_answer.amount must be a non-negative integer")
        evidence_ids = (
            _parse_evidence_ids(value["evidence_ids"])
            if "evidence_ids" in value
            else []
        )
        try:
            _check_active_scope("program", ref_program, program)
            _check_active_scope("catalog_key", ref_catalog, catalog_key)
        except _StaleScope:
            return None
        parsed: dict[str, Any] = {"route": "policy", "policy_kind": kind}
        if ref_program is not None:
            parsed["program"] = ref_program
        if ref_catalog is not None:
            parsed["catalog_key"] = ref_catalog
        if plan is not None:
            parsed["plan"] = plan
        if amount is not None:
            parsed["amount"] = amount
        if evidence_ids:
            parsed["evidence_ids"] = evidence_ids
        return parsed
    if route == "course":
        if set(value) - _COURSE_SCOPE_KEYS:
            raise ValueError("last_answer has unsupported fields")
        code = value.get("course_code")
        if (
            not isinstance(code, str)
            or _COURSE_CODE_PATTERN.fullmatch(code.strip()) is None
        ):
            raise ValueError("last_answer.course_code must be a canonical course code")
        operations = value.get("operations")
        if (
            not isinstance(operations, (list, tuple))
            or not operations
            or any(
                not isinstance(item, str) or item not in _COURSE_OPERATIONS
                for item in operations
            )
        ):
            raise ValueError("last_answer.operations must be bounded course operations")
        ref_program = _optional_scope_text(value.get("program"), "program")
        ref_catalog = _optional_scope_text(value.get("catalog_key"), "catalog_key")
        if ref_catalog is not None and len(ref_catalog) > 128:
            raise ValueError("last_answer.catalog_key is too long")
        try:
            _check_active_scope("program", ref_program, program)
            _check_active_scope("catalog_key", ref_catalog, catalog_key)
        except _StaleScope:
            return None
        parsed = {
            "route": "course",
            "course_code": code.strip(),
            "operations": [item for item in operations],
        }
        if ref_program is not None:
            parsed["program"] = ref_program
        if ref_catalog is not None:
            parsed["catalog_key"] = ref_catalog
        return parsed
    raise ValueError("last_answer.route must be policy or course")


EXPLAIN_PROMPT_HEADER = "EXPLAIN_GROUNDED_EVIDENCE"

RATIONALE_UNAVAILABLE_ANSWER = (
    "หลักฐานที่มีอยู่ระบุเพียงข้อกำหนด ไม่มีเหตุผลประกอบการกำหนด "
    "จึงไม่สามารถอธิบายเหตุผลได้"
)

_EFFECTIVE_QUESTION_TEMPLATES = {
    "sum_credits": "{code} กี่หน่วย",
    "placement": "{code} เรียนตอนไหน",
    "describe": "{code} เรียนเกี่ยวกับอะไร",
    "prerequisite": "{code} มีวิชาบังคับก่อนอะไร",
    "existence": "{code} มีในหลักสูตรไหม",
}


def effective_course_question(course_code: str, operations: Any) -> str | None:
    """Rebuild one canonical single-operation question from a course referent."""
    if not isinstance(course_code, str) or not course_code.strip():
        return None
    if not isinstance(operations, (list, tuple)) or len(operations) != 1:
        return None
    template = _EFFECTIVE_QUESTION_TEMPLATES.get(operations[0])
    if template is None:
        return None
    return template.format(code=course_code.strip())


def build_explain_prompt(evidence_json: str, followup: str) -> str:
    """Build an explanation prompt carrying reloaded canonical evidence only."""
    return "\n".join(
        (
            EXPLAIN_PROMPT_HEADER,
            "อธิบายหลักฐานต่อไปนี้เป็นภาษาไทยอย่างเข้าใจง่าย "
            "โดยใช้เฉพาะข้อเท็จจริงในหลักฐานเท่านั้น",
            "ห้ามแต่งเติมข้อเท็จจริง สร้างตัวเลข/รหัสวิชาใหม่ "
            "หรืออนุมานเหตุผลที่ไม่มีในหลักฐาน",
            "คงรหัสวิชา ชื่อวิชา และตัวเลขตามหลักฐานทุกประการ",
            f"คำขอ: {followup}",
            f"หลักฐาน: {evidence_json}",
            "คำอธิบาย:",
        )
    )


def _digit_runs(text: str) -> set[str]:
    runs: set[str] = set()
    current: list[str] = []
    for character in str(text):
        if character.isdigit():
            current.append(character)
        elif current:
            runs.add("".join(current))
            current = []
    if current:
        runs.add("".join(current))
    return runs


def explanation_output_valid(generated: Any, evidence_text: str) -> bool:
    """Accept only rewrites introducing no new codes or digit sequences."""
    if not isinstance(generated, str) or not generated.strip():
        return False
    if is_fallback_like(generated.strip()):
        return False
    generated_codes = {
        match.group(1) for match in _COURSE_CODE_PATTERN.finditer(generated)
    }
    evidence_codes = {
        match.group(1) for match in _COURSE_CODE_PATTERN.finditer(evidence_text)
    }
    if not generated_codes.issubset(evidence_codes):
        return False
    return _digit_runs(generated).issubset(_digit_runs(evidence_text))


def render_policy_source(answer: Any) -> str | None:
    """Render canonical rule sections from a freshly re-fetched policy answer."""
    rules = tuple(getattr(answer, "rules", ()) or ())
    sections = [
        str(rule.section_number).strip()
        for rule in rules
        if str(getattr(rule, "section_number", "")).strip()
    ]
    if sections:
        return "อ้างอิงจากข้อ " + ", ".join(sections)
    identifiers: list[str] = []
    for fact in tuple(getattr(answer, "facts", ()) or ()):
        source = getattr(fact, "source_rule_id", None)
        if _is_rule_id(source) and source not in identifiers:
            identifiers.append(source)
    if identifiers:
        return "อ้างอิงจาก " + ", ".join(identifiers)
    return None


def render_provenance_source(provenance: Any) -> str | None:
    """Render canonical document references from re-grounded provenance."""
    if not isinstance(provenance, (list, tuple)) or not provenance:
        return None
    lines: list[str] = []
    for reference in provenance:
        if not isinstance(reference, dict):
            return None
        filename = reference.get("source_filename")
        page = reference.get("source_page")
        if not isinstance(filename, str) or not filename:
            return None
        if isinstance(page, bool) or not isinstance(page, int):
            return None
        lines.append(f"อ้างอิงจาก {filename} หน้า {page}")
    return "\n".join(lines)


__all__ = [
    "EXPLAIN_PREVIOUS",
    "SOURCE_PREVIOUS",
    "RATIONALE_PREVIOUS",
    "EXPLAIN_PROMPT_HEADER",
    "RATIONALE_UNAVAILABLE_ANSWER",
    "build_explain_prompt",
    "classify_prev_followup",
    "effective_course_question",
    "explanation_output_valid",
    "parse_last_answer",
    "render_policy_source",
    "render_provenance_source",
]
