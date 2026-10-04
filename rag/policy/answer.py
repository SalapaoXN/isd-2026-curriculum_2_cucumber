"""Deterministic policy evaluation and rendering."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .query import PolicyQuery, parse_policy_question
from .repository import (
    fetch_policy_facts,
    fetch_program_requirement,
    fetch_regulation_rules,
)


DEFAULT_POLICY_DB_PATH = (
    Path(__file__).resolve().parents[2] / "cucumber_outputs" / "runtime" / "curriculum.db"
)
POLICY_STATUSES = ("complete", "unsupported", "insufficient_evidence", "invalid_query")


@dataclass(frozen=True, slots=True)
class PolicyFact:
    category: str
    fact_key: str | None
    value: Any
    unit: str | None
    operator: str
    condition: str | None
    context: str | None
    source_rule_id: str | None
    program: str | None
    verification_status: str | None
    provenance: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class PolicyRuleEvidence:
    rule_id: str
    section_number: str
    category: str
    rule_text: str
    provenance: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class PolicyAnswer:
    status: str
    query_type: str | None = None
    facts: tuple[PolicyFact, ...] = ()
    rules: tuple[PolicyRuleEvidence, ...] = ()
    value: Any = None
    unit: str | None = None
    operator: str | None = None
    condition: str | None = None
    context: str | None = None
    source_rule_id: str | None = None
    program: str | None = None
    provenance: tuple[dict[str, Any], ...] = ()
    verification_status: str | None = None
    rendered_answer: str = ""

    def __post_init__(self) -> None:
        if self.status not in POLICY_STATUSES:
            raise ValueError(f"unsupported policy status: {self.status!r}")


def _fact(record: dict[str, Any]) -> PolicyFact:
    references = tuple(record.get("provenance", ()))
    if not references:
        raise ValueError("policy fact has no provenance")
    return PolicyFact(
        category=record["category"],
        fact_key=record.get("fact_key"),
        value=record["value"],
        unit=record.get("unit"),
        operator=record["operator"],
        condition=record.get("condition"),
        context=record.get("context"),
        source_rule_id=record.get("source_rule_id"),
        program=record.get("program"),
        verification_status=record.get("verification_status"),
        provenance=references,
    )


_TEXT_RULE_SPECS = {
    "leave_of_absence": (
        "การลาพักการศึกษา",
        ("rule:31.1", "rule:31.2", "rule:31.3", "rule:31.4"),
        "ข้อบังคับที่เกี่ยวกับการลาพักการศึกษาระบุว่า",
    ),
    "resignation": (
        "การลาออก",
        ("rule:32",),
        "ข้อบังคับที่เกี่ยวกับการลาออกระบุว่า",
    ),
    "credit_transfer": (
        "การเทียบโอนหน่วยกิต",
        ("rule:28", "rule:29"),
        "ข้อบังคับที่เกี่ยวกับการเทียบโอนหน่วยกิตระบุว่า",
    ),
}


def _rule(record: dict[str, Any]) -> PolicyRuleEvidence:
    references = tuple(record.get("provenance", ()))
    if not references:
        raise ValueError("regulation rule has no provenance")
    return PolicyRuleEvidence(
        rule_id=record["rule_id"],
        section_number=record["section_number"],
        category=record["category"],
        rule_text=record["rule_text"],
        provenance=references,
    )


def _normalized_rule_text(text: str) -> str:
    return " ".join(text.split())


def _dedupe_provenance(
    rules: tuple[PolicyRuleEvidence, ...],
) -> tuple[dict[str, Any], ...]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for rule in rules:
        for reference in rule.provenance:
            identity = (
                reference.get("source_filename"),
                reference.get("source_page"),
                reference.get("document_page"),
                reference.get("document_category"),
                reference.get("source_uri"),
                reference.get("source_locator"),
            )
            if identity in seen:
                continue
            seen.add(identity)
            result.append(reference)
    return tuple(result)


def _text_rule_answer(db_path: str | Path, query: PolicyQuery) -> PolicyAnswer:
    spec = _TEXT_RULE_SPECS.get(query.kind)
    if spec is None:
        raise ValueError("unsupported text policy query")
    category, expected_rule_ids, intro = spec
    rows = fetch_regulation_rules(
        db_path,
        category=category,
        rule_ids=expected_rule_ids,
    )
    rules = tuple(_rule(row) for row in rows)
    if tuple(rule.rule_id for rule in rules) != expected_rule_ids:
        raise ValueError("required regulation rule evidence is missing")

    provenance = _dedupe_provenance(rules)
    if not provenance:
        raise ValueError("text policy answer has no provenance")

    rendered_rules = "\n".join(
        f"ข้อ {rule.section_number}: {_normalized_rule_text(rule.rule_text)}"
        for rule in rules
    )
    return PolicyAnswer(
        status="complete",
        query_type=query.kind,
        rules=rules,
        source_rule_id=rules[0].rule_id if len(rules) == 1 else None,
        provenance=provenance,
        rendered_answer=f"{intro}\n{rendered_rules}",
    )


def _facts(db_path: str | Path, query: PolicyQuery) -> tuple[PolicyFact, ...]:
    common = {"category": "เกณฑ์การลงทะเบียน", "context": "regular_semester"}
    if query.kind == "registration_regular_max":
        rows = fetch_policy_facts(db_path, **common, fact_key="regular_maximum")
    elif query.kind == "registration_regular_min":
        rows = fetch_policy_facts(db_path, **common, fact_key="regular_minimum")
    elif query.kind == "registration_exception_max":
        rows = fetch_policy_facts(
            db_path,
            category="เกณฑ์การลงทะเบียน",
            context="graduation_exception",
            fact_key="graduation_exception_maximum",
        )
    elif query.kind == "registration_special_max":
        rows = fetch_policy_facts(
            db_path,
            category="เกณฑ์การลงทะเบียน",
            context="special_semester",
            fact_key="special_maximum",
        )
    elif query.kind == "probation_entry":
        rows = fetch_policy_facts(
            db_path,
            category="เกณฑ์ภาคทัณฑ์ (probation)",
            condition="below",
            fact_key_contains="ถูกภาคทัณฑ์",
        )
    elif query.kind == "probation_cleared":
        rows = fetch_policy_facts(
            db_path,
            category="เกณฑ์ภาคทัณฑ์ (probation)",
            condition="at_least",
            fact_key_contains="พ้นภาคทัณฑ์",
        )
    elif query.kind == "honors_first":
        rows = fetch_policy_facts(
            db_path,
            category="เกณฑ์เกียรตินิยม",
            condition="at_least",
            fact_key_contains="อันดับหนึ่ง",
            context_any=True,
        )
    elif query.kind == "honors_second":
        rows = fetch_policy_facts(
            db_path,
            category="เกณฑ์เกียรตินิยม",
            condition="at_least",
            fact_key_contains="อันดับสอง",
            context_any=True,
        )
    elif query.kind == "reentry_limit":
        rows = fetch_policy_facts(
            db_path,
            category="การกลับเข้าศึกษา",
            condition="at_most",
        )
    else:
        return ()
    return tuple(_fact(row) for row in rows)


def _single_fact(db_path: str | Path, query: PolicyQuery) -> PolicyFact | None:
    facts = _facts(db_path, query)
    if len(facts) != 1:
        raise ValueError("policy fact is missing or ambiguous")
    return facts[0]


def _answer_from_fact(query: PolicyQuery, fact: PolicyFact) -> PolicyAnswer:
    labels = {
        "registration_regular_max": "การลงทะเบียนภาคปกติ",
        "registration_regular_min": "การลงทะเบียนภาคปกติ",
        "registration_exception_max": "การลงทะเบียนกรณีพิเศษ",
        "registration_special_max": "การลงทะเบียนภาคฤดูร้อน/ภาคพิเศษ",
        "probation_entry": "การเข้าภาคทัณฑ์",
        "probation_cleared": "การพ้นภาคทัณฑ์",
        "reentry_limit": "การกลับเข้าศึกษา",
    }
    subject = labels[query.kind]
    if query.kind == "registration_regular_max":
        answer = f"{subject}ลงทะเบียนได้สูงสุด {fact.value} {fact.unit}"
    elif query.kind == "registration_regular_min":
        answer = f"{subject}ต้องลงทะเบียนอย่างน้อย {fact.value} {fact.unit}"
    elif query.kind == "registration_exception_max":
        answer = f"{subject}ลงทะเบียนได้สูงสุด {fact.value} {fact.unit}"
    elif query.kind == "registration_special_max":
        answer = f"{subject}ลงทะเบียนได้สูงสุด {fact.value} {fact.unit}"
    elif query.kind == "probation_entry":
        answer = f"{subject}เมื่อ GPA ต่ำกว่า {fact.value}"
    elif query.kind == "probation_cleared":
        answer = f"{subject}เมื่อ GPA ตั้งแต่ {fact.value} ขึ้นไป"
    else:
        answer = f"{subject}ได้ภายใน {fact.value} {fact.unit}"
    return PolicyAnswer(
        status="complete",
        query_type=query.kind,
        facts=(fact,),
        value=fact.value,
        unit=fact.unit,
        operator=fact.operator,
        condition=fact.condition,
        context=fact.context,
        source_rule_id=fact.source_rule_id,
        provenance=fact.provenance,
        verification_status=fact.verification_status,
        rendered_answer=answer,
    )


def _comparison_answer(db_path: str | Path, query: PolicyQuery) -> PolicyAnswer:
    regular = _single_fact(db_path, PolicyQuery("registration_regular_max"))
    exception = _single_fact(db_path, PolicyQuery("registration_exception_max"))
    assert regular is not None and exception is not None
    facts = (regular, exception)
    amount = query.amount
    if amount is None or not isinstance(amount, int) or amount < 0:
        raise ValueError("invalid registration amount")
    if amount <= regular.value:
        answer = f"ลงทะเบียน {amount} หน่วยกิตได้ตามเพดานปกติ {regular.value} หน่วยกิต"
    elif amount <= exception.value:
        answer = (
            f"ลงทะเบียน {amount} หน่วยกิตเกินเพดานปกติ {regular.value} หน่วยกิต "
            f"และอาจทำได้เฉพาะกรณีที่เข้าเงื่อนไขเพดานพิเศษไม่เกิน {exception.value} หน่วยกิต"
        )
    else:
        answer = (
            f"ลงทะเบียน {amount} หน่วยกิตไม่ได้ตามเพดานที่มีหลักฐานรองรับ "
            f"(ปกติ {regular.value} และกรณีพิเศษไม่เกิน {exception.value} หน่วยกิต)"
        )
    provenance = tuple(reference for fact in facts for reference in fact.provenance)
    return PolicyAnswer(
        status="complete",
        query_type=query.kind,
        facts=facts,
        value=amount,
        unit="หน่วยกิต",
        operator="comparison",
        condition="normal_limit_then_conditional_exception",
        context="regular_semester",
        source_rule_id=regular.source_rule_id,
        provenance=provenance,
        rendered_answer=answer,
    )


def _program_answer(
    db_path: str | Path,
    query: PolicyQuery,
    *,
    catalog_key: str | None = None,
) -> PolicyAnswer:
    if not query.program:
        raise ValueError("program is required")
    record = fetch_program_requirement(
        db_path, query.program, catalog_key=catalog_key
    )
    if record is None:
        return PolicyAnswer(status="insufficient_evidence", query_type=query.kind)
    fact = PolicyFact(
        category="program_requirement",
        fact_key=record["requirement_type"],
        value=record["value"],
        unit=record["unit"],
        operator=record["operator"],
        condition=None,
        context=None,
        source_rule_id=None,
        program=record["program_code"],
        verification_status=None,
        provenance=tuple(record["provenance"]),
    )
    return PolicyAnswer(
        status="complete",
        query_type=query.kind,
        facts=(fact,),
        value=fact.value,
        unit=fact.unit,
        operator=fact.operator,
        program=fact.program,
        provenance=fact.provenance,
        rendered_answer=f"หลักสูตร {fact.program} ต้องเรียนทั้งหมด {fact.value} {fact.unit}",
    )


def _honors_answer(db_path: str | Path, query: PolicyQuery) -> PolicyAnswer:
    facts = _facts(db_path, query)
    if not facts:
        raise ValueError("honors fact is missing")
    values = tuple(fact.value for fact in facts)
    title = "อันดับหนึ่ง" if query.kind == "honors_first" else "อันดับสอง"
    joined = " และ ".join(str(value) for value in values)
    provenance = tuple(reference for fact in facts for reference in fact.provenance)
    return PolicyAnswer(
        status="complete",
        query_type=query.kind,
        facts=facts,
        value=values,
        unit="GPA",
        operator="at_least",
        context="curriculum_structure_and_cumulative_gpa",
        provenance=provenance,
        rendered_answer=f"เกียรตินิยม{title}มี GPA อย่างน้อย {joined}",
    )


def answer_policy_question(
    db_path: str | Path,
    question: str,
    *,
    catalog_key: str | None = None,
) -> PolicyAnswer:
    """Answer one supported policy question using only the canonical runtime DB."""

    query = parse_policy_question(question)
    if query is None:
        return PolicyAnswer(status="unsupported")
    try:
        if query.kind in _TEXT_RULE_SPECS:
            return _text_rule_answer(db_path, query)
        if query.kind == "program_total_credits":
            return _program_answer(db_path, query, catalog_key=catalog_key)
        if query.kind == "registration_compare":
            return _comparison_answer(db_path, query)
        if query.kind in {"honors_first", "honors_second"}:
            return _honors_answer(db_path, query)
        return _answer_from_fact(query, _single_fact(db_path, query))
    except (OSError, sqlite3.Error, KeyError, TypeError, ValueError):
        return PolicyAnswer(status="insufficient_evidence", query_type=query.kind)


__all__ = [
    "PolicyAnswer",
    "PolicyFact",
    "PolicyRuleEvidence",
    "answer_policy_question",
]
