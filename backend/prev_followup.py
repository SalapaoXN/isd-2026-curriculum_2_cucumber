"""Previous-answer follow-up answering over re-grounded canonical evidence.

Explain/source follow-ups reuse only bounded canonical references retained in
``last_answer``. Every follow-up re-fetches canonical evidence before
answering; retained references never carry prose or factual values.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

from rag.hybrid_demo import answer_question_once
from rag.policy.answer import PolicyAnswer, answer_policy_query
from rag.policy.query import PolicyQuery
from rag.prev_answer import (
    EXPLAIN_PREVIOUS,
    SOURCE_PREVIOUS,
    RATIONALE_PREVIOUS,
    RATIONALE_UNAVAILABLE_ANSWER,
    _is_rule_id,
    build_explain_prompt,
    effective_course_question,
    explanation_output_valid,
    parse_last_answer,
    render_policy_source,
    render_provenance_source,
)
from rag.resolution import QueryContext

_INSUFFICIENT_ANSWER = "ไม่พบหลักฐานที่มีแหล่งอ้างอิงเพียงพอสำหรับคำตอบนี้"


def _project_provenance(raw: Any) -> list[dict[str, Any]]:
    """Project only well-shaped provenance references, else fail closed.

    Mirrors the grounded pipeline contract: every entry must be a mapping
    with a non-empty ``source_filename`` string and an integer
    ``source_page``. Any invalid entry discards the whole set so callers
    fail closed instead of projecting corrupt provenance or crashing.
    """
    projected: list[dict[str, Any]] = []
    if not isinstance(raw, (list, tuple)):
        return projected
    for reference in raw:
        if (
            not isinstance(reference, Mapping)
            or not isinstance(reference.get("source_filename"), str)
            or not reference.get("source_filename")
            or isinstance(reference.get("source_page"), bool)
            or not isinstance(reference.get("source_page"), int)
        ):
            return []
        projected.append(dict(reference))
    return projected


def build_policy_reference(
    query: PolicyQuery,
    answer: PolicyAnswer,
    *,
    catalog_key: str | None,
) -> dict[str, Any] | None:
    """Build a bounded policy referent from a freshly answered policy turn."""
    if not isinstance(answer, PolicyAnswer) or answer.status != "complete":
        return None
    if not answer.provenance:
        return None
    identifiers: list[str] = []
    for rule in tuple(answer.rules or ()):
        rule_id = getattr(rule, "rule_id", None)
        if _is_rule_id(rule_id) and rule_id not in identifiers:
            identifiers.append(rule_id)
    for fact in tuple(answer.facts or ()):
        source = getattr(fact, "source_rule_id", None)
        if _is_rule_id(source) and source not in identifiers:
            identifiers.append(source)
    candidate: dict[str, Any] = {"route": "policy", "policy_kind": query.kind}
    if query.program is not None:
        candidate["program"] = query.program
    if catalog_key is not None:
        candidate["catalog_key"] = catalog_key
    if query.plan is not None:
        candidate["plan"] = query.plan
    if query.amount is not None:
        candidate["amount"] = query.amount
    if identifiers:
        candidate["evidence_ids"] = identifiers
    try:
        return parse_last_answer(
            candidate, program=query.program, catalog_key=catalog_key
        )
    except (TypeError, ValueError):
        return None


def build_course_reference(
    course_code: str,
    operations: Any,
    *,
    program: str | None,
    catalog_key: str | None,
) -> dict[str, Any] | None:
    """Build a bounded course referent from a freshly answered course turn."""
    candidate: dict[str, Any] = {
        "route": "course",
        "course_code": course_code,
        "operations": list(operations) if isinstance(operations, (list, tuple)) else operations,
    }
    if program is not None:
        candidate["program"] = program
    if catalog_key is not None:
        candidate["catalog_key"] = catalog_key
    try:
        return parse_last_answer(candidate, program=program, catalog_key=catalog_key)
    except (TypeError, ValueError):
        return None


def _policy_evidence_json(answer: PolicyAnswer) -> str:
    facts = [
        {
            "fact_key": fact.fact_key,
            "value": fact.value,
            "unit": fact.unit,
            "operator": fact.operator,
            "condition": fact.condition,
            "context": fact.context,
            "source_rule_id": fact.source_rule_id,
        }
        for fact in tuple(answer.facts or ())
    ]
    rules = [
        {
            "rule_id": rule.rule_id,
            "section_number": rule.section_number,
            "rule_text": rule.rule_text,
        }
        for rule in tuple(answer.rules or ())
    ]
    return json.dumps({"facts": facts, "rules": rules}, ensure_ascii=False)


def _explain_with_model(
    model_callable: Callable[[str], str] | None,
    evidence_json: str,
    followup: str,
    deterministic_fallback: str,
) -> str:
    if not callable(model_callable) or not deterministic_fallback.strip():
        return deterministic_fallback
    try:
        generated = model_callable(build_explain_prompt(evidence_json, followup))
    except Exception:
        return deterministic_fallback
    if explanation_output_valid(generated, evidence_json):
        return generated.strip()
    return deterministic_fallback


def _preserved_scope(
    last_answer: dict[str, Any],
    program: str | None,
    catalog_key: str | None,
) -> dict[str, Any]:
    context: dict[str, Any] = {}
    if program is not None:
        context["program"] = program
    if catalog_key is not None:
        context["catalog_key"] = catalog_key
    if last_answer.get("route") == "course":
        if isinstance(last_answer.get("course_code"), str):
            context["course_code"] = last_answer["course_code"]
        if isinstance(last_answer.get("operations"), list):
            context["operations"] = list(last_answer["operations"])
    context["last_answer"] = last_answer
    return context


def answer_previous_followup(
    *,
    db_path: str | Path,
    question: str,
    followup_kind: str,
    last_answer: dict[str, Any] | None,
    program: str | None,
    catalog_key: str | None,
    model_callable: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """Answer an explain/source/rationale follow-up from re-grounded evidence."""
    preserved = (
        _preserved_scope(last_answer, program, catalog_key)
        if isinstance(last_answer, dict)
        else None
    )

    def insufficient(answer: str = _INSUFFICIENT_ANSWER) -> dict[str, Any]:
        return {
            "status": "insufficient_evidence",
            "answer": answer,
            "provenance": [],
            "next_context": preserved,
        }

    if not isinstance(last_answer, dict):
        return insufficient()
    if followup_kind == RATIONALE_PREVIOUS:
        # Canonical evidence records requirements, never their motivation.
        return insufficient(RATIONALE_UNAVAILABLE_ANSWER)
    if followup_kind not in (EXPLAIN_PREVIOUS, SOURCE_PREVIOUS):
        return insufficient()

    route = last_answer.get("route")
    if route == "policy":
        try:
            query = PolicyQuery(
                kind=last_answer["policy_kind"],
                program=last_answer.get("program"),
                amount=last_answer.get("amount"),
                plan=last_answer.get("plan"),
            )
        except (TypeError, ValueError):
            return insufficient()
        answer = answer_policy_query(
            db_path,
            query,
            catalog_key=last_answer.get("catalog_key") or catalog_key,
        )
        if answer.status != "complete" or not answer.provenance:
            return insufficient()
        provenance = _project_provenance(answer.provenance)
        if not provenance:
            return insufficient()
        if followup_kind == SOURCE_PREVIOUS:
            text = render_policy_source(answer)
            if not text:
                return insufficient()
            return {
                "status": "answer",
                "answer": text,
                "provenance": provenance,
                "next_context": preserved,
            }
        evidence_json = _policy_evidence_json(answer)
        text = _explain_with_model(
            model_callable, evidence_json, question, answer.rendered_answer
        )
        if not text.strip():
            return insufficient()
        return {
            "status": "answer",
            "answer": text,
            "provenance": provenance,
            "next_context": preserved,
        }

    if route == "course":
        effective = effective_course_question(
            last_answer.get("course_code"), last_answer.get("operations")
        )
        if effective is None:
            return insufficient()
        scope_program = last_answer.get("program") or program
        scope_catalog = last_answer.get("catalog_key") or catalog_key
        if not isinstance(scope_program, str) or not scope_program.strip():
            return insufficient()
        try:
            regrounded = answer_question_once(
                db_path,
                effective,
                intent_model_callable=None,
                conversation_context=QueryContext(
                    program=scope_program.strip(),
                    catalog_key=scope_catalog,
                    course_code=str(last_answer.get("course_code")).strip(),
                ),
            )
        except (TypeError, ValueError, OSError):
            return insufficient()
        if regrounded.get("route") is not None:
            return insufficient()
        result = regrounded.get("result")
        final_answer = regrounded.get("final_answer", "")
        provenance_raw = regrounded.get("provenance", [])
        if (
            regrounded.get("status") != "answer"
            or not isinstance(final_answer, str)
            or not final_answer.strip()
            or not isinstance(provenance_raw, (list, tuple))
            or not provenance_raw
        ):
            return insufficient()
        _ = result
        provenance = _project_provenance(provenance_raw)
        if not provenance:
            return insufficient()
        if followup_kind == SOURCE_PREVIOUS:
            text = render_provenance_source(provenance)
            if not text:
                return insufficient()
            return {
                "status": "answer",
                "answer": text,
                "provenance": provenance,
                "next_context": preserved,
            }
        evidence_json = json.dumps(
            {
                "course_code": str(last_answer.get("course_code")).strip(),
                "operations": list(last_answer.get("operations") or []),
                "grounded_statements": final_answer.strip(),
                "sources": provenance,
            },
            ensure_ascii=False,
        )
        text = _explain_with_model(
            model_callable, evidence_json, question, final_answer.strip()
        )
        if not text.strip():
            return insufficient()
        return {
            "status": "answer",
            "answer": text,
            "provenance": provenance,
            "next_context": preserved,
        }

    return insufficient()


__all__ = [
    "answer_previous_followup",
    "build_course_reference",
    "build_policy_reference",
]
