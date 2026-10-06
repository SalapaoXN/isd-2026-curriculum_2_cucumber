"""Semantic → execution compiler: structured intent to QuerySpec.

The compiler maps validated semantic structure onto the deterministic
execution representation. It NEVER reads the student's raw wording: every
field comes from ResolvedIntent (canonical scope/identity) or fixed
operation tables. ``normalized_question`` is intentionally empty so legacy
surface-pattern reads cannot misfire on semantic-mode input.
"""

from __future__ import annotations

from rag.query_spec import QuerySpec
from rag.semantic.schema import ResolvedIntent
from rag.semantic.planner import effective_aggregation_group_by
from typing import Any

_RELATION_OPERATIONS = {
    "identity": ("identity",),
    "description": ("describe",),
    "credits": ("sum_credits",),
    "prerequisite": ("prerequisite",),
    "placement": ("placement",),
    "existence": ("existence",),
}


def _operations_for(resolved: ResolvedIntent) -> tuple[str, ...]:
    intent = resolved.intent
    if intent.task == "lookup" and intent.subject == "course":
        operation = _RELATION_OPERATIONS.get(intent.relation or "")
        if operation is not None:
            return operation
    if intent.task == "lookup" and intent.subject == "program":
        return ("sum_credits",)
    if intent.task == "list":
        operations: list[str] = ["list"]
        if any(
            item.field == "has_prerequisite" and item.operator == "eq" and item.value is True
            for item in intent.filters
        ):
            operations.append("prerequisite")
        return tuple(operations)
    if intent.task == "search":
        return ("list",)
    if intent.task == "aggregate" and _is_single_total_aggregate(resolved):
        if intent.aggregation is not None and intent.aggregation.function == "count":
            return ("count",)
        return ("sum_credits",)
    return ()


def _is_single_total_aggregate(resolved: ResolvedIntent) -> bool:
    """Return True for aggregates the deterministic evidence path can prove.

    Single totals (no group breakdown) for supported function/measure pairs
    reuse the frozen COUNT evidence machinery through the normal claim
    pipeline. Grouped breakdowns stay on the SQL path (unsupported until
    the synthesizer can express them) instead of degrading into listings.
    """
    aggregation = resolved.intent.aggregation
    if aggregation is None:
        return False
    if effective_aggregation_group_by(resolved):
        return False
    if aggregation.function == "sum" and aggregation.measure == "credits":
        return True
    return aggregation.function == "count" and aggregation.measure == "course_count"


def compile_resolved_intent_to_query_spec(
    resolved: ResolvedIntent, question: str
) -> QuerySpec:
    """Build an execution QuerySpec from resolved structure (no NL parsing)."""
    intent = resolved.intent
    scope = resolved.scope
    target = resolved.target
    category = next(
        (
            item.value
            for item in intent.filters
            if item.field == "category" and item.operator == "eq"
        ),
        None,
    )
    topic = next(
        (
            item.value
            for item in intent.filters
            if item.field == "topic" and item.operator == "related_to"
        ),
        None,
    )
    codes: tuple[str, ...] = ()
    name: str | None = None
    if target.kind in {"literal", "current_course", "result_ordinal"}:
        if target.course_code is not None:
            codes = (target.course_code,)
        elif target.course_name is not None:
            name = target.course_name
    return QuerySpec(
        original_question=question if isinstance(question, str) else "",
        normalized_question="",
        program=scope.program,
        plans=(scope.plan,) if scope.plan is not None else (),
        years=tuple(scope.years),
        semesters=tuple(scope.semesters),
        course_codes=codes,
        course_name=name,
        category=category if isinstance(category, str) else None,
        topic=topic if isinstance(topic, str) else None,
        operations=_operations_for(resolved),
        group_by=(),
        judgement="none",
        credit_units=None,
        references_previous_result_set=(
            intent.target.kind == "previous_result_set"
        ),
        result_ordinal=None,
    )


_PROGRAM_WORD = {
    "AIT": "AIT",
    "BIT": "BIT",
    "DSBA": "DSBA",
    "IT": "IT",
    "GENED": "GENED",
}


def _scope_words(resolved: ResolvedIntent) -> list[str]:
    words: list[str] = []
    scope = resolved.scope
    if scope.program is not None:
        words.append(_PROGRAM_WORD.get(scope.program.upper(), scope.program))
    if scope.plan is not None:
        words.append(str(scope.plan))
    for year in scope.years:
        words.append(f"ปี {year}")
    for semester in scope.semesters:
        words.append(f"เทอม {semester}")
    return words


def synthesize_canonical_utterance(resolved: ResolvedIntent) -> str | None:
    """Build a canonical-scope utterance for the guarded SQL adapter.

    Constructed ONLY from validated structure with allowlisted vocabulary —
    never a rewrite of the student's wording. Returns None when the shape
    cannot be expressed as a bounded collection/listing request.
    """
    intent = resolved.intent
    if intent.task not in {"list", "search", "aggregate"}:
        return None
    if intent.task == "aggregate" and (
        intent.aggregation is None or intent.aggregation.group_by
    ):
        return None
    parts = _scope_words(resolved)
    topic = next(
        (
            item.value
            for item in intent.filters
            if item.field == "topic" and item.operator == "related_to"
        ),
        None,
    )
    if intent.task == "search" and not isinstance(topic, str):
        return None
    if isinstance(topic, str) and topic.strip():
        parts.append(f"เกี่ยวกับ {topic.strip()}")
    parts.append("มีวิชาอะไรบ้าง")
    utterance = " ".join(part for part in parts if part).strip()
    return utterance or None


__all__ = [
    "compile_resolved_intent_to_query_spec",
    "synthesize_canonical_utterance",
]
