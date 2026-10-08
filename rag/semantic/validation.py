"""Deterministic validation between interpreter and resolver.

Checks: closed task/subject/relation combinations, required companion
structures per task, current-turn grounding of every explicit scope/target
mention (the interpreter must not invent what the student did not write),
target/reference consistency, and policy classification completeness.
Invalid interpretations fail closed; nothing is ever silently repaired.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag.semantic.schema import (
    COMPARISON_OPERATIONS,
    RELATIONS,
    VALID_TASK_SUBJECTS,
    SemanticIntent,
)
from rag.query_spec import (
    _extract_category,
    _extract_topic,
    canonical_plan_meanings,
)

_SUPPORTED_FILTER_SHAPES = frozenset(
    {
        ("topic", "related_to"),
        ("category", "eq"),
    }
)


def _filter_contract_problem(intent: SemanticIntent) -> str | None:
    """Reject filters without a proven compiler and execution consumer."""
    if not intent.filters:
        return None

    if len(intent.filters) == 1:
        item = intent.filters[0]
        if (
            item.field == "has_prerequisite"
            and item.operator == "eq"
            and item.value is True
        ):
            if intent.task == "list" and intent.subject == "course":
                return None
            return "positive prerequisite collection filter requires course list"

    seen_fields: set[str] = set()
    for item in intent.filters:
        shape = (item.field, item.operator)
        if shape not in _SUPPORTED_FILTER_SHAPES:
            return f"unsupported semantic filter: {item.field!r} + {item.operator!r}"
        if item.field in seen_fields:
            return f"duplicate semantic filter field: {item.field!r}"
        seen_fields.add(item.field)

    if intent.task in {"list", "search"} and intent.subject == "course":
        return None

    aggregation = intent.aggregation
    if (
        intent.task == "aggregate"
        and aggregation is not None
        and not aggregation.group_by
        and (
            (aggregation.function == "sum" and aggregation.measure == "credits")
            or (
                aggregation.function == "count"
                and aggregation.measure == "course_count"
            )
        )
    ):
        return None

    return "semantic filters are unsupported for this task/subject shape"


def _filter_grounding_problem(
    intent: SemanticIntent, question: str
) -> str | None:
    """Reject text filters invented beyond the current user message.

    Grounding is semantic normalization, not raw substring equality: a
    topic/category filter is grounded when the CURRENT question contains a
    cue which existing deterministic normalization maps to the SAME
    canonical value. Raw verbatim mention remains a fast path. Broad
    requests stay broad: an ungrounded filter fails the interpretation
    instead of executing narrowed or silently broadened.
    """
    for item in intent.filters:
        if item.field == "topic" and item.operator == "related_to":
            value = item.value
            if not isinstance(value, str) or not value.strip():
                return "semantic topic filter is not grounded in the current question"
            text = value.strip()
            if _contains(question, text):
                continue
            if intent.target.kind == "none":
                normalized = _extract_topic(question, None, ())
                if (
                    isinstance(normalized, str)
                    and normalized.casefold() == text.casefold()
                ):
                    continue
            return "semantic topic filter is not grounded in the current question"
        elif item.field == "category" and item.operator == "eq":
            value = item.value
            if not isinstance(value, str) or not value.strip():
                return "semantic category filter is not grounded in the current question"
            text = value.strip()
            if _contains(question, text):
                continue
            # The cue (e.g. GENED) is only a cue: its meaning comes from the
            # existing deterministic category normalization, which must yield
            # the SAME canonical value. No universal alias is introduced.
            if _extract_category(question) == text:
                continue
            return "semantic category filter is not grounded in the current question"
    return None


def _relation_compatible(task: str, subject: str, relation: str | None) -> bool:
    """Judge whether a relation fits its task shape.

    Lookup requires its property relation. List/search/aggregate/compare/
    rank tolerate a compatible extra relation (it restates the same meaning
    and never widens scope; downstream stages ignore it outside lookup).
    Policy/requirement/unknown require none: a relation there contradicts
    the task (e.g. policy + prerequisite) and stays rejected.
    """
    if task == "lookup":
        if subject == "course":
            return relation in RELATIONS
        if subject == "program":
            return relation == "credits"
        return False
    if task in {"list", "search", "aggregate", "compare", "rank"}:
        return relation is None or relation in RELATIONS
    if task in {"policy", "requirement", "unknown"}:
        return relation is None
    return False


@dataclass(frozen=True, slots=True)
class ValidationResult:
    valid: bool
    reason: str | None = None


def _contains(question: str, text: str) -> bool:
    """Check a mention is grounded in the current-turn question.

    Short codes (program labels and the like) require token boundaries so
    that, for example, ``IT`` is not "found" inside ``security``. Longer
    spans use substring matching because Thai has no word boundaries; the
    lookarounds are boundary-neutral for non-ASCII-adjacent text.
    """
    if len(text) <= 4:
        return (
            re.search(
                r"(?<![A-Za-z0-9_])" + re.escape(text) + r"(?![A-Za-z0-9_])",
                question,
                re.IGNORECASE,
            )
            is not None
        )
    return text.casefold() in question.casefold()


def _plan_hint_consistency_problem(
    raw_plan: str | None,
    plan_hint: str | None,
    field: str,
) -> str | None:
    """Require a canonical hint to agree with deterministic raw-phrase meaning."""
    if plan_hint is None:
        return None
    if not isinstance(plan_hint, str) or not plan_hint.strip() or len(plan_hint) > 32:
        return f"{field} is invalid"
    if raw_plan is None:
        return f"{field} requires a grounded raw plan mention"
    meanings = canonical_plan_meanings(raw_plan)
    if len(meanings) != 1:
        return f"{field} requires a supported deterministic raw plan meaning"
    if plan_hint.strip().casefold() != meanings[0].casefold():
        return f"{field} contradicts deterministic raw plan meaning"
    return None


def _grounded_scope(intent: SemanticIntent, question: str) -> str | None:
    scope = intent.scope
    for label, value in (
        ("program", scope.program),
        ("catalog", scope.catalog),
        ("plan", scope.plan),
    ):
        if value is not None and not _contains(question, value):
            return f"scope.{label} is not grounded in the current question"
    for label, value in (("year", scope.year), ("semester", scope.semester)):
        if value is not None and str(value) not in question:
            return f"scope.{label} is not grounded in the current question"
    return _plan_hint_consistency_problem(
        scope.plan, scope.plan_hint, "scope.plan_hint"
    )


def _target_consistency(intent: SemanticIntent, question: str) -> str | None:
    target = intent.target
    task = intent.task
    if task == "unknown":
        return None
    if target.kind == "none":
        if task in {"lookup"} and intent.subject == "course":
            return "course lookup requires a target reference"
        return None
    if target.kind == "literal":
        if target.raw_text is None or not _contains(question, target.raw_text):
            return "literal target is not grounded in the current question"
        if len(target.raw_text) > 80:
            return "literal target is too long"
        return None
    if target.kind in {"current_course", "previous_result_set"}:
        if target.raw_text is None or not _contains(question, target.raw_text):
            return f"{target.kind} target is not grounded in the current question"
        return None
    if target.kind == "result_ordinal":
        return None
    return f"unsupported target kind: {target.kind}"


def _task_structure(intent: SemanticIntent, question: str) -> str | None:
    task = intent.task
    if (task, intent.subject) not in VALID_TASK_SUBJECTS:
        return "unsupported task/subject combination"
    if not _relation_compatible(task, intent.subject, intent.relation):
        return "incompatible relation for task shape"
    if task == "aggregate" and intent.aggregation is None:
        return "aggregate requires an aggregation specification"
    if task == "rank" and intent.ranking is None:
        return "rank requires a ranking specification"
    if task == "compare" and intent.comparison is None:
        return "compare requires a comparison specification"
    if task == "compare" and intent.comparison is not None:
        comparison = intent.comparison
        if comparison.operation not in COMPARISON_OPERATIONS:
            return "compare requires an explicit comparison operation"
        if comparison.operation in {"set_difference", "overlap"}:
            if intent.comparison.measure != "course_count":
                return "set comparison requires the course_count measure"
        elif intent.comparison.measure != "credits":
            return "numeric comparison requires the credits measure"
        for side_label, side in (
            ("left", comparison.left),
            ("right", comparison.right),
        ):
            if not side:
                return f"comparison.{side_label} must be non-empty"
            side_fields = dict(side)
            for key, value in side:
                if key == "course" and (
                    not isinstance(value, str) or not _contains(question, value)
                ):
                    return "comparison course operand is not grounded in the current question"
                if key in {"program", "catalog", "plan"} and (
                    not isinstance(value, str) or not _contains(question, value)
                ):
                    return f"comparison.{side_label}.{key} is not grounded in the current question"
                if key in {"year", "semester"} and str(value) not in question:
                    return f"comparison.{side_label}.{key} is not grounded in the current question"
            hint_problem = _plan_hint_consistency_problem(
                side_fields.get("plan"),
                side_fields.get("plan_hint"),
                f"comparison.{side_label}.plan_hint",
            )
            if hint_problem is not None:
                return hint_problem
    if task in {"policy", "requirement"} and intent.policy_topic is None:
        return "policy/requirement requires a policy topic"
    if task == "lookup" and intent.subject == "course" and intent.relation is None:
        return "course lookup requires a relation"
    problem = _filter_contract_problem(intent)
    if problem is not None:
        return problem
    problem = _filter_grounding_problem(intent, question)
    if problem is not None:
        return problem
    return None


def validate_semantic_intent(
    intent: SemanticIntent, question: str
) -> ValidationResult:
    """Validate a parsed intent against the question it claims to describe."""
    if not isinstance(intent, SemanticIntent):
        return ValidationResult(False, "interpretation is not a SemanticIntent")
    if not isinstance(question, str) or not question.strip():
        return ValidationResult(False, "question must be non-empty text")
    problem = _task_structure(intent, question)
    if problem is not None:
        return ValidationResult(False, problem)
    problem = _grounded_scope(intent, question)
    if problem is not None:
        return ValidationResult(False, problem)
    problem = _target_consistency(intent, question)
    if problem is not None:
        return ValidationResult(False, problem)
    return ValidationResult(True, None)


__all__ = ["ValidationResult", "validate_semantic_intent"]
