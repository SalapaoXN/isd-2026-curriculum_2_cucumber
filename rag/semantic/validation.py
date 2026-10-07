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


def _grounded_scope(intent: SemanticIntent, question: str) -> str | None:
    scope = intent.scope
    if scope.plan_hint is not None:
        if not isinstance(scope.plan_hint, str) or not scope.plan_hint.strip() or len(scope.plan_hint) > 32:
            return "scope.plan_hint is invalid"
        if scope.plan is None:
            return "scope.plan_hint requires a grounded raw plan mention"
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
    return None


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
            if "plan_hint" in side_fields:
                hint = side_fields["plan_hint"]
                if (
                    not isinstance(hint, str)
                    or not hint.strip()
                    or len(hint) > 32
                ):
                    return f"comparison.{side_label}.plan_hint is invalid"
                if "plan" not in side_fields:
                    return f"comparison.{side_label}.plan_hint requires a grounded raw plan mention"
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
    if task in {"policy", "requirement"} and intent.policy_topic is None:
        return "policy/requirement requires a policy topic"
    if task == "lookup" and intent.subject == "course" and intent.relation is None:
        return "course lookup requires a relation"
    problem = _filter_contract_problem(intent)
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
