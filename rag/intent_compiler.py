"""Pure compilation of validated intent proposals into QuerySpec values."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from typing import Any

from rag.intent_interpreter import (
    IntentInterpretation,
    validate_execution_scope,
)
from rag.query_spec import QuerySpec


class IntentCompilerError(ValueError):
    """Raised when an intent cannot be represented without widening scope."""


_FACT_TO_OPERATION = {
    "course_list": "list",
    "course_description": "describe",
    "placement": "placement",
    "prerequisite": "prerequisite",
    "similarity": "similarity",
    "course_comparison": "compare",
    "plan_comparison": "compare",
    "program_identity": "program_discovery",
}

_INTENT_OPERATION = {
    "topic_course_search": "list",
    "course_description": "describe",
    "placement_query": "placement",
    "prerequisite_query": "prerequisite",
    "similarity_query": "similarity",
    "course_comparison": "compare",
    "plan_comparison": "compare",
    "program_discovery": "program_discovery",
    "count_query": "count",
    "course_credit_query": "sum_credits",
    "existence_query": "existence",
}

_INTENT_FACTS = {
    "topic_course_search": frozenset({"course_list"}),
    "course_description": frozenset({"course_description"}),
    "placement_query": frozenset({"placement"}),
    "prerequisite_query": frozenset({"prerequisite"}),
    "similarity_query": frozenset({"similarity"}),
    "course_comparison": frozenset({"course_comparison"}),
    "plan_comparison": frozenset({"plan_comparison"}),
    "program_discovery": frozenset({"program_identity"}),
    "count_query": frozenset({"course_list"}),
    "course_credit_query": frozenset({"course_credit"}),
    "existence_query": frozenset({"course_list"}),
}

_GROUP_BY_FOR_INTENT = {
    "similarity_query": "course",
    "course_comparison": "course",
    "plan_comparison": "plan",
}

_SUPPORTED_GROUP_BY = frozenset({"course", "plan", "year"})
_PREFERENCE_FACT_SHAPES = (
    ("course_list",),
    ("course_list", "prerequisite"),
)
_PREFERENCE_BASE_OPERATION_SHAPES = frozenset(
    {
        (),
        ("list",),
        ("prerequisite",),
        ("list", "prerequisite"),
    }
)


def _clean_text(value: Any, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise IntentCompilerError(f"{field} must be a string or None")
    value = value.strip()
    return value or None


def _tuple_text(values: Iterable[Any], field: str) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        cleaned = _clean_text(value, field)
        if cleaned is None:
            raise IntentCompilerError(f"{field} contains an empty value")
        result.append(cleaned)
    if len(set(result)) != len(result):
        raise IntentCompilerError(f"{field} contains duplicate values")
    return tuple(result)


def _tuple_int(values: Iterable[Any], field: str) -> tuple[int, ...]:
    result = tuple(values)
    if any(type(value) is not int for value in result):
        raise IntentCompilerError(f"{field} must contain integers")
    if len(set(result)) != len(result):
        raise IntentCompilerError(f"{field} contains duplicate values")
    return result


def _merge_axis(
    base: tuple[Any, ...],
    authoritative: tuple[Any, ...],
    proposed: tuple[Any, ...],
    field: str,
) -> tuple[Any, ...]:
    """Preserve base values, then fill only from caller-authorized values."""
    base = tuple(base)
    authoritative = tuple(authoritative)
    proposed = tuple(proposed)
    if base and authoritative and set(base) != set(authoritative):
        raise IntentCompilerError(f"{field} conflicts with deterministic scope")
    if base:
        if proposed and not set(proposed).issubset(base):
            raise IntentCompilerError(f"{field} interpretation widens scope")
        return base
    if authoritative:
        if proposed and not set(proposed).issubset(authoritative):
            raise IntentCompilerError(f"{field} interpretation widens scope")
        return authoritative
    if proposed:
        raise IntentCompilerError(
            f"{field} interpretation lacks authoritative caller scope"
        )
    return ()


def _merge_program(
    base: str | None,
    authoritative: str | None,
    proposed: str | None,
) -> str | None:
    base = _clean_text(base, "program")
    authoritative = _clean_text(authoritative, "authoritative_program")
    proposed = _clean_text(proposed, "proposed_program")
    if base and authoritative and base.casefold() != authoritative.casefold():
        raise IntentCompilerError("program conflicts with deterministic scope")
    if base:
        if proposed and proposed.casefold() != base.casefold():
            raise IntentCompilerError("program interpretation conflicts")
        return base
    if authoritative:
        if proposed and proposed.casefold() != authoritative.casefold():
            raise IntentCompilerError("program interpretation conflicts")
        return authoritative
    if proposed:
        raise IntentCompilerError("program cannot be inferred by the compiler")
    return None


def _merge_course_codes(
    base: tuple[str, ...],
    proposed: tuple[str, ...],
) -> tuple[str, ...]:
    base = _tuple_text(base, "course_codes")
    proposed = _tuple_text(proposed, "course_codes")
    if base and proposed and not set(proposed).issubset(base):
        raise IntentCompilerError("course-code interpretation widens scope")
    if base:
        return base
    return proposed


def _merge_operations(
    base: tuple[str, ...],
    mapped: tuple[str, ...],
) -> tuple[str, ...]:
    result = list(base)
    for operation in mapped:
        if operation not in result:
            result.append(operation)
    return tuple(result)


def _mapped_operations(
    interpretation: IntentInterpretation,
    base_operations: tuple[str, ...],
) -> tuple[str, ...]:
    intent = interpretation.intent
    if intent == "preference_recommendation_evidence":
        if interpretation.judgement_dimension != "preference":
            raise IntentCompilerError(
                "preference evidence requires the preference dimension"
            )
        requested_facts = tuple(interpretation.requested_facts)
        if requested_facts not in _PREFERENCE_FACT_SHAPES:
            raise IntentCompilerError(
                "preference evidence has an unsupported requested_facts shape"
            )
        if tuple(base_operations) not in _PREFERENCE_BASE_OPERATION_SHAPES:
            raise IntentCompilerError(
                "preference evidence cannot combine with unsupported operations"
            )
        mapped = tuple(
            _FACT_TO_OPERATION[fact]
            for fact in requested_facts
        )
        merged = set(base_operations) | set(mapped)
        return tuple(
            operation
            for operation in ("list", "prerequisite")
            if operation in merged
        )
    if intent in {"course_credit_query", "existence_query"}:
        expected = {
            "course_credit_query": ("course_credit",),
            "existence_query": ("course_list",),
        }[intent]
        if tuple(interpretation.requested_facts) != expected:
            raise IntentCompilerError("exact-course intent has wrong evidence request")
        if tuple(base_operations):
            raise IntentCompilerError("exact-course intent cannot combine with operations")
        return (_INTENT_OPERATION[intent],)
    if intent == "count_query":
        if tuple(interpretation.requested_facts) != ("course_list",):
            raise IntentCompilerError(
                "count_query requires exactly the course_list evidence request"
            )
        if tuple(base_operations):
            raise IntentCompilerError(
                "count_query cannot combine with existing operations"
            )
        return ("count",)
    if intent in _INTENT_OPERATION:
        expected_facts = _INTENT_FACTS[intent]
        facts = set(interpretation.requested_facts)
        if facts and not facts.issubset(expected_facts):
            raise IntentCompilerError(
                f"requested_facts do not match intent {intent!r}"
            )
        return (_INTENT_OPERATION[intent],)

    if intent not in {"workload_judgement", "preference_recommendation_evidence"}:
        raise IntentCompilerError(f"unsupported intent: {intent!r}")

    mapped = tuple(
        _FACT_TO_OPERATION[fact]
        for fact in interpretation.requested_facts
        if fact in _FACT_TO_OPERATION
    )
    if not mapped and not base_operations:
        raise IntentCompilerError("judgement has no representable evidence operation")
    return _merge_operations(base_operations, mapped)


def _validate_group_by(
    base: tuple[str, ...],
    required: str | None,
) -> tuple[str, ...]:
    base = tuple(base)
    if any(value not in _SUPPORTED_GROUP_BY for value in base):
        raise IntentCompilerError("unsupported group-by dimension")
    if required is None:
        return base
    if base and required not in base:
        raise IntentCompilerError("group-by conflicts with intent")
    return base or (required,)


def _require_course_target(spec: QuerySpec, *, exact_count: int | None = None) -> None:
    codes = tuple(spec.course_codes)
    if exact_count is not None and len(codes) != exact_count:
        raise IntentCompilerError("intent requires an exact course-code target")
    if exact_count is None and not codes and spec.course_name is None:
        raise IntentCompilerError("intent requires an exact course target")


def _require_single_course_target(spec: QuerySpec) -> None:
    codes = tuple(spec.course_codes)
    if len(codes) > 1 or (
        not codes
        and not (isinstance(spec.course_name, str) and spec.course_name.strip())
    ):
        raise IntentCompilerError("intent requires one exact course target")


def compile_intent_to_query_spec(
    base_spec: QuerySpec,
    interpretation: IntentInterpretation,
    *,
    authoritative_program: str | None = None,
    authoritative_plans: tuple[str, ...] = (),
    authoritative_years: tuple[int, ...] = (),
    authoritative_semesters: tuple[int, ...] = (),
    allowed_course_codes: tuple[str, ...] = (),
) -> QuerySpec:
    """Compile a validated intent proposal without external side effects."""
    if not isinstance(base_spec, QuerySpec):
        raise TypeError("base_spec must be a QuerySpec")
    if not isinstance(interpretation, IntentInterpretation):
        raise TypeError("interpretation must be an IntentInterpretation")

    effective_program = authoritative_program or base_spec.program
    effective_plans = tuple(authoritative_plans) or tuple(base_spec.plans)
    effective_years = tuple(authoritative_years) or tuple(base_spec.years)
    effective_semesters = tuple(authoritative_semesters) or tuple(base_spec.semesters)
    effective_codes = tuple(allowed_course_codes) or tuple(base_spec.course_codes)
    try:
        eligibility = validate_execution_scope(
            interpretation,
            program=effective_program,
            allowed_plans=effective_plans,
            allowed_years=effective_years,
            allowed_semesters=effective_semesters,
            allowed_course_codes=effective_codes,
        )
    except (TypeError, ValueError) as error:
        raise IntentCompilerError(str(error)) from None
    if not eligibility.eligible:
        raise IntentCompilerError(eligibility.reason)
    if base_spec.judgement == "unsupported":
        raise IntentCompilerError("base QuerySpec has unsupported judgement")

    if interpretation.intent in {"course_credit_query", "existence_query"}:
        if base_spec.topic is not None or base_spec.judgement not in {None, "none"}:
            raise IntentCompilerError("exact-course intent cannot use topic or judgement")
        if interpretation.topic is not None or interpretation.judgement_dimension is not None:
            raise IntentCompilerError("exact-course intent cannot add topic or judgement")
    if interpretation.intent == "count_query":
        if base_spec.topic is not None:
            raise IntentCompilerError("count_query cannot use a topic")
        if base_spec.judgement != "none":
            raise IntentCompilerError("count_query cannot use a judgement")
        if base_spec.course_codes or base_spec.course_name is not None:
            raise IntentCompilerError("count_query cannot target a course")
        if interpretation.topic is not None:
            raise IntentCompilerError("count_query cannot add a topic")
        if interpretation.course_codes:
            raise IntentCompilerError("count_query cannot add course targets")

    program = _merge_program(
        base_spec.program,
        authoritative_program,
        interpretation.proposed_program,
    )
    plans = _merge_axis(
        base_spec.plans,
        tuple(authoritative_plans),
        interpretation.proposed_plans,
        "plans",
    )
    years = _merge_axis(
        base_spec.years,
        tuple(authoritative_years),
        interpretation.proposed_years,
        "years",
    )
    semesters = _merge_axis(
        base_spec.semesters,
        tuple(authoritative_semesters),
        interpretation.proposed_semesters,
        "semesters",
    )
    course_codes = _merge_course_codes(
        tuple(base_spec.course_codes),
        interpretation.course_codes,
    )

    topic = base_spec.topic
    if interpretation.topic:
        if topic and topic.casefold() != interpretation.topic.casefold():
            raise IntentCompilerError("topic interpretation conflicts")
        topic = topic or interpretation.topic
    if interpretation.intent == "preference_recommendation_evidence" and not topic:
        raise IntentCompilerError("preference evidence requires a non-empty topic")

    if base_spec.category is not None and not isinstance(base_spec.category, str):
        raise IntentCompilerError("base category is malformed")

    operations = _mapped_operations(
        interpretation,
        tuple(base_spec.operations),
    )
    if interpretation.intent == "topic_course_search" and not topic:
        raise IntentCompilerError("topic_course_search requires a topic")
    if interpretation.intent in {
        "course_description",
        "placement_query",
        "prerequisite_query",
    }:
        _require_course_target(
            replace(base_spec, course_codes=course_codes),
        )
    if interpretation.intent in {"course_credit_query", "existence_query"}:
        _require_single_course_target(replace(base_spec, course_codes=course_codes))
    if interpretation.intent in {"similarity_query", "course_comparison"}:
        _require_course_target(
            replace(base_spec, course_codes=course_codes),
            exact_count=2,
        )
    if interpretation.intent == "plan_comparison":
        if len(plans) != 2 or len(set(plans)) != 2:
            raise IntentCompilerError("plan_comparison requires two plans")
    if interpretation.intent == "program_discovery":
        _require_course_target(replace(base_spec, course_codes=course_codes))

    required_group = _GROUP_BY_FOR_INTENT.get(interpretation.intent)
    group_by = _validate_group_by(tuple(base_spec.group_by), required_group)
    judgement = (
        interpretation.judgement_dimension
        if interpretation.intent in {
            "workload_judgement",
            "preference_recommendation_evidence",
        }
        else "none"
    )
    if interpretation.intent not in {
        "workload_judgement",
        "preference_recommendation_evidence",
    } and interpretation.judgement_dimension is not None:
        raise IntentCompilerError("non-judgement intent has a judgement dimension")

    return replace(
        base_spec,
        program=program,
        plans=plans,
        years=years,
        semesters=semesters,
        course_codes=course_codes,
        topic=topic,
        operations=operations,
        group_by=group_by,
        judgement=judgement,
    )


__all__ = ["IntentCompilerError", "compile_intent_to_query_spec"]
