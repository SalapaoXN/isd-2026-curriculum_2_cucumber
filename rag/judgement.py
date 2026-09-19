"""Pure judgement evidence over already-derived curriculum evidence.

This module deliberately stops at immutable, grounded evidence.  It does not
generate prose, rank courses, infer difficulty, or make the upstream
``unsupported`` decision.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from decimal import Decimal
import math
from numbers import Real
from types import MappingProxyType
from typing import Any

from rag.aggregation import (
    AGGREGATION_STATES,
    COMPARISON_RELATIONS,
    ComparisonAggregation,
    ComponentAggregation,
    CourseSetAggregation,
)
from rag.evidence_executor import (
    DirectPrerequisiteBurden,
    DirectPrerequisiteRequirement,
)


JUDGEMENT_EVIDENCE_STATES = (
    "supported",
    "descriptive_only",
    "insufficient_evidence",
)
_QUANTITY_FACTS = frozenset({"count", "option_count", "required_load"})
_WORKLOAD_FACTS = frozenset({"course_count", "required_load", "credits"})
_FACT_OPERATIONS = {
    "count": "count",
    "course_count": "count",
    "option_count": "option_count",
    "required_load": "required_load",
    "credits": "sum_credits",
}


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    return value


def _direct_provenance(value: Mapping[str, Any]) -> tuple[Any, ...]:
    provenance = value.get("provenance", ())
    if provenance is None:
        return ()
    if isinstance(provenance, Mapping) or isinstance(provenance, str):
        return (_freeze(provenance),)
    return tuple(_freeze(reference) for reference in provenance)


def _valid_provenance_container(value: Mapping[str, Any]) -> bool:
    provenance = value.get("provenance", ())
    if provenance is None or isinstance(provenance, (Mapping, str)):
        return True
    if not isinstance(provenance, (list, tuple)):
        return False
    return all(isinstance(reference, (Mapping, str)) for reference in provenance)


def _valid_aggregate_provenance(value: Any) -> bool:
    if isinstance(value, CourseSetAggregation):
        components = value.courses
    elif isinstance(value, ComponentAggregation):
        components = value.components
    else:
        return False
    return all(
        isinstance(component, Mapping) and _valid_provenance_container(component)
        for component in components
    )


def _provenance(value: Any) -> tuple[Any, ...]:
    if isinstance(value, CourseSetAggregation):
        return tuple(
            reference
            for course in value.courses
            for reference in _direct_provenance(course)
        )
    if isinstance(value, ComponentAggregation):
        return tuple(
            reference
            for component in value.components
            for reference in _direct_provenance(component)
        )
    if isinstance(value, ComparisonAggregation):
        return _provenance(value.left) + _provenance(value.right)
    if isinstance(value, Mapping):
        references = list(_direct_provenance(value))
        for evidence in value.get("description_evidence", ()):
            if isinstance(evidence, Mapping):
                references.extend(_direct_provenance(evidence))
        burden = value.get("direct_prerequisite_burden")
        if isinstance(burden, DirectPrerequisiteBurden):
            references.extend(burden.provenance)
        return tuple(references)
    return ()


def _aggregate_status(value: Any) -> str | None:
    status = getattr(value, "status", None)
    return status if status in AGGREGATION_STATES else None


def _fact_value(value: Any) -> Any:
    if isinstance(value, CourseSetAggregation):
        return value.count
    if isinstance(value, ComponentAggregation):
        return value.value
    return None


def _fact_operation(value: Any) -> str | None:
    if isinstance(value, CourseSetAggregation):
        return "count"
    if isinstance(value, ComponentAggregation):
        return value.operation
    return None


def _validate_facts(facts: Mapping[str, Any], allowed: frozenset[str]) -> bool:
    if not isinstance(facts, Mapping):
        return False
    if not set(facts).issubset(allowed):
        return False
    return bool(facts) and all(
        _aggregate_status(value) is not None
        and _fact_operation(value) == _FACT_OPERATIONS[name]
        for name, value in facts.items()
    )


def _comparison_sources(comparison: ComparisonAggregation) -> tuple[Any, ...]:
    return tuple(
        operand
        for operand in (comparison.left, comparison.right)
        if isinstance(operand, (CourseSetAggregation, ComponentAggregation))
    )


def _comparison_sources_match(
    comparisons: Mapping[str, ComparisonAggregation],
    facts: Mapping[str, Any],
) -> bool:
    fact_items = tuple(facts.items())
    for name, comparison in comparisons.items():
        expected_operation = _FACT_OPERATIONS.get(name)
        for source in _comparison_sources(comparison):
            source_operation = _fact_operation(source)
            if source_operation is None:
                return False
            if name == "comparison":
                matching_facts = fact_items
                name_must_match = False
            elif name in facts and expected_operation is not None:
                matching_facts = ((name, facts[name]),)
                name_must_match = True
            else:
                matching_facts = fact_items
                name_must_match = True
            if not any(
                (not name_must_match or key == name)
                and _FACT_OPERATIONS.get(key) == source_operation
                and source == value
                for key, value in matching_facts
            ):
                return False
    return True


def _numeric_comparison_value(operand: Any) -> Real | Decimal | None:
    if isinstance(operand, bool) or not isinstance(operand, (Real, Decimal)):
        return None
    try:
        finite = (
            operand.is_finite()
            if isinstance(operand, Decimal)
            else math.isfinite(float(operand))
        )
    except (OverflowError, TypeError, ValueError):
        return None
    return operand if finite else None


def _validated_comparison_operand(
    operand: Any,
    facts: Mapping[str, Any],
) -> Real | Decimal | None:
    if isinstance(operand, (CourseSetAggregation, ComponentAggregation)):
        if _aggregate_status(operand) not in {"complete", "valid_empty"}:
            return None
        if not any(operand == value for value in facts.values()):
            return None
        operand = _fact_value(operand)
    return _numeric_comparison_value(operand)


def _comparison_operands_are_valid(
    comparison: ComparisonAggregation,
    facts: Mapping[str, Any],
) -> bool:
    left = _validated_comparison_operand(comparison.left, facts)
    right = _validated_comparison_operand(comparison.right, facts)
    if left is None or right is None or comparison.relation not in COMPARISON_RELATIONS:
        return False
    expected_relation = (
        "less" if left < right else "greater" if left > right else "equal"
    )
    return comparison.status == "complete" and comparison.relation == expected_relation


def _comparison_status(
    comparisons: Any,
    facts: Mapping[str, Any],
    *,
    allowed_names: frozenset[str] | None = None,
) -> tuple[Mapping[str, Any], str | None]:
    if comparisons is None:
        return MappingProxyType({}), None
    if isinstance(comparisons, ComparisonAggregation):
        comparisons = {"comparison": comparisons}
    if not isinstance(comparisons, Mapping):
        return MappingProxyType({}), "insufficient_evidence"
    for name, comparison in comparisons.items():
        if not isinstance(name, str) or not isinstance(comparison, ComparisonAggregation):
            return MappingProxyType({}), "insufficient_evidence"
    if allowed_names is not None and not set(comparisons).issubset(allowed_names):
        return MappingProxyType({}), "insufficient_evidence"
    if not _comparison_sources_match(comparisons, facts) or not all(
        _comparison_operands_are_valid(comparison, facts)
        for comparison in comparisons.values()
    ):
        sanitized = {
            name: ComparisonAggregation(
                "insufficient_evidence", None, None, None
            )
            for name, comparison in comparisons.items()
        }
        return _freeze(sanitized), "insufficient_evidence"
    frozen = _freeze(comparisons)
    status = (
        "insufficient_evidence"
        if any(comparison.status == "insufficient_evidence" for comparison in comparisons.values())
        else "supported"
    )
    return frozen, status


@dataclass(frozen=True, slots=True)
class JudgementEvidence:
    """Immutable grounded evidence for one non-unsupported judgement."""

    judgement: str
    status: str
    facts: Mapping[str, Any] = field(default_factory=dict)
    comparisons: Mapping[str, ComparisonAggregation] = field(default_factory=dict)
    options: tuple[Mapping[str, Any], ...] = ()
    provenance: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        if self.judgement not in {"quantity", "workload", "preference"}:
            raise ValueError("judgement evidence cannot implement unsupported or similarity")
        if self.status not in JUDGEMENT_EVIDENCE_STATES:
            raise ValueError(f"unsupported judgement evidence state: {self.status!r}")
        facts = _freeze(self.facts)
        comparisons = _freeze(self.comparisons)
        options = tuple(_freeze(option) for option in self.options)
        if not isinstance(facts, Mapping) or not isinstance(comparisons, Mapping):
            raise TypeError("facts and comparisons must be mappings")
        if any(not isinstance(option, Mapping) for option in options):
            raise TypeError("options must contain mappings")
        object.__setattr__(self, "facts", facts)
        object.__setattr__(self, "comparisons", comparisons)
        object.__setattr__(self, "options", options)
        object.__setattr__(self, "provenance", _freeze(self.provenance))

    @property
    def values(self) -> Mapping[str, Any]:
        """Compatibility name for the grounded aggregate facts."""
        return self.facts

    def fact_value(self, name: str) -> Any:
        return _fact_value(self.facts.get(name))


def evaluate_quantity(
    facts: Mapping[str, Any],
    *,
    comparison: ComparisonAggregation | None = None,
) -> JudgementEvidence:
    """Expose grounded quantity facts without inventing a many/few threshold."""
    valid_facts = _validate_facts(facts, _QUANTITY_FACTS)
    if not valid_facts:
        return JudgementEvidence("quantity", "insufficient_evidence")
    if not all(_valid_aggregate_provenance(value) for value in facts.values()):
        return JudgementEvidence("quantity", "insufficient_evidence")
    if any(
        _aggregate_status(value) == "insufficient_evidence" for value in facts.values()
    ):
        return JudgementEvidence("quantity", "insufficient_evidence", facts=facts)
    comparisons, comparison_status = _comparison_status(
        comparison, facts
    )
    status = comparison_status or "descriptive_only"
    provenance = tuple(
        reference
        for value in facts.values()
        for reference in _provenance(value)
    )
    return JudgementEvidence(
        "quantity",
        status,
        facts=facts,
        comparisons=comparisons,
        provenance=provenance,
    )


def evaluate_workload(
    facts: Mapping[str, Any],
    *,
    comparisons: Mapping[str, ComparisonAggregation] | None = None,
) -> JudgementEvidence:
    """Expose count/load/credit proxies without converting them to difficulty."""
    valid_facts = _validate_facts(facts, _WORKLOAD_FACTS)
    if not valid_facts:
        return JudgementEvidence("workload", "insufficient_evidence")
    if not all(_valid_aggregate_provenance(value) for value in facts.values()):
        return JudgementEvidence("workload", "insufficient_evidence")
    if set(facts) != _WORKLOAD_FACTS or any(
        _aggregate_status(value) == "insufficient_evidence" for value in facts.values()
    ):
        return JudgementEvidence("workload", "insufficient_evidence", facts=facts)
    frozen_comparisons, comparison_status = _comparison_status(
        comparisons, facts, allowed_names=_WORKLOAD_FACTS
    )
    status = comparison_status or "supported"
    provenance = tuple(
        reference
        for value in facts.values()
        for reference in _provenance(value)
    )
    return JudgementEvidence(
        "workload",
        status,
        facts=facts,
        comparisons=frozen_comparisons,
        provenance=provenance,
    )


def _valid_preference_option(option: Mapping[str, Any]) -> bool:
    program = option.get("program")
    course_code = option.get("course_code")
    partition = option.get("partition")
    if (
        not isinstance(program, str)
        or not program.strip()
        or not isinstance(course_code, str)
        or not course_code.strip()
        or not isinstance(partition, Mapping)
        or not partition
        or not _valid_preference_provenance(option)
    ):
        return False

    if "direct_prerequisite_burden" in option and not _valid_preference_burden(
        option
    ):
        return False

    descriptions = option.get("description_evidence")
    if isinstance(descriptions, (str, bytes)) or descriptions is None:
        return False
    try:
        descriptions = tuple(descriptions)
    except TypeError:
        return False
    if not descriptions:
        return False

    for description in descriptions:
        if not isinstance(description, Mapping):
            return False
        chunk_id = description.get("chunk_id")
        text = description.get("text", description.get("description"))
        if (
            not isinstance(chunk_id, str)
            or not chunk_id.strip()
            or not isinstance(text, str)
            or not text.strip()
            or description.get("chunk_type") not in (None, "description")
            or not _valid_preference_provenance(description)
        ):
            return False
        if description.get("program") not in (None, program):
            return False
        if description.get("course_code") not in (None, course_code):
            return False
        if (
            "course_id" in option
            and "course_id" in description
            and description["course_id"] != option["course_id"]
        ):
            return False
        evidence_partition = description.get("partition")
        if evidence_partition is not None and (
            not isinstance(evidence_partition, Mapping)
            or dict(evidence_partition) != dict(partition)
        ):
            return False
    return True


def _valid_preference_burden(option: Mapping[str, Any]) -> bool:
    burden = option.get("direct_prerequisite_burden")
    if not isinstance(burden, DirectPrerequisiteBurden):
        return False
    course_id = option.get("course_id")
    if (
        burden.status != "complete"
        or isinstance(course_id, bool)
        or not isinstance(course_id, int)
        or burden.program != option.get("program")
        or burden.course_code != option.get("course_code")
        or burden.course_id != course_id
        or not _valid_preference_provenance({"provenance": burden.provenance})
        or type(burden.required_course_count) is not int
        or burden.required_course_count < 0
        or type(burden.alternative_group_count) is not int
        or burden.alternative_group_count < 0
    ):
        return False
    groups = burden.ordered_requirement_groups
    if not isinstance(groups, tuple):
        return False
    if len(groups) != burden.required_course_count + burden.alternative_group_count:
        return False
    if not isinstance(burden.alternative_member_counts, tuple):
        return False
    if len(burden.alternative_member_counts) != burden.alternative_group_count:
        return False
    if any(type(count) is not int or count < 0 for count in burden.alternative_member_counts):
        return False

    alternative_count = 0
    required_count = 0
    for group in groups:
        if not isinstance(group, DirectPrerequisiteRequirement):
            return False
        if not _valid_preference_provenance({"provenance": group.provenance}):
            return False
        if group.kind == "required_course":
            required_count += 1
            if (
                isinstance(group.prerequisite_course_id, bool)
                or not isinstance(group.prerequisite_course_id, int)
                or not isinstance(group.prerequisite_code, str)
                or not group.prerequisite_code.strip()
                or group.alternative_group_id is not None
                or group.alternative_members != ()
            ):
                return False
        elif group.kind == "alternative_group":
            alternative_count += 1
            if (
                isinstance(group.alternative_group_id, bool)
                or not isinstance(group.alternative_group_id, int)
                or type(group.minimum_choices) is not int
                or type(group.maximum_choices) is not int
                or group.minimum_choices < 1
                or group.maximum_choices < group.minimum_choices
                or not isinstance(group.alternative_members, tuple)
                or not group.alternative_members
            ):
                return False
            for member in group.alternative_members:
                if (
                    not isinstance(member, Mapping)
                    or isinstance(member.get("course_id"), bool)
                    or not isinstance(member.get("course_id"), int)
                    or not isinstance(member.get("course_code"), str)
                    or not member.get("course_code", "").strip()
                    or not _valid_preference_provenance(member)
                ):
                    return False
        else:
            return False
    return required_count == burden.required_course_count and alternative_count == burden.alternative_group_count


def _valid_preference_provenance(value: Mapping[str, Any]) -> bool:
    provenance = value.get("provenance")
    if isinstance(provenance, (Mapping, str)):
        references = (provenance,)
    elif isinstance(provenance, (list, tuple)):
        references = tuple(provenance)
    else:
        return False
    return bool(references) and all(
        isinstance(reference, (Mapping, str)) for reference in references
    )


def evaluate_preference(
    options: Iterable[Mapping[str, Any]] | Any,
) -> JudgementEvidence:
    """Return supplied topic-relevant options without ranking or recommending."""
    source_status = getattr(options, "status", None)
    if source_status is not None:
        source_options = getattr(options, "scored_candidates", ())
        if source_status in {
            "empty_structural_candidates",
            "description_missing",
            "vector_missing_or_invalid",
        }:
            return JudgementEvidence("preference", "insufficient_evidence")
    else:
        source_options = options

    if isinstance(source_options, (str, bytes)):
        return JudgementEvidence("preference", "insufficient_evidence")
    try:
        materialized = tuple(source_options)
    except TypeError:
        return JudgementEvidence("preference", "insufficient_evidence")
    if any(
        not isinstance(option, Mapping) or not _valid_preference_option(option)
        for option in materialized
    ):
        return JudgementEvidence("preference", "insufficient_evidence")
    frozen_options = tuple(_freeze(option) for option in materialized)
    provenance = tuple(
        reference
        for option in materialized
        for reference in _provenance(option)
    )
    return JudgementEvidence(
        "preference",
        "supported",
        options=frozen_options,
        provenance=provenance,
    )


__all__ = [
    "JUDGEMENT_EVIDENCE_STATES",
    "JudgementEvidence",
    "evaluate_preference",
    "evaluate_quantity",
    "evaluate_workload",
]
