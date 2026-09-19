"""Pure deterministic aggregation of already-scoped course evidence."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import math
from numbers import Real
from types import MappingProxyType
from typing import Any

from rag.structured.queries import (
    earliest_year_semester,
    earliest_year_semester_from_choices,
)


AGGREGATION_STATES = ("complete", "valid_empty", "insufficient_evidence")
AGGREGATION_OPERATIONS = ("option_count", "required_load", "sum_credits")
COMPARISON_RELATIONS = ("less", "equal", "greater")


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    return value


def _stable_key(value: Any) -> str:
    if isinstance(value, Mapping):
        return repr(
            tuple(
                sorted(
                    (str(key), _stable_key(item))
                    for key, item in value.items()
                )
            )
        )
    if isinstance(value, (list, tuple)):
        return repr(tuple(_stable_key(item) for item in value))
    if isinstance(value, (set, frozenset)):
        return repr(tuple(sorted(_stable_key(item) for item in value)))
    return repr(value)


def _identity(component: Mapping[str, Any]) -> tuple[str, str]:
    program = component.get("program")
    course_code = component.get("course_code")
    if not isinstance(program, str) or not program.strip():
        raise ValueError("each course component needs a non-empty program")
    if not isinstance(course_code, str) or not course_code.strip():
        raise ValueError("each course component needs a non-empty course_code")
    return program, course_code


def _alternative_group_id(component: Mapping[str, Any]) -> tuple[str, Any]:
    program = component.get("program")
    group_id = component.get("alternative_group_id")
    if not isinstance(program, str) or not program.strip():
        raise ValueError("each alternative group needs a non-empty program")
    if group_id in (None, ""):
        raise ValueError("alternative groups need an alternative_group_id")
    return program, group_id


def _partition_key(component: Mapping[str, Any]) -> str:
    partition = component.get("partition", {})
    if not isinstance(partition, Mapping):
        raise ValueError("course component partition must be a mapping")
    return _stable_key(partition)


def _course_set_identity(component: Mapping[str, Any]) -> tuple[str, Any]:
    """Return the logical identity used by course-set aggregation."""
    if (
        component.get("alternative_group_id") is not None
        or component.get("is_alternative") is True
    ):
        return "alternative_group", _alternative_group_id(component)
    return "course", _identity(component)


def _merge_provenance(
    components: Iterable[Mapping[str, Any]],
) -> tuple[Any, ...]:
    references: dict[str, Any] = {}
    for component in components:
        provenance = component.get("provenance", ())
        if provenance is None:
            continue
        if isinstance(provenance, Mapping) or isinstance(provenance, str):
            provenance = (provenance,)
        for reference in provenance:
            frozen = _freeze(reference)
            references.setdefault(_stable_key(frozen), frozen)
    return tuple(references[key] for key in sorted(references))


@dataclass(frozen=True, slots=True)
class CourseSetAggregation:
    """Immutable list/count/existence result for one course-set relation."""

    status: str
    courses: tuple[Mapping[str, Any], ...] = ()
    count: int | None = None
    exists: bool | None = None

    def __post_init__(self) -> None:
        if self.status not in AGGREGATION_STATES:
            raise ValueError(f"unsupported aggregation status: {self.status!r}")
        courses = tuple(_freeze(course) for course in self.courses)
        if any(not isinstance(course, Mapping) for course in courses):
            raise ValueError("courses must contain mappings")
        if self.status == "valid_empty" and (courses or self.count != 0 or self.exists):
            raise ValueError("valid_empty must represent zero courses")
        if self.status == "insufficient_evidence" and (
            self.count is not None or self.exists is not None
        ):
            raise ValueError(
                "insufficient_evidence cannot claim count or existence"
            )
        if self.status == "complete" and (
            self.count != len(courses) or self.exists is not bool(courses)
        ):
            raise ValueError("complete aggregation fields disagree")
        object.__setattr__(self, "courses", courses)


@dataclass(frozen=True, slots=True)
class ComponentAggregation:
    """Immutable scalar aggregation with explicit evidence state."""

    operation: str
    status: str
    value: int | float | None = None
    components: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if self.operation not in AGGREGATION_OPERATIONS:
            raise ValueError(f"unsupported aggregation operation: {self.operation!r}")
        if self.status not in AGGREGATION_STATES:
            raise ValueError(f"unsupported aggregation status: {self.status!r}")
        components = tuple(_freeze(component) for component in self.components)
        if any(not isinstance(component, Mapping) for component in components):
            raise ValueError("components must contain mappings")
        if self.status == "insufficient_evidence" and self.value is not None:
            raise ValueError("insufficient_evidence cannot claim a value")
        if self.status in {"complete", "valid_empty"} and self.value is None:
            raise ValueError("complete aggregation must have a value")
        if self.status == "valid_empty" and (components or self.value != 0):
            raise ValueError("valid_empty must represent zero components")
        object.__setattr__(self, "components", components)

    @property
    def count(self) -> int | None:
        """Compatibility view for count-like aggregation operations."""
        if self.operation in {"option_count", "required_load"}:
            return self.value if isinstance(self.value, int) else None
        return None

    @property
    def exists(self) -> bool | None:
        """Compatibility view for count-like aggregation operations."""
        count = self.count
        return None if count is None else count > 0

    @property
    def total_credits(self) -> int | float | None:
        if self.operation == "sum_credits":
            return self.value
        return None


@dataclass(frozen=True, slots=True)
class EarliestPartition:
    """The tied earliest placements for one preserved structural partition."""

    partition: Mapping[str, Any]
    value: tuple[int, int]
    placements: tuple[Mapping[str, Any], ...]
    provenance: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        if not _valid_year_semester(self.value):
            raise ValueError("earliest value must be a valid (year, semester)")
        partition = _freeze(self.partition)
        placements = tuple(_freeze(placement) for placement in self.placements)
        if not isinstance(partition, Mapping) or any(
            not isinstance(placement, Mapping) for placement in placements
        ):
            raise ValueError("earliest partition data must contain mappings")
        object.__setattr__(self, "partition", partition)
        object.__setattr__(self, "placements", placements)
        object.__setattr__(self, "provenance", _freeze(self.provenance))


@dataclass(frozen=True, slots=True)
class EarliestAggregation:
    """Immutable partition-local earliest placement result."""

    status: str
    partitions: tuple[EarliestPartition, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in AGGREGATION_STATES:
            raise ValueError(f"unsupported aggregation status: {self.status!r}")
        partitions = tuple(self.partitions)
        if any(not isinstance(partition, EarliestPartition) for partition in partitions):
            raise ValueError("partitions must contain EarliestPartition values")
        if self.status == "valid_empty" and partitions:
            raise ValueError("valid_empty earliest cannot contain placements")
        object.__setattr__(self, "partitions", partitions)

    @property
    def value(self) -> tuple[int, int] | None:
        """Return a scalar value only when exactly one partition is present."""
        if self.status != "complete" or len(self.partitions) != 1:
            return None
        return self.partitions[0].value


@dataclass(frozen=True, slots=True)
class ComparisonAggregation:
    """Immutable deterministic relation between two aggregate operands."""

    status: str
    relation: str | None
    left: Any
    right: Any

    def __post_init__(self) -> None:
        if self.status not in AGGREGATION_STATES:
            raise ValueError(f"unsupported aggregation status: {self.status!r}")
        if self.status == "insufficient_evidence" and self.relation is not None:
            raise ValueError("insufficient_evidence cannot claim a relation")
        if self.status == "complete" and self.relation not in COMPARISON_RELATIONS:
            raise ValueError("complete comparison needs a valid relation")
        if self.status == "valid_empty" and self.relation is not None:
            raise ValueError("valid_empty comparison cannot claim a relation")


@dataclass(frozen=True, slots=True)
class PlanComparisonInput:
    """One complete, explicitly identified plan partition for comparison."""

    plan: str
    course_set: CourseSetAggregation
    placements: tuple[Mapping[str, Any], ...] = ()
    placements_complete: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.plan, str) or not self.plan.strip():
            raise ValueError("plan comparison inputs need a non-empty plan")
        if not isinstance(self.course_set, CourseSetAggregation):
            raise TypeError("course_set must be a CourseSetAggregation")
        if not isinstance(self.placements_complete, bool):
            raise ValueError("placements_complete must be a boolean")
        placements = tuple(_freeze(placement) for placement in self.placements)
        if any(not isinstance(placement, Mapping) for placement in placements):
            raise ValueError("placements must contain mappings")
        object.__setattr__(self, "placements", placements)


@dataclass(frozen=True, slots=True)
class PlanPlacementDifference:
    """Canonical placement periods for one shared course that differs."""

    course_key: tuple[str, Any]
    left_periods: tuple[tuple[int, int], ...]
    right_periods: tuple[tuple[int, int], ...]
    left_placements: tuple[Mapping[str, Any], ...] = ()
    right_placements: tuple[Mapping[str, Any], ...] = ()
    provenance: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.course_key, tuple) or len(self.course_key) != 2:
            raise ValueError("course_key must be a two-part logical identity")
        left_periods = tuple(self.left_periods)
        right_periods = tuple(self.right_periods)
        if (
            not left_periods
            or not right_periods
            or any(not _valid_year_semester(period) for period in left_periods)
            or any(not _valid_year_semester(period) for period in right_periods)
        ):
            raise ValueError("placement differences need valid periods")
        if left_periods == right_periods:
            raise ValueError("placement differences need unequal periods")
        left_placements = tuple(_freeze(placement) for placement in self.left_placements)
        right_placements = tuple(_freeze(placement) for placement in self.right_placements)
        if any(not isinstance(placement, Mapping) for placement in left_placements + right_placements):
            raise ValueError("placement differences need mapping evidence")
        object.__setattr__(self, "left_periods", left_periods)
        object.__setattr__(self, "right_periods", right_periods)
        object.__setattr__(self, "left_placements", left_placements)
        object.__setattr__(self, "right_placements", right_placements)
        object.__setattr__(self, "provenance", _freeze(self.provenance))


@dataclass(frozen=True, slots=True)
class PlanComparisonAggregation:
    """Typed set/placement differences between two explicit plan partitions."""

    status: str
    left_plan: str
    right_plan: str
    only_left: tuple[Mapping[str, Any], ...] = ()
    only_right: tuple[Mapping[str, Any], ...] = ()
    placement_differences: tuple[PlanPlacementDifference, ...] = ()
    provenance: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in AGGREGATION_STATES:
            raise ValueError(f"unsupported aggregation status: {self.status!r}")
        if not isinstance(self.left_plan, str) or not self.left_plan.strip():
            raise ValueError("left_plan must be non-empty")
        if not isinstance(self.right_plan, str) or not self.right_plan.strip():
            raise ValueError("right_plan must be non-empty")
        only_left = tuple(_freeze(course) for course in self.only_left)
        only_right = tuple(_freeze(course) for course in self.only_right)
        differences = tuple(self.placement_differences)
        if any(not isinstance(course, Mapping) for course in only_left + only_right):
            raise ValueError("plan differences must contain course mappings")
        if any(not isinstance(item, PlanPlacementDifference) for item in differences):
            raise ValueError("placement_differences must contain typed differences")
        object.__setattr__(self, "only_left", only_left)
        object.__setattr__(self, "only_right", only_right)
        object.__setattr__(self, "placement_differences", differences)
        object.__setattr__(self, "provenance", _freeze(self.provenance))


def _numeric_credit(value: Any) -> Decimal | None:
    if isinstance(value, bool) or not isinstance(value, (Real, Decimal)):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Decimal) and not value.is_finite():
        return None
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return parsed if parsed.is_finite() else None


def _numeric_result(value: Decimal) -> int | float:
    if value == value.to_integral_value():
        return int(value)
    return float(value)


def _valid_year_semester(value: Any) -> bool:
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and all(isinstance(item, int) and not isinstance(item, bool) for item in value)
        and 1 <= value[0] <= 4
        and 1 <= value[1] <= 2
    )


def _placement_earliest(placement: Mapping[str, Any]) -> tuple[int, int] | None:
    if not isinstance(placement, Mapping):
        return None
    if "year_semester_choices" in placement:
        return earliest_year_semester_from_choices(
            placement.get("year_semester_choices")
        )
    return earliest_year_semester(
        placement.get("year", placement.get("year_number")),
        placement.get("semester", placement.get("semester_number")),
        placement.get("flexible_year_semester_raw"),
    )


def _placement_periods(
    placement: Mapping[str, Any],
) -> tuple[tuple[int, int], ...] | None:
    choices = placement.get("year_semester_choices")
    if choices is not None:
        if not isinstance(choices, (list, tuple)) or not choices:
            return None
        normalized: set[tuple[int, int]] = set()
        for choice in choices:
            if not _valid_year_semester(choice):
                return None
            normalized.add(choice)
        return tuple(sorted(normalized))
    earliest = _placement_earliest(placement)
    return (earliest,) if earliest is not None else None


def _placement_plan(placement: Mapping[str, Any]) -> str | None:
    for key in ("plan_key", "plan"):
        value = placement.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    partition = placement.get("partition")
    if not isinstance(partition, Mapping):
        return None
    for key in ("plan_key", "plan"):
        value = partition.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    plans = partition.get("plans")
    if isinstance(plans, (list, tuple)) and len(plans) == 1:
        value = plans[0]
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _plan_course_map(
    course_set: CourseSetAggregation,
) -> dict[tuple[str, Any], Mapping[str, Any]] | None:
    courses: dict[tuple[str, Any], Mapping[str, Any]] = {}
    for course in course_set.courses:
        try:
            identity = _plan_logical_identity(course)
        except (TypeError, ValueError):
            return None
        if identity in courses:
            return None
        courses[identity] = course
    return courses


def _plan_logical_identity(component: Mapping[str, Any]) -> tuple[str, Any]:
    """Use stable member identities for alternative groups across plans."""
    alternatives = component.get("alternative_courses")
    if component.get("alternative_group_id") is not None and alternatives is not None:
        if not isinstance(alternatives, (list, tuple)) or not alternatives:
            raise ValueError("alternative groups need complete member identities")
        members: list[tuple[str, str]] = []
        for member in alternatives:
            if not isinstance(member, Mapping):
                raise ValueError("alternative group members must be mappings")
            member_data = dict(member)
            if member_data.get("program") in (None, ""):
                member_data["program"] = component.get("program")
            members.append(_identity(member_data))
        if len(set(members)) != len(members):
            raise ValueError("alternative group members must be unique")
        return "alternative_members", tuple(sorted(members))
    return _course_set_identity(component)


def _plan_placement_map(
    plan: str,
    placements: Iterable[Mapping[str, Any]],
) -> dict[
    tuple[str, Any],
    tuple[tuple[tuple[int, int], ...], tuple[Mapping[str, Any], ...]],
] | None:
    grouped: dict[
        tuple[str, Any],
        tuple[set[tuple[int, int]], list[Mapping[str, Any]]],
    ] = {}
    for placement in placements:
        try:
            identity = _plan_logical_identity(placement)
        except (TypeError, ValueError):
            return None
        record_plan = _placement_plan(placement)
        if record_plan is not None and record_plan != plan:
            return None
        periods = _placement_periods(placement)
        if periods is None:
            return None
        period_set, records = grouped.setdefault(identity, (set(), []))
        period_set.update(periods)
        records.append(placement)
    return {
        identity: (tuple(sorted(periods)), tuple(sorted(records, key=_stable_key)))
        for identity, (periods, records) in grouped.items()
    }


def _incomplete_plan_comparison(
    left_plan: str,
    right_plan: str,
) -> PlanComparisonAggregation:
    return PlanComparisonAggregation(
        status="insufficient_evidence",
        left_plan=left_plan,
        right_plan=right_plan,
    )


def aggregate_plan_comparison(
    left: PlanComparisonInput,
    right: PlanComparisonInput,
) -> PlanComparisonAggregation:
    """Compare two canonical plan partitions without flattening their identity."""
    if not isinstance(left, PlanComparisonInput) or not isinstance(right, PlanComparisonInput):
        raise TypeError("plan comparison inputs must use PlanComparisonInput")
    if left.plan == right.plan:
        raise ValueError("plan comparison inputs must identify distinct plans")
    if (
        left.course_set.status == "insufficient_evidence"
        or right.course_set.status == "insufficient_evidence"
        or not left.placements_complete
        or not right.placements_complete
    ):
        return _incomplete_plan_comparison(left.plan, right.plan)

    left_courses = _plan_course_map(left.course_set)
    right_courses = _plan_course_map(right.course_set)
    left_placements = _plan_placement_map(left.plan, left.placements)
    right_placements = _plan_placement_map(right.plan, right.placements)
    if (
        left_courses is None
        or right_courses is None
        or left_placements is None
        or right_placements is None
        or not set(left_placements).issubset(left_courses)
        or not set(right_placements).issubset(right_courses)
    ):
        return _incomplete_plan_comparison(left.plan, right.plan)

    only_left = tuple(
        left_courses[key]
        for key in sorted(set(left_courses) - set(right_courses), key=repr)
    )
    only_right = tuple(
        right_courses[key]
        for key in sorted(set(right_courses) - set(left_courses), key=repr)
    )
    placement_differences: list[PlanPlacementDifference] = []
    for key in sorted(set(left_courses) & set(right_courses), key=repr):
        left_placement = left_placements.get(key)
        right_placement = right_placements.get(key)
        if (left_placement is None) != (right_placement is None):
            return _incomplete_plan_comparison(left.plan, right.plan)
        if left_placement is None or right_placement is None:
            continue
        left_periods, left_records = left_placement
        right_periods, right_records = right_placement
        if left_periods != right_periods:
            placement_differences.append(
                PlanPlacementDifference(
                    course_key=key,
                    left_periods=left_periods,
                    right_periods=right_periods,
                    left_placements=left_records,
                    right_placements=right_records,
                    provenance=_merge_provenance(left_records + right_records),
                )
            )

    all_evidence = (
        tuple(left_courses.values())
        + tuple(right_courses.values())
        + tuple(left.placements)
        + tuple(right.placements)
    )
    return PlanComparisonAggregation(
        status="complete",
        left_plan=left.plan,
        right_plan=right.plan,
        only_left=only_left,
        only_right=only_right,
        placement_differences=tuple(placement_differences),
        provenance=_merge_provenance(all_evidence),
    )


def _comparison_operand(value: Any) -> tuple[str, Any] | None:
    if isinstance(value, ComponentAggregation):
        return value.status, value.value
    if isinstance(value, CourseSetAggregation):
        return value.status, value.count
    if isinstance(value, EarliestAggregation):
        return value.status, value.value
    if isinstance(value, bool):
        return None
    if isinstance(value, (Real, Decimal)):
        return "complete", value
    if _valid_year_semester(value):
        return "complete", value
    return None


def _component_kind(component: Mapping[str, Any]) -> str:
    return "alternative_group" if component.get("alternative_group_id") is not None else "course"


def _dedup_scalar_components(
    components: Iterable[Mapping[str, Any]],
) -> tuple[tuple[Mapping[str, Any], ...], bool]:
    """Deduplicate normal courses/groups only within each exact partition."""
    grouped: dict[tuple[str, str, Any], list[Mapping[str, Any]]] = {}
    for component in components:
        if not isinstance(component, Mapping):
            raise ValueError("components must be mappings")
        partition = _partition_key(component)
        if _component_kind(component) == "course":
            program, code = _identity(component)
            key = (partition, "course", (program, code))
        else:
            program, group_id = _alternative_group_id(component)
            key = (partition, "alternative_group", (program, group_id))
        grouped.setdefault(key, []).append(component)

    selected: list[Mapping[str, Any]] = []
    consistent = True
    for key in sorted(grouped, key=lambda item: repr(item)):
        members = sorted(grouped[key], key=_stable_key)
        first = dict(members[0])
        # Conflicting authoritative credit facts cannot be safely collapsed.
        credit_values = {
            _stable_key(member.get("counted_credit_units")) for member in members
        }
        if len(credit_values) > 1:
            consistent = False
        if key[1] == "alternative_group":
            completeness = {
                member.get("membership_complete") is True for member in members
            }
            if len(completeness) > 1:
                consistent = False
            alternative_members: dict[
                tuple[Any, Any], list[Mapping[str, Any]]
            ] = {}
            for member in members:
                for alternative in _alternative_members(member):
                    identity = (
                        alternative.get("program", member.get("program")),
                        alternative.get("course_code"),
                    )
                    alternative_members.setdefault(identity, []).append(alternative)
            if alternative_members:
                first["alternative_courses"] = tuple(
                    _merge_member_records(alternative_members[identity])
                    for identity in sorted(alternative_members, key=repr)
                )
        provenance = _merge_provenance(members)
        if provenance:
            first["provenance"] = provenance
        selected.append(first)
    return tuple(selected), consistent


def _alternative_members(component: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    members = component.get("alternative_courses", ())
    if members is None:
        return ()
    if isinstance(members, (str, bytes)) or not isinstance(members, Iterable):
        raise ValueError("alternative_courses must be iterable")
    normalized: list[Mapping[str, Any]] = []
    for member in members:
        if not isinstance(member, Mapping):
            raise ValueError("alternative_courses must contain mappings")
        normalized.append(member)
    return tuple(sorted(normalized, key=_stable_key))


def _merge_member_records(
    members: Iterable[Mapping[str, Any]],
) -> Mapping[str, Any]:
    ordered = sorted(members, key=_stable_key)
    merged = dict(ordered[0])
    provenance = _merge_provenance(ordered)
    if provenance:
        merged["provenance"] = provenance
    return merged


def _option_identities(
    component: Mapping[str, Any],
) -> tuple[tuple[str, str], ...]:
    if _component_kind(component) == "course":
        return (_identity(component),)
    parent_program = component["program"]
    identities: list[tuple[str, str]] = []
    for member in _alternative_members(component):
        program = member.get("program", parent_program)
        code = member.get("course_code")
        if not isinstance(program, str) or not program.strip():
            raise ValueError("alternative member needs a non-empty program")
        if not isinstance(code, str) or not code.strip():
            raise ValueError("alternative member needs a non-empty course_code")
        identities.append((program, code))
    return tuple(identities)


def aggregate_components(
    components: Iterable[Mapping[str, Any]],
    operation: str,
    *,
    evidence_complete: bool = True,
) -> ComponentAggregation:
    """Aggregate already-scoped components for one supported scalar operation."""
    if operation not in AGGREGATION_OPERATIONS:
        raise ValueError(f"unsupported aggregation operation: {operation!r}")
    if not isinstance(evidence_complete, bool):
        raise ValueError("evidence_complete must be a boolean")

    deduped, consistent = _dedup_scalar_components(components)
    if not evidence_complete or not consistent:
        return ComponentAggregation(
            operation, "insufficient_evidence", components=deduped
        )
    if not deduped:
        return ComponentAggregation(operation, "valid_empty", value=0)

    if operation == "option_count":
        identities: set[tuple[str, str, str]] = set()
        for component in deduped:
            partition = _partition_key(component)
            identities.update(
                (partition, program, code)
                for program, code in _option_identities(component)
            )
        return ComponentAggregation(
            operation, "complete", value=len(identities), components=deduped
        )

    if operation == "required_load":
        total = 0
        for component in deduped:
            if _component_kind(component) == "course":
                total += 1
                continue
            if component.get("membership_complete") is not True:
                return ComponentAggregation(
                    operation, "insufficient_evidence", components=deduped
                )
            choices = component.get("minimum_choices")
            if isinstance(choices, bool) or not isinstance(choices, int) or choices < 1:
                return ComponentAggregation(
                    operation, "insufficient_evidence", components=deduped
                )
            if len(_alternative_members(component)) < choices:
                return ComponentAggregation(
                    operation, "insufficient_evidence", components=deduped
                )
            total += choices
        return ComponentAggregation(
            operation, "complete", value=total, components=deduped
        )

    total = Decimal(0)
    for component in deduped:
        value = _numeric_credit(component.get("counted_credit_units"))
        if value is None:
            return ComponentAggregation(
                operation, "insufficient_evidence", components=deduped
            )
        total += value
    return ComponentAggregation(
        operation, "complete", value=_numeric_result(total), components=deduped
    )


def aggregate_option_count(
    components: Iterable[Mapping[str, Any]],
    *,
    evidence_complete: bool = True,
) -> ComponentAggregation:
    return aggregate_components(
        components, "option_count", evidence_complete=evidence_complete
    )


def aggregate_required_load(
    components: Iterable[Mapping[str, Any]],
    *,
    evidence_complete: bool = True,
) -> ComponentAggregation:
    return aggregate_components(
        components, "required_load", evidence_complete=evidence_complete
    )


def aggregate_sum_credits(
    components: Iterable[Mapping[str, Any]],
    *,
    evidence_complete: bool = True,
) -> ComponentAggregation:
    return aggregate_components(
        components, "sum_credits", evidence_complete=evidence_complete
    )


def aggregate_earliest(
    placements: Iterable[Mapping[str, Any]],
    *,
    evidence_complete: bool = True,
) -> EarliestAggregation:
    """Find tied earliest placement facts independently per partition."""
    if not isinstance(evidence_complete, bool):
        raise ValueError("evidence_complete must be a boolean")

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    malformed = False
    for placement in placements:
        if not isinstance(placement, Mapping):
            malformed = True
            continue
        try:
            partition_key = _partition_key(placement)
        except ValueError:
            malformed = True
            continue
        if _placement_earliest(placement) is None:
            malformed = True
        grouped.setdefault(partition_key, []).append(placement)

    if malformed or not evidence_complete:
        return EarliestAggregation(status="insufficient_evidence")
    if not grouped:
        return EarliestAggregation(status="valid_empty")

    partitions: list[EarliestPartition] = []
    for partition_key in sorted(grouped):
        facts = grouped[partition_key]
        with_values = [
            (placement, _placement_earliest(placement)) for placement in facts
        ]
        earliest = min(value for _, value in with_values if value is not None)
        tied = tuple(
            placement
            for placement, value in sorted(with_values, key=lambda item: _stable_key(item[0]))
            if value == earliest
        )
        partitions.append(
            EarliestPartition(
                partition=facts[0].get("partition", {}),
                value=earliest,
                placements=tied,
                provenance=_merge_provenance(tied),
            )
        )
    return EarliestAggregation(status="complete", partitions=tuple(partitions))


def compare_aggregates(left: Any, right: Any) -> ComparisonAggregation:
    """Compare two already-computed numeric or (year, semester) values."""
    left_operand = _comparison_operand(left)
    right_operand = _comparison_operand(right)
    if left_operand is None or right_operand is None:
        raise ValueError("comparison operands must be numeric or (year, semester) values")
    left_status, left_value = left_operand
    right_status, right_value = right_operand
    if left_status == "insufficient_evidence" or right_status == "insufficient_evidence":
        return ComparisonAggregation(
            status="insufficient_evidence",
            relation=None,
            left=left,
            right=right,
        )
    if left_value is None or right_value is None:
        return ComparisonAggregation(
            status="insufficient_evidence",
            relation=None,
            left=left,
            right=right,
        )
    if isinstance(left_value, (Real, Decimal)) and not isinstance(left_value, bool):
        if not isinstance(right_value, (Real, Decimal)) or isinstance(right_value, bool):
            raise ValueError("comparison operands must have the same value kind")
        if _numeric_credit(left_value) is None or _numeric_credit(right_value) is None:
            raise ValueError("numeric comparison operands must be finite")
    elif _valid_year_semester(left_value):
        if not _valid_year_semester(right_value):
            raise ValueError("comparison operands must have the same value kind")
    else:
        raise ValueError("comparison operands must be numeric or (year, semester) values")

    if left_value < right_value:
        relation = "less"
    elif left_value > right_value:
        relation = "greater"
    else:
        relation = "equal"
    return ComparisonAggregation(
        status="complete",
        relation=relation,
        left=left,
        right=right,
    )


def aggregate_course_set(
    components: Iterable[Mapping[str, Any]],
    *,
    evidence_complete: bool = True,
) -> CourseSetAggregation:
    """Deduplicate and aggregate one already-scoped course relation.

    Logical identity is either ``(program, course_code)`` for a concrete
    course or ``(program, alternative_group_id)`` for an alternative parent.
    The identity is deduped only within the exact supplied partition mapping.
    Duplicate components merge their provenance; plan/year/semester/category
    partition metadata is retained on each resulting component.
    """
    if not isinstance(evidence_complete, bool):
        raise ValueError("evidence_complete must be a boolean")

    grouped: dict[tuple[tuple[str, Any], str], list[Mapping[str, Any]]] = {}
    for component in components:
        if not isinstance(component, Mapping):
            raise ValueError("course components must be mappings")
        identity = _course_set_identity(component)
        partition_key = _partition_key(component)
        grouped.setdefault((identity, partition_key), []).append(component)

    courses: list[dict[str, Any]] = []
    for group in sorted(
        grouped, key=lambda key: (key[1], key[0][0], repr(key[0][1]))
    ):
        members = sorted(grouped[group], key=_stable_key)
        merged = dict(members[0])
        if group[0][0] == "alternative_group" and len(members) > 1:
            alternative_members: dict[
                tuple[Any, Any], list[Mapping[str, Any]]
            ] = {}
            for member in members:
                for alternative in _alternative_members(member):
                    identity = (
                        alternative.get("program", member.get("program")),
                        alternative.get("course_code"),
                    )
                    alternative_members.setdefault(identity, []).append(alternative)
            if alternative_members:
                merged["alternative_courses"] = tuple(
                    _merge_member_records(alternative_members[identity])
                    for identity in sorted(alternative_members, key=repr)
                )
        provenance = _merge_provenance(members)
        if provenance:
            merged["provenance"] = provenance
        courses.append(merged)

    if not evidence_complete:
        return CourseSetAggregation(
            status="insufficient_evidence",
            courses=tuple(courses),
        )
    if not courses:
        return CourseSetAggregation(status="valid_empty", count=0, exists=False)
    return CourseSetAggregation(
        status="complete",
        courses=tuple(courses),
        count=len(courses),
        exists=True,
    )


__all__ = [
    "AGGREGATION_OPERATIONS",
    "AGGREGATION_STATES",
    "COMPARISON_RELATIONS",
    "ComparisonAggregation",
    "ComponentAggregation",
    "CourseSetAggregation",
    "EarliestAggregation",
    "EarliestPartition",
    "PlanComparisonAggregation",
    "PlanComparisonInput",
    "PlanPlacementDifference",
    "aggregate_components",
    "aggregate_course_set",
    "aggregate_earliest",
    "aggregate_plan_comparison",
    "aggregate_option_count",
    "aggregate_required_load",
    "aggregate_sum_credits",
    "compare_aggregates",
]
