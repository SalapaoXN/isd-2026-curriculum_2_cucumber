"""Deterministic execution of relational/direct EvidencePlan requests."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
from pathlib import Path
import math
import re
import sqlite3
from types import MappingProxyType
from typing import Any

from rag.evidence_planner import EvidencePlan, EvidenceRequest, StructuralScope
from rag.retrieval.retrieve import (
    ConstrainedTopicRetrievalResult,
    SimilarityEvidence,
    aggregate_exact_course_similarity,
    fetch_course_description_evidence,
    retrieve_constrained_topic_evidence,
)
from rag.structured.queries import (
    applicable_plan_keys,
    course_facts,
    get_semester_credits,
    prerequisite_state,
    scoped_course_set,
)


EXECUTION_STATES = ("complete", "valid_empty", "insufficient_evidence")
DIRECT_PRIMITIVES = (
    "course_set",
    "placement_facts",
    "credit_facts",
    "prerequisite_facts",
    "description_evidence",
)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    return value


def _scope_payload(scope: StructuralScope) -> dict[str, Any]:
    return {
        "program": scope.program,
        "plans": scope.plans,
        "years": scope.years,
        "semesters": scope.semesters,
        "category": scope.category,
        "credit_units": getattr(scope, "credit_units", None),
        "group_by": scope.group_by,
    }


def _valid_provenance(value: Any) -> bool:
    if isinstance(value, Mapping):
        if "provenance" in value:
            provenance = value["provenance"]
            if (
                not isinstance(provenance, (list, tuple))
                or not provenance
                or any(not isinstance(reference, Mapping) for reference in provenance)
            ):
                return False
        return all(_valid_provenance(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_valid_provenance(item) for item in value)
    return True


def _payload_has_provenance(value: Any) -> bool:
    if isinstance(value, Mapping):
        if isinstance(value.get("provenance"), (list, tuple)) and value["provenance"]:
            return _valid_provenance(value)
        return any(_payload_has_provenance(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_payload_has_provenance(item) for item in value)
    return False


def _status_for_payload(payload: Any, *, empty: bool = False) -> str:
    if empty:
        return "valid_empty"
    if not _payload_has_provenance(payload) or not _valid_provenance(payload):
        return "insufficient_evidence"
    return "complete"


@dataclass(frozen=True, slots=True)
class DirectPrerequisiteRequirement:
    """One canonical direct prerequisite requirement group."""

    kind: str
    requirement_type: str | None
    prerequisite_course_id: int | None
    prerequisite_code: str | None
    prerequisite_name_th: str | None
    prerequisite_name_en: str | None
    alternative_group_id: int | None
    minimum_choices: int | None
    maximum_choices: int | None
    alternative_members: tuple[Mapping[str, Any], ...]
    provenance: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True, slots=True)
class DirectPrerequisiteBurden:
    """Canonical direct-prerequisite facts for one grounded candidate."""

    program: str
    course_id: int
    course_code: str
    status: str
    required_course_count: int
    alternative_group_count: int
    alternative_member_counts: tuple[int, ...]
    ordered_requirement_groups: tuple[DirectPrerequisiteRequirement, ...]
    provenance: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True, slots=True)
class DirectPrerequisiteBurdenResult:
    """Typed candidate-set result; incomplete candidates fail the whole pack."""

    status: str
    burdens: tuple[DirectPrerequisiteBurden, ...] = ()
    provenance: tuple[Mapping[str, Any], ...] = ()
    primitive_state: str | None = None


def _burden_provenance(value: Any) -> tuple[Mapping[str, Any], ...] | None:
    if not isinstance(value, (list, tuple)) or not value:
        return None
    if any(not isinstance(reference, Mapping) for reference in value):
        return None
    references = tuple(_freeze(reference) for reference in value)
    if not _valid_provenance({"provenance": references}):
        return None
    return references


def _merge_burden_provenance(
    *values: Iterable[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    merged: list[Mapping[str, Any]] = []
    for value in values:
        for reference in value:
            frozen = _freeze(reference)
            if frozen not in merged:
                merged.append(frozen)
    return tuple(merged)


def _alternative_group_signature(
    record: Mapping[str, Any],
    group: Mapping[str, Any],
    members: tuple[Mapping[str, Any], ...],
) -> tuple[Any, ...]:
    return (
        record.get("requirement_type"),
        group.get("group_key"),
        group.get("label"),
        group.get("minimum_choices"),
        group.get("maximum_choices"),
        group.get("notes"),
        tuple(
            (
                member.get("course_id"),
                member.get("course_code"),
                member.get("name_th"),
                member.get("name_en"),
                member.get("requirement_type"),
            )
            for member in members
        ),
    )


def _normalize_direct_prerequisite_record(
    record: Mapping[str, Any],
) -> tuple[DirectPrerequisiteRequirement, tuple[Any, ...] | None] | None:
    record_provenance = _burden_provenance(record.get("provenance"))
    if record_provenance is None:
        return None

    group_id = record.get("alternative_group_id")
    if group_id is None:
        if record.get("is_alternative"):
            return None
        prerequisite_course_id = record.get("prerequisite_course_id")
        prerequisite_code = record.get("prerequisite_code")
        if (
            isinstance(prerequisite_course_id, bool)
            or not isinstance(prerequisite_course_id, int)
            or not isinstance(prerequisite_code, str)
            or not prerequisite_code.strip()
        ):
            return None
        return (
            DirectPrerequisiteRequirement(
                kind="required_course",
                requirement_type=record.get("requirement_type"),
                prerequisite_course_id=prerequisite_course_id,
                prerequisite_code=prerequisite_code,
                prerequisite_name_th=record.get("prerequisite_name_th"),
                prerequisite_name_en=record.get("prerequisite_name_en"),
                alternative_group_id=None,
                minimum_choices=None,
                maximum_choices=None,
                alternative_members=(),
                provenance=record_provenance,
            ),
            None,
        )

    if (
        isinstance(group_id, bool)
        or not isinstance(group_id, int)
        or not record.get("is_alternative")
    ):
        return None
    group = record.get("alternative_group")
    members = record.get("alternative_courses")
    if not isinstance(group, Mapping) or not isinstance(members, (list, tuple)):
        return None
    if group.get("alternative_group_id") != group_id or not members:
        return None
    minimum_choices = group.get("minimum_choices")
    maximum_choices = group.get("maximum_choices")
    if (
        isinstance(minimum_choices, bool)
        or not isinstance(minimum_choices, int)
        or isinstance(maximum_choices, bool)
        or not isinstance(maximum_choices, int)
        or minimum_choices < 1
        or maximum_choices < minimum_choices
    ):
        return None

    normalized_members: list[Mapping[str, Any]] = []
    member_ids: set[int] = set()
    for member in members:
        if not isinstance(member, Mapping):
            return None
        member_id = member.get("course_id")
        member_code = member.get("course_code")
        if (
            isinstance(member_id, bool)
            or not isinstance(member_id, int)
            or member_id in member_ids
            or not isinstance(member_code, str)
            or not member_code.strip()
            or _burden_provenance(member.get("provenance")) is None
        ):
            return None
        member_ids.add(member_id)
        normalized_members.append(_freeze(member))
    normalized_members_tuple = tuple(normalized_members)
    signature = _alternative_group_signature(
        record,
        group,
        normalized_members_tuple,
    )
    return (
        DirectPrerequisiteRequirement(
            kind="alternative_group",
            requirement_type=record.get("requirement_type"),
            prerequisite_course_id=None,
            prerequisite_code=None,
            prerequisite_name_th=None,
            prerequisite_name_en=None,
            alternative_group_id=group_id,
            minimum_choices=minimum_choices,
            maximum_choices=maximum_choices,
            alternative_members=normalized_members_tuple,
            provenance=record_provenance,
        ),
        signature,
    )


def _candidate_identity(candidate: Mapping[str, Any]) -> tuple[str, int, str] | None:
    program = candidate.get("program")
    course_id = candidate.get("course_id")
    course_code = candidate.get("course_code")
    if (
        not isinstance(program, str)
        or not program.strip()
        or isinstance(course_id, bool)
        or not isinstance(course_id, int)
        or not isinstance(course_code, str)
        or not course_code.strip()
    ):
        return None
    return (program.strip(), course_id, course_code.strip())


def _build_candidate_burden(
    db_path: str,
    candidate: Mapping[str, Any],
) -> DirectPrerequisiteBurden | None:
    identity = _candidate_identity(candidate)
    if identity is None:
        return None
    program, course_id, course_code = identity
    state = prerequisite_state(db_path, course_id)
    if not isinstance(state, Mapping):
        return None
    state_provenance = _burden_provenance(state.get("provenance"))
    if state_provenance is None:
        return None
    state_name = state.get("state")
    if state_name == "explicit_none":
        return DirectPrerequisiteBurden(
            program=program,
            course_id=course_id,
            course_code=course_code,
            status="complete",
            required_course_count=0,
            alternative_group_count=0,
            alternative_member_counts=(),
            ordered_requirement_groups=(),
            provenance=state_provenance,
        )
    if state_name != "required":
        return None
    records = state.get("records")
    if not isinstance(records, (list, tuple)) or not records:
        return None

    groups: list[DirectPrerequisiteRequirement] = []
    alternative_signatures: dict[int, tuple[Any, ...]] = {}
    required_course_count = 0
    for record in records:
        if not isinstance(record, Mapping):
            return None
        normalized = _normalize_direct_prerequisite_record(record)
        if normalized is None:
            return None
        requirement, signature = normalized
        if requirement.kind == "required_course":
            required_course_count += 1
            groups.append(requirement)
            continue
        group_id = requirement.alternative_group_id
        if group_id is None or signature is None:
            return None
        previous_signature = alternative_signatures.get(group_id)
        if previous_signature is not None:
            if previous_signature != signature:
                return None
            continue
        alternative_signatures[group_id] = signature
        groups.append(requirement)

    return DirectPrerequisiteBurden(
        program=program,
        course_id=course_id,
        course_code=course_code,
        status="complete",
        required_course_count=required_course_count,
        alternative_group_count=len(alternative_signatures),
        alternative_member_counts=tuple(
            len(requirement.alternative_members)
            for requirement in groups
            if requirement.kind == "alternative_group"
        ),
        ordered_requirement_groups=tuple(groups),
        provenance=_merge_burden_provenance(
            state_provenance,
            *(requirement.provenance for requirement in groups),
        ),
    )


def build_direct_prerequisite_burden(
    db_path: str,
    candidates: Iterable[Mapping[str, Any]],
) -> DirectPrerequisiteBurdenResult:
    """Build canonical direct-prerequisite burden facts for grounded candidates.

    This helper deliberately does not traverse prerequisites transitively, rank
    candidates, or classify a burden as ``few`` or ``many``.  It is an internal
    evidence builder; every candidate must be complete for the pack to pass.
    """
    candidate_list = tuple(candidates)
    if not candidate_list:
        return DirectPrerequisiteBurdenResult(status="valid_empty")

    unique_candidates: list[Mapping[str, Any]] = []
    seen_identities: set[tuple[str, int, str]] = set()
    for candidate in candidate_list:
        if not isinstance(candidate, Mapping):
            return DirectPrerequisiteBurdenResult(
                status="insufficient_evidence",
                primitive_state="invalid_candidate",
            )
        identity = _candidate_identity(candidate)
        if identity is None:
            return DirectPrerequisiteBurdenResult(
                status="insufficient_evidence",
                primitive_state="invalid_candidate",
            )
        if identity in seen_identities:
            continue
        seen_identities.add(identity)
        unique_candidates.append(candidate)

    burdens: list[DirectPrerequisiteBurden] = []
    for candidate in unique_candidates:
        burden = _build_candidate_burden(db_path, candidate)
        if burden is None:
            return DirectPrerequisiteBurdenResult(
                status="insufficient_evidence",
                primitive_state="prerequisite_burden_incomplete",
            )
        burdens.append(burden)
    provenance = _merge_burden_provenance(*(burden.provenance for burden in burdens))
    return DirectPrerequisiteBurdenResult(
        status="complete",
        burdens=tuple(burdens),
        provenance=provenance,
    )


@dataclass(frozen=True, slots=True)
class EvidenceExecutionResult:
    """One typed primitive result for one planned/effective scope."""

    request_id: str
    kind: str
    planned_request: EvidenceRequest
    effective_scope: StructuralScope
    status: str
    payload: Any = None
    primitive_state: str | None = None

    def __post_init__(self) -> None:
        if self.status not in EXECUTION_STATES:
            raise ValueError(f"unsupported execution status: {self.status!r}")
        if self.kind not in DIRECT_PRIMITIVES and self.kind != "topic_matches":
            raise ValueError(f"unsupported execution primitive: {self.kind!r}")
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("request_id must be a non-empty string")
        if not isinstance(self.planned_request, EvidenceRequest):
            raise TypeError("planned_request must be an EvidenceRequest")
        if not isinstance(self.effective_scope, StructuralScope):
            raise TypeError("effective_scope must be a StructuralScope")
        object.__setattr__(self, "payload", _freeze(self.payload))


@dataclass(frozen=True, slots=True)
class EvidenceBundle:
    """Immutable execution results retaining the original EvidencePlan."""

    plan: EvidencePlan
    results: tuple[EvidenceExecutionResult, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.plan, EvidencePlan):
            raise TypeError("plan must be an EvidencePlan")
        results = tuple(self.results)
        if any(not isinstance(result, EvidenceExecutionResult) for result in results):
            raise TypeError("results must contain EvidenceExecutionResult values")
        object.__setattr__(self, "results", results)


def _graph_order(plan: EvidencePlan) -> tuple[EvidenceRequest, ...] | None:
    requests = tuple(plan.requests)
    by_id = {request.request_id: request for request in requests}
    if len(by_id) != len(requests):
        return None
    positions = {request.request_id: index for index, request in enumerate(requests)}
    indegree = {request.request_id: len(request.depends_on) for request in requests}
    dependents: dict[str, list[str]] = {request.request_id: [] for request in requests}
    for request in requests:
        for dependency in request.depends_on:
            if dependency not in by_id or dependency == request.request_id:
                return None
            dependents[dependency].append(request.request_id)

    ready = [request.request_id for request in requests if indegree[request.request_id] == 0]
    ordered: list[EvidenceRequest] = []
    while ready:
        ready.sort(key=positions.__getitem__)
        request_id = ready.pop(0)
        ordered.append(by_id[request_id])
        for dependent in dependents[request_id]:
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                ready.append(dependent)
    if len(ordered) != len(requests):
        return None
    return tuple(ordered)


def _course_target_keys(targets: Iterable[Mapping[str, Any]]) -> tuple[tuple[Any, Any], ...]:
    keys: list[tuple[Any, Any]] = []
    for target in targets:
        if not isinstance(target, Mapping):
            continue
        key = (target.get("course_id"), target.get("course_code"))
        if key not in keys:
            keys.append(key)
    return tuple(keys)


def _axis_values(
    db_path: str,
    scope: StructuralScope,
    plan_key: str,
    *,
    axis: str,
    years: tuple[int, ...] = (),
) -> tuple[int, ...]:
    result = scoped_course_set(
        db_path,
        scope.program or "",
        plan_key,
        years=years,
        category=scope.category,
        course_targets=scope.course_targets,
    )
    values: set[int] = set()
    for course in result.get("courses", ()):
        for choice in course.get("year_semester_choices", ()):
            if isinstance(choice, (list, tuple)) and len(choice) == 2:
                value = choice[0] if axis == "year" else choice[1]
                if isinstance(value, int) and not isinstance(value, bool):
                    values.add(value)
    return tuple(sorted(values))


def _is_flexible_only_exact_credit_request(
    db_path: str,
    request: EvidenceRequest,
    plan_keys: Iterable[str],
    targets: tuple[Mapping[str, Any], ...],
) -> bool:
    if (
        request.kind != "credit_facts"
        or request.scope.years
        or request.scope.semesters
        or request.scope.group_by
        or len(targets) != 1
        or not isinstance(request.scope.program, str)
        or not request.scope.program.strip()
    ):
        return False

    placements: list[Mapping[str, Any]] = []
    for plan_key in plan_keys:
        result = scoped_course_set(
            db_path,
            request.scope.program,
            plan_key,
            course_targets=targets,
        )
        plan_courses = result.get("courses")
        if not isinstance(plan_courses, (list, tuple)) or not plan_courses:
            return False
        placements.extend(
            course
            for course in plan_courses
            if isinstance(course, Mapping)
        )

    return bool(placements) and all(
        placement.get("year_number") is None
        and placement.get("semester_number") is None
        for placement in placements
    )


def _materialize_scopes(
    db_path: str,
    request: EvidenceRequest,
) -> tuple[StructuralScope, ...]:
    source = request.scope
    if not isinstance(source.program, str) or not source.program.strip():
        raise ValueError("answerable execution needs a resolved program")

    request_targets = request.course_targets or source.course_targets
    materialization_source = replace(source, course_targets=request_targets)
    plan_keys = source.plans or applicable_plan_keys(db_path, source.program)
    if not plan_keys:
        return (materialization_source,)

    if _is_flexible_only_exact_credit_request(
        db_path,
        request,
        plan_keys,
        request_targets,
    ):
        return (
            replace(
                materialization_source,
                plans=tuple(plan_keys),
                expand_applicable=tuple(
                    axis
                    for axis in materialization_source.expand_applicable
                    if axis != "plan"
                ),
                unconstrained=tuple(
                    axis
                    for axis in materialization_source.unconstrained
                    if axis != "plan"
                ),
            ),
        )

    scopes: list[StructuralScope] = []
    for plan_key in plan_keys:
        expand = set(source.expand_applicable)
        unconstrained = set(source.unconstrained)
        expand.discard("plan")
        unconstrained.discard("plan")

        year_materialized = False
        if source.years:
            year_options = (
                tuple((year,) for year in source.years)
                if "year" in source.group_by or request.kind == "credit_facts"
                else (source.years,)
            )
            expand.discard("year")
        elif (
            "year" in expand
            or "year" in unconstrained
            or "year" in source.group_by
            or request.kind == "credit_facts"
        ):
            values = _axis_values(db_path, materialization_source, plan_key, axis="year")
            year_options = (
                tuple((value,) for value in values)
                if "year" in source.group_by or request.kind == "credit_facts"
                else (values,)
            ) or ((),)
            expand.discard("year")
            unconstrained.discard("year")
            year_materialized = True
        else:
            year_options = ((),)

        for selected_years in year_options:
            semester_materialized = False
            if source.semesters:
                semester_options = (
                    tuple((semester,) for semester in source.semesters)
                    if "semester" in source.group_by or request.kind == "credit_facts"
                    else (source.semesters,)
                )
                expand.discard("semester")
            elif (
                "semester" in expand
                or "semester" in unconstrained
                or "semester" in source.group_by
                or request.kind == "credit_facts"
            ):
                values = _axis_values(
                    db_path,
                    materialization_source,
                    plan_key,
                    axis="semester",
                    years=selected_years,
                )
                semester_options = (
                    tuple((value,) for value in values)
                    if "semester" in source.group_by
                    or request.kind == "credit_facts"
                    else (values,)
                ) or ((),)
                expand.discard("semester")
                unconstrained.discard("semester")
                semester_materialized = True
            else:
                semester_options = ((),)

            for selected_semesters in semester_options:
                targets = request_targets
                if "course" in source.group_by and len(targets) > 1:
                    target_options = tuple((target,) for target in targets)
                else:
                    target_options = (targets,)
                for selected_targets in target_options:
                    selected_expand = set(expand)
                    selected_unconstrained = set(unconstrained)
                    if selected_years:
                        selected_expand.discard("year")
                        selected_unconstrained.discard("year")
                    if year_materialized:
                        selected_expand.discard("year")
                        selected_unconstrained.discard("year")
                    if selected_semesters:
                        selected_expand.discard("semester")
                        selected_unconstrained.discard("semester")
                    if semester_materialized:
                        selected_expand.discard("semester")
                        selected_unconstrained.discard("semester")
                    if selected_targets:
                        selected_unconstrained.discard("course")
                    scopes.append(
                        StructuralScope(
                            program=source.program,
                            plans=(plan_key,),
                            years=selected_years or source.years,
                            semesters=selected_semesters or source.semesters,
                            category=source.category,
                            credit_units=getattr(source, "credit_units", None),
                            expand_applicable=tuple(
                                axis for axis in source.expand_applicable
                                if axis in selected_expand
                            ),
                            unconstrained=tuple(
                                axis for axis in source.unconstrained
                                if axis in selected_unconstrained
                            ),
                            course_targets=selected_targets,
                            group_by=source.group_by,
                        )
                    )
    return tuple(scopes)


def _result(
    request: EvidenceRequest,
    scope: StructuralScope,
    status: str,
    payload: Any = None,
    primitive_state: str | None = None,
) -> EvidenceExecutionResult:
    return EvidenceExecutionResult(
        request_id=request.request_id,
        kind=request.kind,
        planned_request=request,
        effective_scope=scope,
        status=status,
        payload=payload,
        primitive_state=primitive_state,
    )


def _execute_course_set(
    db_path: str,
    request: EvidenceRequest,
    scope: StructuralScope,
    *,
    exact_term_placements: bool = False,
) -> EvidenceExecutionResult:
    result = scoped_course_set(
        db_path,
        scope.program or "",
        scope.plans,
        years=scope.years,
        semesters=scope.semesters,
        category=scope.category,
        course_targets=scope.course_targets,
        exact_term_placements=exact_term_placements,
        credit_units=getattr(scope, "credit_units", None),
    )
    if result.get("status") == "insufficient_evidence":
        return _result(
            request, scope, "insufficient_evidence", result, "credit_filter_incomplete"
        )
    courses = tuple(result.get("courses", ()))
    if result.get("status") == "no_data":
        return _result(request, scope, "valid_empty", result, "empty_relation")
    status = _status_for_payload(result)
    return _result(request, scope, status, result)


def _execute_placement(
    db_path: str,
    request: EvidenceRequest,
    scope: StructuralScope,
) -> EvidenceExecutionResult:
    result = scoped_course_set(
        db_path,
        scope.program or "",
        scope.plans,
        years=scope.years,
        semesters=scope.semesters,
        category=scope.category,
        course_targets=scope.course_targets,
        credit_units=getattr(scope, "credit_units", None),
    )
    if result.get("status") == "insufficient_evidence":
        return _result(
            request, scope, "insufficient_evidence", result, "credit_filter_incomplete"
        )
    if result.get("status") == "no_data":
        return _result(request, scope, "valid_empty", result, "empty_relation")
    return _result(request, scope, _status_for_payload(result), result)


_CATEGORY_CREDIT_RE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)")


def _parse_category_credit(value: Any) -> Decimal | None:
    """Parse one canonical per-course credit value with explicit None handling.

    Mirrors the lenient leading-number read of the existing semester-total
    path, but returns None (unknown) instead of silently using zero. Never
    uses truthiness, so explicit ``0`` credits keep working.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        try:
            parsed = Decimal(str(value))
        except InvalidOperation:
            return None
        return parsed if parsed.is_finite() else None
    if isinstance(value, str):
        if not value.strip():
            return None
        match = _CATEGORY_CREDIT_RE.search(value)
        if match is None:
            return None
        try:
            parsed = Decimal(match.group(1))
        except InvalidOperation:
            return None
        return parsed if parsed.is_finite() else None
    return None


def _has_category_override(value: Any) -> bool:
    """Return True only when a credit override is explicitly present."""
    if value is None:
        return False
    if isinstance(value, str) and not value.strip():
        return False
    return True


def _counted_category_credit(course: Mapping[str, Any]) -> Decimal | None:
    """Return the counted credit for one filtered logical course, if provable.

    Frozen H23/H27-A alternative rule: an explicit placement override wins;
    otherwise every member must carry a known equal credit (member
    ``credits`` with ``credits_raw`` fallback, mirroring the list filter).
    Mixed/unknown/conflicting members, invalid choice counts, or missing
    credits yield None so the caller fails closed. Never chooses a member.
    """
    override = course.get("credits_override")
    if _has_category_override(override):
        return _parse_category_credit(override)
    if not course.get("is_alternative"):
        return _parse_category_credit(course.get("credits"))
    members = course.get("alternative_courses", ())
    if not isinstance(members, (list, tuple)) or not members:
        return None
    units: list[Decimal] = []
    for member in members:
        if not isinstance(member, Mapping):
            return None
        unit = _parse_category_credit(member.get("credits"))
        if unit is None:
            unit = _parse_category_credit(member.get("credits_raw"))
        if unit is None:
            return None
        units.append(unit)
    if len(set(units)) != 1:
        return None
    choices = course.get("minimum_choices")
    if isinstance(choices, bool) or not isinstance(choices, int) or choices < 1:
        return None
    if len(members) < choices:
        return None
    return units[0] * choices


def _execute_category_credit(
    db_path: str,
    request: EvidenceRequest,
    scope: StructuralScope,
    *,
    course_targets: Iterable[Mapping[str, Any]] = (),
) -> EvidenceExecutionResult:
    """Compose a category-restricted credit total from the filtered set.

    Defensive second layer behind the H27-B completeness guard: only concrete
    single plan/year/semester scopes execute; anything else fails closed.
    Components mirror the existing credit payload shape (with provenance from
    the filtered set only) so downstream aggregation/dedup contracts apply
    unchanged. Empty filtered sets are valid-empty (0).
    """
    if (
        len(scope.plans) != 1
        or len(scope.years) != 1
        or len(scope.semesters) != 1
    ):
        return _result(
            request,
            scope,
            "insufficient_evidence",
            primitive_state="credit_scope_not_concrete",
        )
    try:
        filtered = scoped_course_set(
            db_path,
            scope.program or "",
            scope.plans,
            years=scope.years,
            semesters=scope.semesters,
            category=scope.category,
            course_targets=tuple(course_targets or ()),
        )
    except (FileNotFoundError, OSError, sqlite3.Error, TypeError, ValueError, KeyError):
        return _result(
            request,
            scope,
            "insufficient_evidence",
            primitive_state="execution_failure",
        )
    if not isinstance(filtered, Mapping) or filtered.get("status") == "insufficient_evidence":
        return _result(
            request,
            scope,
            "insufficient_evidence",
            filtered if isinstance(filtered, Mapping) else None,
            "category_credit_incomplete",
        )
    if filtered.get("status") == "no_data":
        return _result(request, scope, "valid_empty", filtered, "empty_relation")
    courses = filtered.get("courses", ())
    if not isinstance(courses, (list, tuple)):
        return _result(
            request,
            scope,
            "insufficient_evidence",
            primitive_state="category_credit_incomplete",
        )
    components: list[dict[str, Any]] = []
    for course in courses:
        if not isinstance(course, Mapping):
            return _result(
                request,
                scope,
                "insufficient_evidence",
                primitive_state="category_credit_incomplete",
            )
        counted = _counted_category_credit(course)
        if counted is None:
            return _result(
                request,
                scope,
                "insufficient_evidence",
                filtered,
                "category_credit_incomplete",
            )
        numeric: int | float = (
            int(counted) if counted == counted.to_integral_value() else float(counted)
        )
        components.append(
            {
                "placement_id": course.get("placement_id"),
                "program": course.get("program"),
                "plan_key": course.get("plan_key"),
                "course_id": course.get("course_id"),
                "course_code": course.get("course_code"),
                "name_th": course.get("name_th"),
                "name_en": course.get("name_en"),
                "credits_raw": course.get("credits_raw"),
                "credit_units": (
                    None if course.get("is_alternative") else counted
                ),
                "counted_credit_units": numeric,
                "alternative_group_id": course.get("alternative_group_id"),
                "alternative_courses": course.get("alternative_courses", ()),
                "year": course.get("year_number"),
                "semester": course.get("semester_number"),
                "provenance": tuple(course.get("provenance", ())),
            }
        )
    payload = {
        "status": "ok",
        "program": scope.program,
        "plan_key": scope.plans[0],
        "year": scope.years[0],
        "semester": scope.semesters[0],
        "total_credits": None,
        "plans": (),
        "components": tuple(components),
    }
    return _result(request, scope, _status_for_payload(payload), payload)


def _execute_credit(
    db_path: str,
    request: EvidenceRequest,
    scope: StructuralScope,
    *,
    course_targets: Iterable[Mapping[str, Any]] | None = None,
) -> EvidenceExecutionResult:
    targets = tuple(course_targets or ())
    if scope.category is not None:
        # H27-B: never route a category-filtered sum through the
        # category-blind get_semester_credits() path. Compose the total from
        # the canonical filtered course set instead; fail closed on any
        # missing/conflicting credit evidence.
        return _execute_category_credit(
            db_path,
            request,
            scope,
            course_targets=tuple(targets) or tuple(scope.course_targets),
        )
    if targets and (not scope.years or not scope.semesters):
        if len(targets) != 1:
            return _result(
                request,
                scope,
                "insufficient_evidence",
                primitive_state="course_credit_target_not_single",
            )
        target = targets[0]
        course_code = target.get("course_code")
        if not isinstance(course_code, str) or not course_code.strip():
            return _result(
                request,
                scope,
                "insufficient_evidence",
                primitive_state="invalid_course_credit_target",
            )
        try:
            direct = course_facts(db_path, course_code.strip(), scope.program)
        except (FileNotFoundError, OSError, sqlite3.Error, TypeError, ValueError, KeyError):
            return _result(
                request,
                scope,
                "insufficient_evidence",
                primitive_state="course_credit_lookup_failure",
            )
        facts = direct.get("courses") if isinstance(direct, Mapping) else None
        if not isinstance(facts, (list, tuple)):
            return _result(
                request,
                scope,
                "insufficient_evidence",
                primitive_state="course_credit_missing",
            )
        target_id = target.get("course_id")
        target_catalog = target.get("catalog_id")
        matching = [
            fact
            for fact in facts
            if isinstance(fact, Mapping)
            and fact.get("course_code") == course_code.strip()
        ]
        if isinstance(target_id, int) and not isinstance(target_id, bool):
            identified = [fact for fact in matching if fact.get("course_id") == target_id]
            if identified:
                matching = identified
        elif isinstance(target_catalog, int) and not isinstance(target_catalog, bool):
            identified = [fact for fact in matching if fact.get("catalog_id") == target_catalog]
            if identified:
                matching = identified
        if len(matching) != 1:
            return _result(
                request,
                scope,
                "insufficient_evidence",
                primitive_state="course_credit_ambiguous_or_missing",
            )
        fact = matching[0]
        credit_units = fact.get("credit_units")
        provenance = fact.get("provenance")
        if (
            isinstance(credit_units, bool)
            or not isinstance(credit_units, (int, float))
            or not isinstance(provenance, (list, tuple))
            or not provenance
        ):
            return _result(
                request,
                scope,
                "insufficient_evidence",
                primitive_state="course_credit_incomplete",
            )
        component = {
            "course_id": fact.get("course_id"),
            "catalog_id": fact.get("catalog_id"),
            "program": scope.program,
            "plan_key": scope.plans[0] if len(scope.plans) == 1 else None,
            "course_code": fact.get("course_code"),
            "name_th": fact.get("name_th"),
            "name_en": fact.get("name_en"),
            "credits_raw": fact.get("credits_raw"),
            "credit_units": credit_units,
            "counted_credit_units": credit_units,
            "alternative_group_id": None,
            "alternative_courses": (),
            "year": None,
            "semester": None,
            "provenance": tuple(provenance),
        }
        payload = {
            "status": "ok",
            "program": scope.program,
            "components": (component,),
            "provenance": tuple(provenance),
        }
        return _result(request, scope, "complete", payload)
    if (
        len(scope.plans) != 1
        or len(scope.years) != 1
        or len(scope.semesters) != 1
    ):
        return _result(
            request,
            scope,
            "insufficient_evidence",
            primitive_state="credit_scope_not_concrete",
        )
    credit_args = (
        db_path,
        scope.program or "",
        scope.plans[0],
        scope.years[0],
        scope.semesters[0],
    )
    if not targets:
        result = get_semester_credits(*credit_args)
    else:
        result = get_semester_credits(*credit_args, course_targets=targets)
    if result.get("status") == "no_data":
        return _result(request, scope, "valid_empty", result, "empty_relation")
    return _result(request, scope, _status_for_payload(result), result)


def _execute_prerequisites(
    db_path: str,
    request: EvidenceRequest,
    scope: StructuralScope,
) -> EvidenceExecutionResult:
    targets = request.course_targets or scope.course_targets
    course_ids = _scoped_course_ids(db_path, scope, targets)
    if course_ids is None:
        return _result(
            request,
            scope,
            "insufficient_evidence",
            primitive_state="invalid_course_target",
        )
    records: list[Mapping[str, Any]] = []
    explicit_none_records: list[Mapping[str, Any]] = []
    for course_id in course_ids:
        state = prerequisite_state(db_path, course_id)
        if state.get("state") == "unknown":
            return _result(
                request,
                scope,
                "insufficient_evidence",
                primitive_state="prerequisite_state_unknown",
            )
        if state.get("state") == "required":
            records.extend(state.get("records", ()))
        elif state.get("state") == "explicit_none":
            explicit_none_records.append(
                {
                    "course_id": course_id,
                    "prerequisite_state": "explicit_none",
                    "prerequisite_text": state.get("prerequisite_text"),
                    "provenance": tuple(state.get("provenance", ())),
                }
            )
    payload = tuple(records)
    if not payload:
        return _result(
            request,
            scope,
            "valid_empty",
            tuple(explicit_none_records),
            "explicit_none",
        )
    return _result(request, scope, _status_for_payload(payload), payload)


def _scoped_course_ids(
    db_path: str,
    scope: StructuralScope,
    targets: Iterable[Mapping[str, Any]],
) -> tuple[int, ...] | None:
    """Resolve logical targets to physical rows in one concrete scope."""
    result = scoped_course_set(
        db_path,
        scope.program or "",
        scope.plans,
        years=scope.years,
        semesters=scope.semesters,
        category=scope.category,
        course_targets=targets,
        credit_units=getattr(scope, "credit_units", None),
    )
    if not isinstance(result, Mapping) or result.get("status") != "ok":
        return None
    courses = result.get("courses")
    if not isinstance(courses, (list, tuple)):
        return None

    physical_ids: dict[tuple[str, str], list[int]] = {}
    for course in courses:
        if not isinstance(course, Mapping):
            return None
        members = course.get("alternative_courses", ())
        records = members if course.get("is_alternative") else (course,)
        if not isinstance(records, (list, tuple)):
            return None
        for record in records:
            if not isinstance(record, Mapping):
                return None
            program = record.get("program") or course.get("program")
            course_code = record.get("course_code")
            course_id = record.get("course_id")
            if (
                not isinstance(program, str)
                or not program.strip()
                or not isinstance(course_code, str)
                or not course_code.strip()
                or isinstance(course_id, bool)
                or not isinstance(course_id, int)
            ):
                continue
            key = (program.strip().upper(), course_code.strip())
            ids = physical_ids.setdefault(key, [])
            if course_id not in ids:
                ids.append(course_id)

    resolved: list[int] = []
    for target in targets:
        if not isinstance(target, Mapping):
            return None
        program = target.get("program")
        course_code = target.get("course_code")
        if (
            not isinstance(program, str)
            or not program.strip()
            or not isinstance(course_code, str)
            or not course_code.strip()
        ):
            return None
        ids = physical_ids.get((program.strip().upper(), course_code.strip()), [])
        if len(ids) != 1:
            return None
        resolved.append(ids[0])
    return tuple(resolved)


def _execute_descriptions(
    db_path: str,
    request: EvidenceRequest,
    scope: StructuralScope,
) -> EvidenceExecutionResult:
    if not request.course_targets:
        return _result(request, scope, "insufficient_evidence", primitive_state="missing_course_target")
    course_ids = _scoped_course_ids(
        db_path,
        scope,
        request.course_targets,
    )
    if course_ids is None:
        return _result(request, scope, "insufficient_evidence", primitive_state="description_missing")
    evidence: list[dict[str, Any]] = []
    for target, course_id in zip(request.course_targets, course_ids, strict=True):
        for item in fetch_course_description_evidence(db_path, course_id):
            enriched = dict(item)
            enriched.setdefault("program", target.get("program"))
            enriched.setdefault("course_code", target.get("course_code"))
            enriched["partition"] = _scope_payload(scope)
            evidence.append(enriched)
    if not evidence:
        return _result(request, scope, "insufficient_evidence", (), "description_missing")
    return _result(request, scope, _status_for_payload(evidence), tuple(evidence))


def _topic_dependency_result(
    request: EvidenceRequest,
    dependency: EvidenceExecutionResult,
) -> EvidenceExecutionResult:
    if dependency.status == "insufficient_evidence":
        return _result(
            request,
            dependency.effective_scope,
            "insufficient_evidence",
            primitive_state=dependency.primitive_state or "dependency_insufficient",
        )
    if dependency.status == "valid_empty":
        return _result(
            request,
            dependency.effective_scope,
            "valid_empty",
            dependency.payload,
            "empty_structural_candidates",
        )
    return _result(
        request,
        dependency.effective_scope,
        "insufficient_evidence",
        primitive_state="invalid_course_set_dependency",
    )


def _execute_topic_matches(
    db_path: str,
    request: EvidenceRequest,
    dependency: EvidenceExecutionResult,
) -> EvidenceExecutionResult:
    scope = dependency.effective_scope
    payload = dependency.payload
    if not isinstance(payload, Mapping):
        return _result(
            request,
            scope,
            "insufficient_evidence",
            primitive_state="malformed_course_set_dependency",
        )
    candidates = payload.get("courses")
    if not isinstance(candidates, (list, tuple)) or any(
        not isinstance(candidate, Mapping) for candidate in candidates
    ):
        return _result(
            request,
            scope,
            "insufficient_evidence",
            primitive_state="malformed_course_set_dependency",
        )
    try:
        retrieval = retrieve_constrained_topic_evidence(
            db_path,
            request.topic or "",
            tuple(candidates),
        )
    except (OSError, TypeError, ValueError, KeyError, sqlite3.Error):
        return _result(
            request,
            scope,
            "insufficient_evidence",
            primitive_state="topic_retrieval_failure",
        )

    state = retrieval.status
    if state == "scored":
        return _result(request, scope, "complete", retrieval, state)
    if state == "no_threshold_matches":
        return _result(request, scope, "valid_empty", retrieval, state)
    if state == "description_missing":
        return _result(request, scope, "insufficient_evidence", retrieval, state)
    if state == "vector_missing_or_invalid":
        return _result(request, scope, "insufficient_evidence", retrieval, state)
    if state == "empty_structural_candidates":
        return _result(request, scope, "valid_empty", retrieval, state)
    return _result(request, scope, "insufficient_evidence", primitive_state=state)


def _execute_request(
    db_path: str,
    request: EvidenceRequest,
    scope: StructuralScope,
    *,
    credit_targets: Iterable[Mapping[str, Any]] | None = None,
    exact_term_placements: bool = False,
) -> EvidenceExecutionResult:
    if request.provenance_required is not True:
        return _result(request, scope, "insufficient_evidence", primitive_state="provenance_required")
    if request.kind == "course_set":
        return _execute_course_set(
            db_path,
            request,
            scope,
            exact_term_placements=exact_term_placements,
        )
    if request.kind == "placement_facts":
        return _execute_placement(db_path, request, scope)
    if request.kind == "credit_facts":
        if credit_targets is None:
            return _execute_credit(db_path, request, scope)
        return _execute_credit(
            db_path,
            request,
            scope,
            course_targets=credit_targets,
        )
    if request.kind == "prerequisite_facts":
        return _execute_prerequisites(db_path, request, scope)
    if request.kind == "description_evidence":
        return _execute_descriptions(db_path, request, scope)
    return _result(request, scope, "insufficient_evidence", primitive_state="topic_matches_pending_4I3c")


def _execute_materialized_request(
    db_path: str,
    request: EvidenceRequest,
    scope: StructuralScope,
    *,
    credit_targets: Iterable[Mapping[str, Any]] | None = None,
    exact_term_placements: bool = False,
) -> EvidenceExecutionResult:
    """Execute one concrete scope without affecting sibling partitions."""
    try:
        kwargs: dict[str, Any] = {}
        if credit_targets is not None:
            kwargs["credit_targets"] = credit_targets
        if exact_term_placements:
            kwargs["exact_term_placements"] = True
        return _execute_request(db_path, request, scope, **kwargs)
    except (FileNotFoundError, OSError, sqlite3.Error, TypeError, ValueError, KeyError):
        return _result(
            request,
            scope,
            "insufficient_evidence",
            primitive_state="execution_failure",
        )


def _topic_credit_targets(
    payload: Any,
) -> tuple[Mapping[str, Any], ...] | None:
    candidates = getattr(payload, "scored_candidates", None)
    if not isinstance(candidates, (list, tuple)):
        return None
    targets: list[Mapping[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            return None
        program = candidate.get("program")
        course_code = candidate.get("course_code")
        course_id = candidate.get("course_id")
        if (
            not isinstance(program, str)
            or not program.strip()
            or not isinstance(course_code, str)
            or not course_code.strip()
            or isinstance(course_id, bool)
            or not isinstance(course_id, int)
        ):
            return None
        targets.append(candidate)
    return tuple(targets)


def _execute_topic_dependent_credit(
    db_path: str,
    request: EvidenceRequest,
    dependency: EvidenceExecutionResult,
) -> EvidenceExecutionResult:
    scope = dependency.effective_scope
    if dependency.status == "insufficient_evidence":
        return _result(
            request,
            scope,
            "insufficient_evidence",
            primitive_state=dependency.primitive_state or "dependency_insufficient",
        )
    if dependency.status == "valid_empty":
        return _execute_materialized_request(
            db_path,
            request,
            scope,
            credit_targets=(),
        )
    targets = _topic_credit_targets(dependency.payload)
    if targets is None:
        return _result(
            request,
            scope,
            "insufficient_evidence",
            primitive_state="malformed_topic_dependency",
        )
    return _execute_materialized_request(
        db_path,
        request,
        scope,
        credit_targets=targets,
    )


def _topic_prerequisite_targets(
    payload: Any,
) -> tuple[Mapping[str, Any], ...] | None:
    if not isinstance(payload, ConstrainedTopicRetrievalResult):
        return None
    candidates = payload.scored_candidates
    if not isinstance(candidates, (list, tuple)):
        return None
    targets: list[Mapping[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            return None
        program = candidate.get("program")
        course_code = candidate.get("course_code")
        course_id = candidate.get("course_id")
        if (
            not isinstance(program, str)
            or not program.strip()
            or not isinstance(course_code, str)
            or not course_code.strip()
            or isinstance(course_id, bool)
            or not isinstance(course_id, int)
        ):
            return None
        targets.append(
            {
                "program": program.strip(),
                "course_code": course_code.strip(),
                "course_id": course_id,
            }
        )
    return tuple(targets)


def _execute_topic_dependent_prerequisites(
    db_path: str,
    request: EvidenceRequest,
    dependency: EvidenceExecutionResult,
) -> EvidenceExecutionResult:
    scope = dependency.effective_scope
    if dependency.status == "insufficient_evidence":
        return _result(
            request,
            scope,
            "insufficient_evidence",
            primitive_state=dependency.primitive_state or "dependency_insufficient",
        )
    if dependency.status == "valid_empty":
        return _result(
            request,
            scope,
            "valid_empty",
            (),
            "empty_topic_candidates",
        )
    targets = _topic_prerequisite_targets(dependency.payload)
    if targets is None:
        return _result(
            request,
            scope,
            "insufficient_evidence",
            primitive_state="malformed_topic_dependency",
        )
    burden = build_direct_prerequisite_burden(db_path, targets)
    return _result(
        request,
        scope,
        burden.status,
        burden.burdens,
        burden.primitive_state,
    )


def _course_set_prerequisite_targets(
    payload: Any,
) -> tuple[Mapping[str, Any], ...] | None:
    if not isinstance(payload, Mapping):
        return None
    courses = payload.get("courses")
    if not isinstance(courses, (list, tuple)):
        return None

    targets: list[Mapping[str, Any]] = []
    for course in courses:
        if not isinstance(course, Mapping):
            return None
        if course.get("is_alternative"):
            members = course.get("alternative_courses")
            if not isinstance(members, (list, tuple)) or not members:
                return None
            for member in members:
                if not isinstance(member, Mapping):
                    return None
                program = member.get("program") or course.get("program")
                candidate = _canonical_prerequisite_candidate(member, program)
                if candidate is None:
                    return None
                targets.append(candidate)
            continue

        candidate = _canonical_prerequisite_candidate(course, course.get("program"))
        if candidate is None:
            return None
        targets.append(candidate)
    return tuple(targets)


def _canonical_prerequisite_candidate(
    value: Mapping[str, Any],
    fallback_program: Any,
) -> Mapping[str, Any] | None:
    program = value.get("program") or fallback_program
    course_id = value.get("course_id")
    course_code = value.get("course_code")
    if (
        not isinstance(program, str)
        or not program.strip()
        or isinstance(course_id, bool)
        or not isinstance(course_id, int)
        or not isinstance(course_code, str)
        or not course_code.strip()
    ):
        return None
    return {
        "program": program.strip(),
        "course_id": course_id,
        "course_code": course_code.strip(),
    }


def _execute_course_set_dependent_prerequisites(
    db_path: str,
    request: EvidenceRequest,
    dependency: EvidenceExecutionResult,
) -> EvidenceExecutionResult:
    scope = dependency.effective_scope
    if dependency.status == "insufficient_evidence":
        return _result(
            request,
            scope,
            "insufficient_evidence",
            primitive_state=dependency.primitive_state or "dependency_insufficient",
        )
    if dependency.status == "valid_empty":
        return _result(
            request,
            scope,
            "valid_empty",
            (),
            "empty_course_set",
        )
    targets = _course_set_prerequisite_targets(dependency.payload)
    if targets is None:
        return _result(
            request,
            scope,
            "insufficient_evidence",
            primitive_state="malformed_course_set_dependency",
        )
    burden = build_direct_prerequisite_burden(db_path, targets)
    return _result(
        request,
        scope,
        burden.status,
        burden.burdens,
        burden.primitive_state,
    )


def execute_evidence_plan(
    db_path: str,
    plan: EvidencePlan,
) -> EvidenceBundle:
    """Execute only relational/direct requests in deterministic dependency order."""
    order = _graph_order(plan)
    if order is None:
        return EvidenceBundle(
            plan,
            tuple(
                _result(
                    request,
                    request.scope,
                    "insufficient_evidence",
                    primitive_state="invalid_dependency_graph",
                )
                for request in plan.requests
            ),
        )

    results: list[EvidenceExecutionResult] = []
    by_request: dict[str, list[EvidenceExecutionResult]] = {}
    request_by_id = {request.request_id: request for request in plan.requests}
    for request in order:
        dependencies = [
            result
            for dependency in request.depends_on
            for result in by_request.get(dependency, ())
        ]
        if request.kind == "topic_matches":
            dependency_ids = tuple(request.depends_on)
            course_set_ids = tuple(
                dependency_id
                for dependency_id in dependency_ids
                if request_by_id.get(dependency_id, None) is not None
                and request_by_id[dependency_id].kind == "course_set"
            )
            if len(dependency_ids) != 1 or len(course_set_ids) != 1:
                request_results = [
                    _result(
                        request,
                        request.scope,
                        "insufficient_evidence",
                        primitive_state="invalid_course_set_dependency",
                    )
                ]
            else:
                dependency_results = by_request.get(course_set_ids[0], ())
                if not dependency_results:
                    request_results = [
                        _result(
                            request,
                            request.scope,
                            "insufficient_evidence",
                            primitive_state="missing_course_set_dependency",
                        )
                    ]
                else:
                    request_results = []
                    for dependency in dependency_results:
                        if dependency.planned_request.scope != request.scope:
                            request_results.append(
                                _result(
                                    request,
                                    dependency.effective_scope,
                                    "insufficient_evidence",
                                    primitive_state="incompatible_course_set_scope",
                                )
                            )
                        elif dependency.status != "complete":
                            request_results.append(
                                _topic_dependency_result(request, dependency)
                            )
                        else:
                            request_results.append(
                                _execute_topic_matches(db_path, request, dependency)
                            )
        elif request.kind == "credit_facts" and any(
            request_by_id.get(dependency_id, None) is not None
            and request_by_id[dependency_id].kind == "topic_matches"
            for dependency_id in request.depends_on
        ):
            topic_dependency_ids = tuple(
                dependency_id
                for dependency_id in request.depends_on
                if request_by_id.get(dependency_id, None) is not None
                and request_by_id[dependency_id].kind == "topic_matches"
            )
            if len(request.depends_on) != 1 or len(topic_dependency_ids) != 1:
                request_results = [
                    _result(
                        request,
                        request.scope,
                        "insufficient_evidence",
                        primitive_state="invalid_topic_dependency",
                    )
                ]
            else:
                dependency_results = by_request.get(topic_dependency_ids[0], ())
                if not dependency_results:
                    request_results = [
                        _result(
                            request,
                            request.scope,
                            "insufficient_evidence",
                            primitive_state="missing_topic_dependency",
                        )
                    ]
                else:
                    request_results = []
                    for dependency in dependency_results:
                        if dependency.planned_request.scope != request.scope:
                            request_results.append(
                                _result(
                                    request,
                                    dependency.effective_scope,
                                    "insufficient_evidence",
                                    primitive_state="incompatible_topic_scope",
                                )
                            )
                        else:
                            request_results.append(
                                _execute_topic_dependent_credit(
                                    db_path,
                                    request,
                                    dependency,
                                )
                            )
        elif request.kind == "prerequisite_facts" and any(
            request_by_id.get(dependency_id, None) is not None
            and request_by_id[dependency_id].kind == "course_set"
            for dependency_id in request.depends_on
        ):
            course_set_dependency_ids = tuple(
                dependency_id
                for dependency_id in request.depends_on
                if request_by_id.get(dependency_id, None) is not None
                and request_by_id[dependency_id].kind == "course_set"
            )
            if len(request.depends_on) != 1 or len(course_set_dependency_ids) != 1:
                request_results = [
                    _result(
                        request,
                        request.scope,
                        "insufficient_evidence",
                        primitive_state="invalid_course_set_dependency",
                    )
                ]
            else:
                dependency_results = by_request.get(course_set_dependency_ids[0], ())
                if not dependency_results:
                    request_results = [
                        _result(
                            request,
                            request.scope,
                            "insufficient_evidence",
                            primitive_state="missing_course_set_dependency",
                        )
                    ]
                else:
                    request_results = []
                    for dependency in dependency_results:
                        if dependency.planned_request.scope != request.scope:
                            request_results.append(
                                _result(
                                    request,
                                    dependency.effective_scope,
                                    "insufficient_evidence",
                                    primitive_state="incompatible_course_set_scope",
                                )
                            )
                        else:
                            request_results.append(
                                _execute_course_set_dependent_prerequisites(
                                    db_path,
                                    request,
                                    dependency,
                                )
                            )
        elif request.kind == "prerequisite_facts" and any(
            request_by_id.get(dependency_id, None) is not None
            and request_by_id[dependency_id].kind == "topic_matches"
            for dependency_id in request.depends_on
        ):
            topic_dependency_ids = tuple(
                dependency_id
                for dependency_id in request.depends_on
                if request_by_id.get(dependency_id, None) is not None
                and request_by_id[dependency_id].kind == "topic_matches"
            )
            if len(request.depends_on) != 1 or len(topic_dependency_ids) != 1:
                request_results = [
                    _result(
                        request,
                        request.scope,
                        "insufficient_evidence",
                        primitive_state="invalid_topic_dependency",
                    )
                ]
            else:
                dependency_results = by_request.get(topic_dependency_ids[0], ())
                if not dependency_results:
                    request_results = [
                        _result(
                            request,
                            request.scope,
                            "insufficient_evidence",
                            primitive_state="missing_topic_dependency",
                        )
                    ]
                else:
                    request_results = []
                    for dependency in dependency_results:
                        if dependency.planned_request.scope != request.scope:
                            request_results.append(
                                _result(
                                    request,
                                    dependency.effective_scope,
                                    "insufficient_evidence",
                                    primitive_state="incompatible_topic_scope",
                                )
                            )
                        else:
                            request_results.append(
                                _execute_topic_dependent_prerequisites(
                                    db_path,
                                    request,
                                    dependency,
                                )
                            )
        elif any(result.status == "insufficient_evidence" for result in dependencies):
            request_results = [
                _result(
                    request,
                    request.scope,
                    "insufficient_evidence",
                    primitive_state="dependency_insufficient",
                )
            ]
        else:
            try:
                scopes = _materialize_scopes(db_path, request)
            except (FileNotFoundError, OSError, sqlite3.Error, TypeError, ValueError, KeyError):
                request_results = [
                    _result(
                        request,
                        request.scope,
                        "insufficient_evidence",
                        primitive_state="materialization_failure",
                    )
                ]
            else:
                credit_targets = (
                    request.course_targets
                    if request.kind == "credit_facts" and request.course_targets
                    else None
                )
                exact_term_placements = (
                    request.kind == "course_set"
                    and request.scope.category is None
                    and bool(request.scope.years)
                    and bool(request.scope.semesters)
                    and not request.scope.course_targets
                    and not any(
                        dependent.kind == "topic_matches"
                        and request.request_id in dependent.depends_on
                        for dependent in plan.requests
                    )
                )
                request_results = [
                    _execute_materialized_request(
                        db_path,
                        request,
                        scope,
                        credit_targets=credit_targets,
                        exact_term_placements=exact_term_placements,
                    )
                    for scope in scopes
                ]
        results.extend(request_results)
        by_request[request.request_id] = request_results

    return EvidenceBundle(plan, tuple(results))


def _similarity_partition(scope: StructuralScope) -> dict[str, Any] | None:
    """Project one concrete scope to its stable similarity identity."""
    if (
        not isinstance(scope.program, str)
        or not scope.program.strip()
        or len(scope.plans) != 1
        or not isinstance(scope.plans[0], str)
        or not scope.plans[0].strip()
    ):
        return None
    return {
        "program": scope.program,
        "plan": scope.plans[0],
        "plans": tuple(scope.plans),
    }


def _similarity_target(request: EvidenceRequest) -> Mapping[str, Any] | None:
    if request.kind != "description_evidence" or len(request.course_targets) != 1:
        return None
    target = request.course_targets[0]
    if not isinstance(target, Mapping):
        return None
    if (
        not isinstance(target.get("program"), str)
        or not target["program"].strip()
        or not isinstance(target.get("course_code"), str)
        or not target["course_code"].strip()
        or isinstance(target.get("course_id"), bool)
        or not isinstance(target.get("course_id"), int)
    ):
        return None
    return target


def _similarity_scope_key(partition: Mapping[str, Any]) -> tuple[str, str]:
    return (str(partition.get("plan")), repr(tuple(sorted(partition.items(), key=lambda item: item[0]))))


def _valid_description_record(
    evidence: Mapping[str, Any],
    target: Mapping[str, Any],
    partition: Mapping[str, Any],
) -> bool:
    if (
        not isinstance(evidence.get("chunk_id"), str)
        or not evidence["chunk_id"].strip()
        or evidence.get("chunk_type") != "description"
        or evidence.get("program") != target.get("program")
        or evidence.get("course_code") != target.get("course_code")
    ):
        return False
    if "course_id" in evidence and evidence.get("course_id") != target.get("course_id"):
        return False
    text = evidence.get("text", evidence.get("description"))
    if not isinstance(text, str) or not text.strip():
        return False
    provenance = evidence.get("provenance")
    if (
        not isinstance(provenance, (list, tuple))
        or not provenance
        or any(not isinstance(reference, Mapping) for reference in provenance)
    ):
        return False
    supplied_partition = evidence.get("partition")
    if supplied_partition is None:
        return True
    if not isinstance(supplied_partition, Mapping):
        return False
    if "program" in supplied_partition and supplied_partition["program"] != partition["program"]:
        return False
    if "plan" in supplied_partition and supplied_partition["plan"] != partition["plan"]:
        return False
    if "plans" in supplied_partition:
        plans = supplied_partition["plans"]
        if not isinstance(plans, (list, tuple)) or tuple(plans) != partition["plans"]:
            return False
    return True


def _similarity_course_records(
    results: Iterable[EvidenceExecutionResult],
    target: Mapping[str, Any],
) -> tuple[tuple[dict[str, Any], ...], bool] | None:
    records: list[dict[str, Any]] = []
    seen_scopes: set[tuple[str, str]] = set()
    has_failed_result = False
    for result in results:
        if not isinstance(result, EvidenceExecutionResult):
            return None
        partition = _similarity_partition(result.effective_scope)
        if partition is None:
            return None
        scope_key = _similarity_scope_key(partition)
        if scope_key in seen_scopes:
            return None
        seen_scopes.add(scope_key)

        payload = result.payload
        if payload is None:
            if result.status == "complete":
                return None
            descriptions: tuple[Mapping[str, Any], ...] = ()
            has_failed_result = True
        elif isinstance(payload, (list, tuple)):
            descriptions = tuple(payload)
            if result.status != "complete":
                has_failed_result = True
            if result.status == "complete" and not descriptions:
                return None
            if any(not isinstance(evidence, Mapping) for evidence in descriptions):
                return None
        else:
            return None

        physical_course_ids = {
            evidence.get("course_id")
            for evidence in descriptions
            if "course_id" in evidence
        }
        if any(
            isinstance(course_id, bool) or not isinstance(course_id, int)
            for course_id in physical_course_ids
        ) or len(physical_course_ids) > 1:
            return None
        validation_target = dict(target)
        if physical_course_ids:
            validation_target["course_id"] = next(iter(physical_course_ids))
        if any(
            not _valid_description_record(evidence, validation_target, partition)
            for evidence in descriptions
        ):
            return None

        records.append(
            {
                "program": target["program"],
                "course_code": target["course_code"],
                "course_id": target["course_id"],
                "partition": partition,
                "description_evidence": descriptions,
            }
        )
    if not records:
        return None
    return tuple(records), has_failed_result


def execute_exact_similarity_from_bundle(
    db_path: str | Path,
    bundle: EvidenceBundle,
    left_request_id: str,
    right_request_id: str,
    *,
    selected_plan: str | None = None,
) -> SimilarityEvidence:
    """Bridge executed description evidence to the exact similarity API.

    Similarity is intentionally not an EvidencePlan primitive.  This helper
    consumes only the two already-executed description request streams and
    delegates persisted validation/vector access to the existing exact-course
    similarity implementation exactly once.
    """
    def insufficient() -> SimilarityEvidence:
        return SimilarityEvidence(status="insufficient_evidence")
    if (
        not isinstance(db_path, (str, Path))
        or not str(db_path)
        or not isinstance(bundle, EvidenceBundle)
        or not isinstance(left_request_id, str)
        or not left_request_id.strip()
        or not isinstance(right_request_id, str)
        or not right_request_id.strip()
        or left_request_id == right_request_id
    ):
        return insufficient()

    requests = {
        request.request_id: request
        for request in bundle.plan.requests
        if isinstance(request, EvidenceRequest)
    }
    left_request = requests.get(left_request_id)
    right_request = requests.get(right_request_id)
    left_target = _similarity_target(left_request) if left_request is not None else None
    right_target = _similarity_target(right_request) if right_request is not None else None
    if left_target is None or right_target is None:
        return insufficient()

    left_results = tuple(
        result
        for result in bundle.results
        if result.request_id == left_request_id
        and result.planned_request is left_request
        and result.kind == "description_evidence"
    )
    right_results = tuple(
        result
        for result in bundle.results
        if result.request_id == right_request_id
        and result.planned_request is right_request
        and result.kind == "description_evidence"
    )
    left_records = _similarity_course_records(left_results, left_target)
    right_records = _similarity_course_records(right_results, right_target)
    if left_records is None or right_records is None:
        return insufficient()

    try:
        result = aggregate_exact_course_similarity(
            db_path,
            left_records[0],
            right_records[0],
            selected_plan=selected_plan,
        )
    except (FileNotFoundError, OSError, sqlite3.Error, TypeError, ValueError, KeyError):
        return insufficient()
    if left_records[1] or right_records[1]:
        if result.status == "valid_empty":
            return SimilarityEvidence(
                status="insufficient_evidence",
                unmatched_partitions=result.unmatched_partitions,
            )
        if result.status == "complete":
            return SimilarityEvidence(
                status="insufficient_evidence",
                pairs=result.pairs,
                unmatched_partitions=result.unmatched_partitions,
            )
    return result


__all__ = [
    "DIRECT_PRIMITIVES",
    "EXECUTION_STATES",
    "DirectPrerequisiteBurden",
    "DirectPrerequisiteBurdenResult",
    "DirectPrerequisiteRequirement",
    "EvidenceBundle",
    "EvidenceExecutionResult",
    "build_direct_prerequisite_burden",
    "execute_exact_similarity_from_bundle",
    "execute_evidence_plan",
]
