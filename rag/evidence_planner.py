"""Immutable Phase 3C evidence-plan representations.

This module deliberately contains data-model types only.  It does not parse
questions, resolve database identities, execute relational queries, retrieve
vectors, or derive answers.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from rag.query_spec import QuerySpec
from rag.resolution import ResolutionOutcome


APPLICABLE = "APPLICABLE"
UNCONSTRAINED = "UNCONSTRAINED"
EVIDENCE_PRIMITIVES = (
    "course_set",
    "placement_facts",
    "credit_facts",
    "prerequisite_facts",
    "description_evidence",
    "topic_matches",
)
GROUP_BY_VALUES = ("plan", "year", "semester", "course")
SCOPE_EXPANSION_AXES = GROUP_BY_VALUES


def _ordered_unique(values: Any) -> tuple[Any, ...]:
    result: list[Any] = []
    for value in values:
        if value not in result:
            result.append(value)
    return tuple(result)


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


def _validate_axes(values: tuple[str, ...], field_name: str) -> None:
    invalid = [value for value in values if value not in SCOPE_EXPANSION_AXES]
    if invalid:
        raise ValueError(f"unsupported {field_name}: {invalid!r}")


@dataclass(frozen=True, slots=True)
class StructuralScope:
    """Symbolic structural scope without database-resolved identifiers.

    Explicit axis values are stored in ``plans``, ``years``, and
    ``semesters``.  An axis listed in ``expand_applicable`` means that a later
    executor must enumerate applicable values.  An axis listed in
    ``unconstrained`` is intentionally unrestricted.  These states are
    separate so an empty tuple never silently selects a value.
    """

    program: str | None = None
    plans: tuple[str, ...] = ()
    years: tuple[int, ...] = ()
    semesters: tuple[int, ...] = ()
    category: str | None = None
    expand_applicable: tuple[str, ...] = ()
    unconstrained: tuple[str, ...] = ()
    course_targets: tuple[Mapping[str, Any], ...] = ()
    group_by: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        plans = _ordered_unique(self.plans)
        years = _ordered_unique(self.years)
        semesters = _ordered_unique(self.semesters)
        expand_applicable = _ordered_unique(self.expand_applicable)
        unconstrained = _ordered_unique(self.unconstrained)
        course_targets = tuple(_freeze(target) for target in self.course_targets)
        group_by = _ordered_unique(self.group_by)
        _validate_axes(expand_applicable, "expand_applicable axes")
        _validate_axes(unconstrained, "unconstrained axes")
        _validate_axes(group_by, "group_by")
        if any(not isinstance(target, Mapping) for target in course_targets):
            raise ValueError("course_targets must contain mappings")
        overlap = set(expand_applicable) & set(unconstrained)
        if overlap:
            raise ValueError(f"axes cannot be both expanded and unconstrained: {sorted(overlap)!r}")
        explicit_axes = {
            axis
            for axis, values in (
                ("plan", plans),
                ("year", years),
                ("semester", semesters),
            )
            if values
        }
        conflicting = explicit_axes & (set(expand_applicable) | set(unconstrained))
        if conflicting:
            raise ValueError(
                f"explicit values conflict with axis state: {sorted(conflicting)!r}"
            )
        object.__setattr__(self, "plans", plans)
        object.__setattr__(self, "years", years)
        object.__setattr__(self, "semesters", semesters)
        object.__setattr__(self, "expand_applicable", expand_applicable)
        object.__setattr__(self, "unconstrained", unconstrained)
        object.__setattr__(self, "course_targets", course_targets)
        object.__setattr__(self, "group_by", group_by)


@dataclass(frozen=True, slots=True)
class EvidenceRequest:
    """One explicit request for one of the six evidence primitives."""

    request_id: str
    kind: str
    scope: StructuralScope
    depends_on: tuple[str, ...] = ()
    course_targets: tuple[Mapping[str, Any], ...] = ()
    topic: str | None = None
    provenance_required: bool = True

    def __post_init__(self) -> None:
        if self.kind not in EVIDENCE_PRIMITIVES:
            raise ValueError(f"unsupported evidence primitive: {self.kind!r}")
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("request_id must be a non-empty string")
        if not isinstance(self.provenance_required, bool):
            raise ValueError("provenance_required must be a boolean")
        dependencies = _ordered_unique(self.depends_on)
        if any(not isinstance(value, str) or not value for value in dependencies):
            raise ValueError("depends_on must contain non-empty request IDs")
        targets = tuple(_freeze(target) for target in self.course_targets)
        if any(not isinstance(target, Mapping) for target in targets):
            raise ValueError("course_targets must contain mappings")
        object.__setattr__(self, "depends_on", dependencies)
        object.__setattr__(self, "course_targets", targets)


@dataclass(frozen=True, slots=True)
class EvidencePlan:
    """Immutable evidence dependency graph for one normalized QA request."""

    scope: StructuralScope
    requests: tuple[EvidenceRequest, ...] = ()
    group_by: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        requests = tuple(self.requests)
        group_by = _ordered_unique(self.group_by)
        _validate_axes(group_by, "group_by")
        request_ids = [request.request_id for request in requests]
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("evidence request IDs must be unique")
        known_ids = set(request_ids)
        for request in requests:
            missing = set(request.depends_on) - known_ids
            if missing:
                raise ValueError(
                    f"request {request.request_id!r} depends on unknown IDs: "
                    f"{sorted(missing)!r}"
                )
            if request.request_id in request.depends_on:
                raise ValueError(f"request {request.request_id!r} cannot depend on itself")
        object.__setattr__(self, "requests", requests)
        object.__setattr__(self, "group_by", group_by)


def _resolved_course_targets(
    resolution: ResolutionOutcome,
) -> tuple[Mapping[str, Any], ...]:
    targets: list[Mapping[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for reference in resolution.course_references:
        for candidate in reference.candidates:
            identity = (
                candidate.get("program"),
                candidate.get("course_code"),
            )
            if identity in seen:
                continue
            seen.add(identity)
            targets.append(candidate)
    return tuple(targets)


def build_structural_scope(
    query_spec: QuerySpec,
    resolution: ResolutionOutcome,
) -> StructuralScope:
    """Construct symbolic structural scope without querying or expanding data.

    The resolution outcome owns the program scope.  Empty plans always remain
    an ``APPLICABLE`` expansion because plan partitions are preserved for
    every answerable program-scoped request.  Empty year/semester axes expand
    only when explicitly present in ``group_by``; otherwise they are marked
    unconstrained.  The returned scope contains no materialized cross-product.
    """

    if resolution.action != "answer":
        raise ValueError("blocked resolution cannot produce a structural scope")

    group_by = tuple(query_spec.group_by)
    expand_applicable: list[str] = []
    unconstrained: list[str] = []

    if query_spec.plans:
        plans = tuple(query_spec.plans)
    else:
        plans = ()
        expand_applicable.append("plan")

    if query_spec.years:
        years = tuple(query_spec.years)
    elif "year" in group_by:
        years = ()
        expand_applicable.append("year")
    else:
        years = ()
        unconstrained.append("year")

    if query_spec.semesters:
        semesters = tuple(query_spec.semesters)
    elif "semester" in group_by:
        semesters = ()
        expand_applicable.append("semester")
    else:
        semesters = ()
        unconstrained.append("semester")

    course_targets = _resolved_course_targets(resolution)
    if not course_targets:
        unconstrained.append("course")

    return StructuralScope(
        program=resolution.resolved_program,
        plans=plans,
        years=years,
        semesters=semesters,
        category=query_spec.category,
        expand_applicable=tuple(expand_applicable),
        unconstrained=tuple(unconstrained),
        course_targets=course_targets,
        group_by=group_by,
    )


def _plan_partition_scopes(scope: StructuralScope) -> tuple[StructuralScope, ...]:
    """Return explicit plan partitions when a plan comparison requires them."""
    if not scope.plans or "plan" not in scope.group_by:
        return (scope,)
    return tuple(
        StructuralScope(
            program=scope.program,
            plans=(plan,),
            years=scope.years,
            semesters=scope.semesters,
            category=scope.category,
            expand_applicable=scope.expand_applicable,
            unconstrained=scope.unconstrained,
            course_targets=scope.course_targets,
            group_by=scope.group_by,
        )
        for plan in scope.plans
    )


def _request(
    request_id: str,
    kind: str,
    scope: StructuralScope,
    *,
    depends_on: tuple[str, ...] = (),
    course_targets: tuple[Mapping[str, Any], ...] = (),
    topic: str | None = None,
) -> EvidenceRequest:
    return EvidenceRequest(
        request_id=request_id,
        kind=kind,
        scope=scope,
        depends_on=depends_on,
        course_targets=course_targets,
        topic=topic,
        provenance_required=True,
    )


def plan_evidence(
    query_spec: QuerySpec,
    resolution: ResolutionOutcome,
) -> EvidencePlan:
    """Map a QuerySpec and answerable resolution to evidence requests.

    This function only constructs an immutable dependency graph.  It does not
    parse natural language, access the database, execute vector search, or
    perform any aggregation or comparison.
    """
    scope = build_structural_scope(query_spec, resolution)
    requests: list[EvidenceRequest] = []
    by_kind: dict[str, list[str]] = {}

    def add(request: EvidenceRequest) -> None:
        requests.append(request)
        by_kind.setdefault(request.kind, []).append(request.request_id)

    exact_targets = scope.course_targets
    exact_operations = {"describe", "similarity"}
    if query_spec.operations and exact_targets:
        if "describe" in query_spec.operations:
            for index, target in enumerate(exact_targets, start=1):
                add(
                    _request(
                        f"description_{index}",
                        "description_evidence",
                        scope,
                        course_targets=(target,),
                    )
                )
        if "similarity" in query_spec.operations:
            for index, target in enumerate(exact_targets, start=1):
                add(
                    _request(
                        f"similarity_description_{index}",
                        "description_evidence",
                        scope,
                        course_targets=(target,),
                    )
                )

    topic_target_id: str | None = None
    collection_operations = {"list", "count", "sum_credits", "existence"}
    needs_collection = bool(collection_operations & set(query_spec.operations))
    if query_spec.topic is not None and needs_collection:
        add(_request("course_set", "course_set", scope))
        add(
            _request(
                "topic_matches",
                "topic_matches",
                scope,
                depends_on=("course_set",),
                topic=query_spec.topic,
            )
        )
        topic_target_id = "topic_matches"
    elif needs_collection:
        add(_request("course_set", "course_set", scope))

    target_relation_id = topic_target_id or (
        "course_set" if "course_set" in by_kind else None
    )

    if "sum_credits" in query_spec.operations:
        add(
            _request(
                "credit_facts",
                "credit_facts",
                scope,
                depends_on=(target_relation_id,) if target_relation_id else (),
                course_targets=exact_targets,
            )
        )

    if "placement" in query_spec.operations or "earliest" in query_spec.operations:
        for index, partition_scope in enumerate(
            _plan_partition_scopes(scope), start=1
        ):
            add(
                _request(
                    f"placement_facts_{index}",
                    "placement_facts",
                    partition_scope,
                    course_targets=exact_targets,
                )
            )

    if "prerequisite" in query_spec.operations and exact_targets:
        for index, target in enumerate(exact_targets, start=1):
            add(
                _request(
                    f"prerequisite_facts_{index}",
                    "prerequisite_facts",
                    scope,
                    course_targets=(target,),
                )
            )

    # A bare plan comparison needs a relational course set per explicit plan.
    # Placement facts are included so later composition can report available
    # placement differences without inventing a new evidence primitive.
    if (
        "compare" in query_spec.operations
        and query_spec.group_by == ("plan",)
        and not requests
    ):
        for index, partition_scope in enumerate(
            _plan_partition_scopes(scope), start=1
        ):
            add(
                _request(
                    f"course_set_plan_{index}",
                    "course_set",
                    partition_scope,
                )
            )
            add(
                _request(
                    f"placement_facts_plan_{index}",
                    "placement_facts",
                    partition_scope,
                    depends_on=(f"course_set_plan_{index}",),
                )
            )

    return EvidencePlan(scope=scope, requests=tuple(requests), group_by=scope.group_by)


__all__ = [
    "EVIDENCE_PRIMITIVES",
    "GROUP_BY_VALUES",
    "APPLICABLE",
    "UNCONSTRAINED",
    "build_structural_scope",
    "plan_evidence",
    "EvidencePlan",
    "EvidenceRequest",
    "StructuralScope",
]
