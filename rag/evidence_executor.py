"""Deterministic execution of relational/direct EvidencePlan requests."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import sqlite3
from types import MappingProxyType
from typing import Any

from rag.evidence_planner import EvidencePlan, EvidenceRequest, StructuralScope
from rag.retrieval.retrieve import (
    fetch_course_description_evidence,
    retrieve_constrained_topic_evidence,
)
from rag.structured.queries import (
    applicable_plan_keys,
    get_semester_credits,
    prerequisites_of_course,
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


def _materialize_scopes(
    db_path: str,
    request: EvidenceRequest,
) -> tuple[StructuralScope, ...]:
    source = request.scope
    if not isinstance(source.program, str) or not source.program.strip():
        raise ValueError("answerable execution needs a resolved program")

    plan_keys = source.plans or applicable_plan_keys(db_path, source.program)
    if not plan_keys:
        return (source,)

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
            values = _axis_values(db_path, source, plan_key, axis="year")
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
                    source,
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
                targets = source.course_targets
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
) -> EvidenceExecutionResult:
    result = scoped_course_set(
        db_path,
        scope.program or "",
        scope.plans,
        years=scope.years,
        semesters=scope.semesters,
        category=scope.category,
        course_targets=scope.course_targets,
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
    )
    if result.get("status") == "no_data":
        return _result(request, scope, "valid_empty", result, "empty_relation")
    return _result(request, scope, _status_for_payload(result), result)


def _execute_credit(
    db_path: str,
    request: EvidenceRequest,
    scope: StructuralScope,
) -> EvidenceExecutionResult:
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
    result = get_semester_credits(
        db_path,
        scope.program or "",
        scope.plans[0],
        scope.years[0],
        scope.semesters[0],
    )
    if result.get("status") == "no_data":
        return _result(request, scope, "valid_empty", result, "empty_relation")
    return _result(request, scope, _status_for_payload(result), result)


def _execute_prerequisites(
    db_path: str,
    request: EvidenceRequest,
    scope: StructuralScope,
) -> EvidenceExecutionResult:
    records: list[Mapping[str, Any]] = []
    for target in request.course_targets or scope.course_targets:
        course_id = target.get("course_id") if isinstance(target, Mapping) else None
        if isinstance(course_id, bool) or not isinstance(course_id, int):
            return _result(request, scope, "insufficient_evidence", primitive_state="invalid_course_target")
        records.extend(prerequisites_of_course(db_path, course_id))
    payload = tuple(records)
    if not payload:
        return _result(request, scope, "valid_empty", payload, "empty_relation")
    return _result(request, scope, _status_for_payload(payload), payload)


def _execute_descriptions(
    db_path: str,
    request: EvidenceRequest,
    scope: StructuralScope,
) -> EvidenceExecutionResult:
    if not request.course_targets:
        return _result(request, scope, "insufficient_evidence", primitive_state="missing_course_target")
    evidence: list[dict[str, Any]] = []
    for target in request.course_targets:
        course_id = target.get("course_id") if isinstance(target, Mapping) else None
        if isinstance(course_id, bool) or not isinstance(course_id, int):
            return _result(request, scope, "insufficient_evidence", primitive_state="invalid_course_target")
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
) -> EvidenceExecutionResult:
    if request.provenance_required is not True:
        return _result(request, scope, "insufficient_evidence", primitive_state="provenance_required")
    if request.kind == "course_set":
        return _execute_course_set(db_path, request, scope)
    if request.kind == "placement_facts":
        return _execute_placement(db_path, request, scope)
    if request.kind == "credit_facts":
        return _execute_credit(db_path, request, scope)
    if request.kind == "prerequisite_facts":
        return _execute_prerequisites(db_path, request, scope)
    if request.kind == "description_evidence":
        return _execute_descriptions(db_path, request, scope)
    return _result(request, scope, "insufficient_evidence", primitive_state="topic_matches_pending_4I3c")


def _execute_materialized_request(
    db_path: str,
    request: EvidenceRequest,
    scope: StructuralScope,
) -> EvidenceExecutionResult:
    """Execute one concrete scope without affecting sibling partitions."""
    try:
        return _execute_request(db_path, request, scope)
    except (FileNotFoundError, OSError, sqlite3.Error, TypeError, ValueError, KeyError):
        return _result(
            request,
            scope,
            "insufficient_evidence",
            primitive_state="execution_failure",
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
                request_results = [
                    _execute_materialized_request(db_path, request, scope)
                    for scope in scopes
                ]
        results.extend(request_results)
        by_request[request.request_id] = request_results

    return EvidenceBundle(plan, tuple(results))


__all__ = [
    "DIRECT_PRIMITIVES",
    "EXECUTION_STATES",
    "EvidenceBundle",
    "EvidenceExecutionResult",
    "execute_evidence_plan",
]
