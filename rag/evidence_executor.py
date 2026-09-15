"""Deterministic execution of relational/direct EvidencePlan requests."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
import sqlite3
from types import MappingProxyType
from typing import Any

from rag.evidence_planner import EvidencePlan, EvidenceRequest, StructuralScope
from rag.retrieval.retrieve import (
    SimilarityEvidence,
    aggregate_exact_course_similarity,
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

    request_targets = request.course_targets or source.course_targets
    materialization_source = replace(source, course_targets=request_targets)
    plan_keys = source.plans or applicable_plan_keys(db_path, source.program)
    if not plan_keys:
        return (materialization_source,)

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
    *,
    course_targets: Iterable[Mapping[str, Any]] | None = None,
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
    credit_args = (
        db_path,
        scope.program or "",
        scope.plans[0],
        scope.years[0],
        scope.semesters[0],
    )
    if course_targets is None:
        result = get_semester_credits(*credit_args)
    else:
        result = get_semester_credits(*credit_args, course_targets=course_targets)
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
    for course_id in course_ids:
        records.extend(prerequisites_of_course(db_path, course_id))
    payload = tuple(records)
    if not payload:
        return _result(request, scope, "valid_empty", payload, "empty_relation")
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
) -> EvidenceExecutionResult:
    if request.provenance_required is not True:
        return _result(request, scope, "insufficient_evidence", primitive_state="provenance_required")
    if request.kind == "course_set":
        return _execute_course_set(db_path, request, scope)
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
) -> EvidenceExecutionResult:
    """Execute one concrete scope without affecting sibling partitions."""
    try:
        if credit_targets is None:
            return _execute_request(db_path, request, scope)
        return _execute_request(
            db_path,
            request,
            scope,
            credit_targets=credit_targets,
        )
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
    "EvidenceBundle",
    "EvidenceExecutionResult",
    "execute_exact_similarity_from_bundle",
    "execute_evidence_plan",
]
