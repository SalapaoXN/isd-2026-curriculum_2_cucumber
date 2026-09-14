"""Top-level curriculum QA over one relational and vector database."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from rag.aggregation import (
    ComparisonAggregation,
    ComponentAggregation,
    CourseSetAggregation,
    aggregate_course_set,
    aggregate_earliest,
    aggregate_required_load,
    aggregate_sum_credits,
)
from rag.evidence_executor import EvidenceBundle, EvidenceExecutionResult
from rag.grounded_answer import GroundedClaim
from rag.judgement import (
    JudgementEvidence,
    evaluate_preference,
    evaluate_quantity,
    evaluate_workload,
)
from rag.query_spec import parse_query_spec
from rag.retrieval.retrieve import ConstrainedTopicRetrievalResult, retrieve
from rag.resolution import QueryContext, ResolutionOutcome, resolve_query_spec
from rag.router import route_question
from rag.structured.qa import ask_structured


SCHEMA_PATH = Path(__file__).with_name("structured") / "schema.sql"


def _execution_results(
    bundle: EvidenceBundle,
    kind: str,
) -> tuple[EvidenceExecutionResult, ...]:
    """Select typed executor results without changing executor ordering."""
    if not isinstance(bundle, EvidenceBundle) or not isinstance(kind, str):
        return ()
    return tuple(result for result in bundle.results if result.kind == kind)


def _effective_scope_partition(scope: Any) -> dict[str, Any] | None:
    """Represent one effective scope as the partition metadata expected downstream."""
    if scope is None or not all(
        hasattr(scope, field)
        for field in (
            "program",
            "plans",
            "years",
            "semesters",
            "category",
            "group_by",
        )
    ):
        return None
    return {
        "program": scope.program,
        "plans": tuple(scope.plans),
        "years": tuple(scope.years),
        "semesters": tuple(scope.semesters),
        "category": scope.category,
        "group_by": tuple(scope.group_by),
    }


def _attach_effective_scope(
    records: Iterable[Mapping[str, Any]],
    scope: Any,
) -> tuple[Mapping[str, Any], ...] | None:
    """Copy payload records and attach, without overwriting, their effective partition."""
    partition = _effective_scope_partition(scope)
    if partition is None:
        return None
    attached: list[Mapping[str, Any]] = []
    try:
        for record in records:
            if not isinstance(record, Mapping):
                return None
            copied = dict(record)
            if "partition" not in copied:
                copied["partition"] = partition
            elif not isinstance(copied["partition"], Mapping):
                return None
            attached.append(copied)
    except (TypeError, ValueError):
        return None
    return tuple(attached)


def _payload_records(
    result: EvidenceExecutionResult,
    field: str | None = None,
) -> tuple[Mapping[str, Any], ...] | None:
    """Extract mapping records from one result and attach its concrete scope."""
    payload = result.payload
    if field is not None:
        if not isinstance(payload, Mapping):
            return None
        payload = payload.get(field, ())
    if isinstance(payload, Mapping) or isinstance(payload, (str, bytes)):
        return None
    try:
        records = tuple(payload)
    except TypeError:
        return None
    return _attach_effective_scope(records, result.effective_scope)


def _course_set_aggregate(
    result: EvidenceExecutionResult,
) -> CourseSetAggregation | None:
    """Adapt one course-set result to the frozen list/count/existence aggregate."""
    if result.status == "valid_empty":
        return aggregate_course_set((), evidence_complete=True)
    if result.kind == "topic_matches":
        if result.status != "complete":
            return aggregate_course_set((), evidence_complete=False)
        records = _topic_candidates(result)
    else:
        records = _payload_records(result, "courses")
    if records is None:
        return None
    try:
        return aggregate_course_set(
            records,
            evidence_complete=result.status == "complete",
        )
    except (TypeError, ValueError, OverflowError):
        return None


def _cached_course_aggregate(
    result: EvidenceExecutionResult,
    cache: dict[int, CourseSetAggregation | None],
) -> CourseSetAggregation | None:
    key = id(result)
    if key not in cache:
        cache[key] = _course_set_aggregate(result)
    return cache[key]


def _credit_aggregate(
    result: EvidenceExecutionResult,
) -> ComponentAggregation | None:
    """Adapt one credit result using authoritative counted-credit components."""
    if result.status == "valid_empty":
        return aggregate_sum_credits((), evidence_complete=True)
    records = _payload_records(result, "components")
    if records is None:
        return None
    try:
        return aggregate_sum_credits(
            records,
            evidence_complete=result.status == "complete",
        )
    except (TypeError, ValueError, OverflowError):
        return None


def _provenance_from_records(records: Iterable[Mapping[str, Any]]) -> tuple[Any, ...]:
    """Return first-seen component provenance without deriving new references."""
    result: list[Any] = []
    for record in records:
        provenance = record.get("provenance", ())
        if isinstance(provenance, Mapping) or isinstance(provenance, str):
            provenance = (provenance,)
        if not isinstance(provenance, (list, tuple)):
            return ()
        for reference in provenance:
            if not isinstance(reference, (Mapping, str)):
                return ()
            if reference not in result:
                result.append(reference)
    return tuple(result)


def _provenance_from_value(value: Any) -> tuple[Any, ...]:
    """Read provenance from supported typed evidence without inventing it."""
    if isinstance(value, CourseSetAggregation):
        return _provenance_from_records(value.courses)
    if isinstance(value, ComponentAggregation):
        return _provenance_from_records(value.components)
    if isinstance(value, ComparisonAggregation):
        return _provenance_from_value(value.left) + _provenance_from_value(value.right)
    if isinstance(value, JudgementEvidence):
        return tuple(value.provenance)
    if isinstance(value, Mapping):
        return _provenance_from_records((value,))
    if isinstance(value, (list, tuple)) and all(isinstance(item, Mapping) for item in value):
        return _provenance_from_records(value)
    return ()


def _claim(
    operation: str,
    result: EvidenceExecutionResult,
    *,
    value: Any = None,
    evidence: Any = None,
    provenance: tuple[Any, ...] = (),
    kind: str = "deterministic_fact",
    status: str | None = None,
) -> GroundedClaim:
    """Build one immutable claim while redacting failed evidence."""
    claim_status = status or result.status
    if claim_status == "complete" and result.planned_request.provenance_required and not provenance:
        claim_status = "insufficient_evidence"
    if claim_status == "insufficient_evidence":
        value = None
        evidence = None
        provenance = ()
    return GroundedClaim(
        claim_id="pending",
        operation=operation,
        effective_scope=result.effective_scope,
        status=claim_status,
        kind=kind,
        value=value,
        evidence=evidence,
        provenance=provenance,
    )


def _relation_results(
    query_spec: Any,
    bundle: EvidenceBundle,
) -> tuple[EvidenceExecutionResult, ...]:
    """Choose topic matches for topic operations and course sets otherwise."""
    if getattr(query_spec, "topic", None) is not None:
        topic_results = _execution_results(bundle, "topic_matches")
        if topic_results:
            return topic_results
    return _execution_results(bundle, "course_set")


def _topic_candidates(
    result: EvidenceExecutionResult,
) -> tuple[Mapping[str, Any], ...] | None:
    payload = result.payload
    if not isinstance(payload, ConstrainedTopicRetrievalResult):
        return None
    if result.status != "complete":
        return ()
    candidates = _attach_effective_scope(
        payload.scored_candidates,
        result.effective_scope,
    )
    return candidates


def _claim_for_relation_operation(
    operation: str,
    result: EvidenceExecutionResult,
    cache: dict[int, CourseSetAggregation | None],
) -> GroundedClaim | None:
    aggregate = _cached_course_aggregate(result, cache)
    if aggregate is None:
        return _claim(operation, result, status="insufficient_evidence")
    if operation == "list":
        value = aggregate.courses if aggregate.status != "insufficient_evidence" else None
    elif operation == "count":
        value = aggregate.count
    else:
        value = aggregate.exists
    provenance = _provenance_from_records(aggregate.courses)
    return _claim(
        operation,
        result,
        value=value,
        evidence=aggregate,
        provenance=provenance,
        status=aggregate.status,
    )


def _claim_for_credit_operation(
    operation: str,
    result: EvidenceExecutionResult,
) -> GroundedClaim:
    aggregate = _credit_aggregate(result)
    if aggregate is None:
        return _claim(operation, result, status="insufficient_evidence")
    provenance = _provenance_from_records(aggregate.components)
    return _claim(
        operation,
        result,
        value=aggregate.value,
        evidence=aggregate,
        provenance=provenance,
        status=aggregate.status,
    )


def _claim_for_placement_operation(
    operation: str,
    result: EvidenceExecutionResult,
) -> GroundedClaim:
    records = _payload_records(result, "courses")
    if records is None:
        return _claim(operation, result, status="insufficient_evidence")
    if operation == "earliest":
        try:
            aggregate = aggregate_earliest(
                records,
                evidence_complete=result.status == "complete",
            )
        except (TypeError, ValueError, OverflowError):
            return _claim(operation, result, status="insufficient_evidence")
        return _claim(
            operation,
            result,
            value=aggregate,
            evidence=aggregate,
            provenance=tuple(
                reference
                for partition in aggregate.partitions
                for reference in partition.provenance
            ),
            status=aggregate.status,
        )
    return _claim(
        operation,
        result,
        value=records if result.status == "complete" else None,
        evidence=records if result.status == "complete" else None,
        provenance=_provenance_from_records(records),
    )


def _claim_for_prerequisite_operation(
    operation: str,
    result: EvidenceExecutionResult,
) -> GroundedClaim:
    records = _payload_records(result)
    if records is None:
        return _claim(operation, result, status="insufficient_evidence")
    return _claim(
        operation,
        result,
        value=records if result.status == "complete" else None,
        evidence=records if result.status == "complete" else None,
        provenance=_provenance_from_records(records),
    )


def _claim_for_description_operation(
    operation: str,
    result: EvidenceExecutionResult,
) -> GroundedClaim:
    records = _payload_records(result)
    if records is None:
        return _claim(operation, result, status="insufficient_evidence", kind="grounded_summary")
    return _claim(
        operation,
        result,
        value=records if result.status == "complete" else None,
        evidence=records if result.status == "complete" else None,
        provenance=_provenance_from_records(records),
        kind="grounded_summary",
    )


def _claim_for_judgement_operation(
    operation: str,
    result: EvidenceExecutionResult,
    relation_results: tuple[EvidenceExecutionResult, ...],
    credit_results: tuple[EvidenceExecutionResult, ...],
) -> GroundedClaim:
    if operation == "preference":
        if result.status != "complete":
            return _claim(operation, result, status=result.status, kind="grounded_summary")
        payload = result.payload
        if not isinstance(payload, ConstrainedTopicRetrievalResult):
            return _claim(operation, result, status="insufficient_evidence", kind="grounded_summary")
        candidates = _topic_candidates(result)
        if candidates is None:
            return _claim(operation, result, status="insufficient_evidence", kind="grounded_summary")
        evidence = evaluate_preference(candidates)
        claim_status = (
            "complete" if evidence.status == "supported" else evidence.status
        )
        return _claim(
            operation,
            result,
            value=evidence,
            evidence=evidence,
            provenance=evidence.provenance,
            kind="grounded_summary",
            status=claim_status,
        )

    if operation == "quantity":
        facts: dict[str, Any] = {}
        aggregate = _course_set_aggregate(result)
        if aggregate is not None:
            facts["count"] = aggregate
        evidence = evaluate_quantity(facts)
    else:
        facts = {}
        course_aggregate = _course_set_aggregate(result)
        if course_aggregate is not None:
            facts["count"] = course_aggregate
            records = (
                _topic_candidates(result)
                if result.kind == "topic_matches"
                else _payload_records(result, "courses")
            )
            if records is not None:
                try:
                    facts["required_load"] = aggregate_required_load(
                        records,
                        evidence_complete=result.status == "complete",
                    )
                except (TypeError, ValueError, OverflowError):
                    pass
        credit_result = next(
            (
                candidate
                for candidate in credit_results
                if candidate.effective_scope == result.effective_scope
            ),
            None,
        )
        if credit_result is not None:
            credit_aggregate = _credit_aggregate(credit_result)
            if credit_aggregate is not None:
                facts["credits"] = credit_aggregate
        evidence = evaluate_workload(facts)
    return _claim(
        operation,
        result,
        value=evidence,
        evidence=evidence,
        provenance=evidence.provenance,
        status=evidence.status,
    )


def _comparison_claims(
    result: EvidenceExecutionResult,
) -> tuple[GroundedClaim, ...]:
    payload = result.payload
    if not isinstance(payload, ComparisonAggregation):
        return ()
    provenance = _provenance_from_value(payload)
    return (
        _claim(
            "compare",
            result,
            value=payload,
            evidence=payload,
            provenance=provenance,
            status=payload.status,
        ),
    )


def _compose_evidence_claims(
    query_spec: Any,
    bundle: EvidenceBundle,
) -> tuple[GroundedClaim, ...]:
    """Adapt an EvidenceBundle into ordered, partition-preserving typed claims.

    This is intentionally private until the 4I.4d runtime integration.  It
    performs no database/vector work and delegates all derived facts to the
    frozen aggregation/judgement contracts.
    """
    if not isinstance(bundle, EvidenceBundle) or query_spec is None:
        return ()
    operations = tuple(getattr(query_spec, "operations", ()))
    judgement = getattr(query_spec, "judgement", "none")
    if judgement in {"quantity", "workload", "preference"} and judgement not in operations:
        operations += (judgement,)
    relation_results = _relation_results(query_spec, bundle)
    credit_results = _execution_results(bundle, "credit_facts")
    placement_results = _execution_results(bundle, "placement_facts")
    prerequisite_results = _execution_results(bundle, "prerequisite_facts")
    description_results = _execution_results(bundle, "description_evidence")
    comparison_results = tuple(
        result
        for result in bundle.results
        if isinstance(result.payload, ComparisonAggregation)
    )
    course_cache: dict[int, CourseSetAggregation | None] = {}
    claims: list[GroundedClaim] = []
    for operation in operations:
        if operation in {"list", "count", "existence"}:
            for result in relation_results:
                claim = _claim_for_relation_operation(operation, result, course_cache)
                if claim is not None:
                    claims.append(claim)
        elif operation == "sum_credits":
            claims.extend(
                _claim_for_credit_operation(operation, result)
                for result in credit_results
            )
        elif operation in {"placement", "earliest"}:
            claims.extend(
                _claim_for_placement_operation(operation, result)
                for result in placement_results
            )
        elif operation == "prerequisite":
            claims.extend(
                _claim_for_prerequisite_operation(operation, result)
                for result in prerequisite_results
            )
        elif operation == "describe":
            claims.extend(
                _claim_for_description_operation(operation, result)
                for result in description_results
            )
        elif operation in {"quantity", "workload", "preference"}:
            judgement_results = (
                relation_results
                if operation != "preference"
                else _execution_results(bundle, "topic_matches")
            )
            for result in judgement_results:
                claims.append(
                    _claim_for_judgement_operation(
                        operation,
                        result,
                        relation_results,
                        credit_results,
                    )
                )
        elif operation == "compare":
            for result in comparison_results:
                claims.extend(_comparison_claims(result))

    numbered: list[GroundedClaim] = []
    for index, claim in enumerate(claims, start=1):
        numbered.append(
            GroundedClaim(
                claim_id=f"claim_{index:03d}",
                operation=claim.operation,
                effective_scope=claim.effective_scope,
                status=claim.status,
                kind=claim.kind,
                value=claim.value,
                evidence=claim.evidence,
                provenance=claim.provenance,
            )
        )
    return tuple(numbered)


def _blocked_result(outcome: ResolutionOutcome) -> dict[str, Any]:
    return {
        "status": outcome.action,
        "action": outcome.action,
        "blocking_ambiguity": outcome.blocking_ambiguity,
        "context_conflicts": outcome.context_conflicts,
        "resolved_program": outcome.resolved_program,
        "resolved_plans": outcome.resolved_plans,
        "course_references": [
            {
                "reference_type": reference.reference_type,
                "reference": reference.reference,
                "candidates": [dict(candidate) for candidate in reference.candidates],
            }
            for reference in outcome.course_references
        ],
    }


def _identity_result(outcome: ResolutionOutcome) -> dict[str, Any]:
    identities: list[dict[str, Any]] = []
    for reference in outcome.course_references:
        for candidate in reference.candidates:
            identities.append(
                {
                    "reference_type": reference.reference_type,
                    "reference": reference.reference,
                    **dict(candidate),
                }
            )
    return {
        "status": "answer",
        "action": "answer",
        "operation": "identity",
        "resolved_program": outcome.resolved_program,
        "resolved_plans": outcome.resolved_plans,
        "identities": identities,
    }


def ask(
    db_path: str | Path,
    question: str,
    structured_model_callable: Callable[[str], str] | None = None,
    top_k: int = 5,
    *,
    context: QueryContext | None = None,
) -> dict[str, Any]:
    """Retrieve SQL evidence, semantic evidence, or both for one question."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    spec = parse_query_spec(question)
    resolution = resolve_query_spec(spec, db_path, context=context)
    if resolution.action != "answer":
        return {"route": None, "result": _blocked_result(resolution)}

    if "identity" in spec.operations:
        return {"route": None, "result": _identity_result(resolution)}

    route = route_question(question)
    structured_result: dict[str, Any] | None = None
    semantic_result: list[dict[str, Any]] | None = None
    if route in {"structured", "hybrid"}:
        structured_result = ask_structured(
            db_path,
            question,
            SCHEMA_PATH.read_text(encoding="utf-8"),
            structured_model_callable,
        )
    if route in {"semantic", "hybrid"}:
        semantic_result = retrieve(db_path, question, k=top_k)

    if route == "hybrid":
        result: dict[str, Any] | list[dict[str, Any]] = {
            "structured": structured_result,
            "semantic": semantic_result,
        }
    elif route == "structured":
        result = structured_result
    else:
        result = semantic_result
    return {"route": route, "result": result}


__all__ = ["ask"]
