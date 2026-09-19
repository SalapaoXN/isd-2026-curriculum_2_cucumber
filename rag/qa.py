"""Top-level curriculum QA over one relational and vector database."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
import re
from typing import Any

from rag.aggregation import (
    ComparisonAggregation,
    ComponentAggregation,
    CourseSetAggregation,
    EarliestAggregation,
    PlanComparisonAggregation,
    PlanComparisonInput,
    aggregate_plan_comparison,
    aggregate_course_set,
    aggregate_earliest,
    aggregate_required_load,
    aggregate_sum_credits,
    compare_aggregates,
)
from rag.evidence_executor import (
    EvidenceBundle,
    EvidenceExecutionResult,
    execute_evidence_plan,
    execute_exact_similarity_from_bundle,
)
from rag.evidence_planner import (
    EvidencePlan,
    EvidenceRequest,
    StructuralScope,
    build_structural_scope,
    plan_evidence,
)
from rag.grounded_answer import (
    GroundedClaim,
    compose_grounded_answer,
)
from rag.intent_compiler import compile_intent_to_query_spec
from rag.intent_interpreter import interpret_question_intent
from rag.judgement import (
    JudgementEvidence,
    evaluate_preference,
    evaluate_quantity,
    evaluate_workload,
)
from rag.query_spec import parse_query_spec
from rag.retrieval.retrieve import (
    ConstrainedTopicRetrievalResult,
    SimilarityEvidence,
)
from rag.answer import render_grounded_answer
from rag.resolution import QueryContext, ResolutionOutcome, resolve_query_spec
from rag.structured.fallback import (
    GroundedCourseCreditResult,
    GroundedCourseListResult,
    GroundedPlacementResult,
    StructuredFallbackScope,
    ground_course_credit,
    ground_course_list,
    ground_placement,
    run_structured_fallback,
)
from rag.structured.queries import exact_course_candidates, prerequisite_state


_STRUCTURED_FALLBACK_OPERATIONS = frozenset(
    {"list", "count", "existence", "sum_credits", "placement", "earliest", "prerequisite"}
)
_STRUCTURED_FALLBACK_CUE = re.compile(
    r"วิชา|หลักสูตร|ลงเรียน|ลงทะเบียน|หน่วยกิต|เครดิต|วิชาบังคับ|"
    r"ปี|เทอม|ภาคเรียน|course|semester|year|credits?|prerequisite",
    re.IGNORECASE,
)
_CATEGORY_FILTER_RESIDUE = re.compile(
    r"ศึกษาทั่วไป|(?<![A-Za-z0-9_])gen\s*ed(?![A-Za-z0-9_])|วิชาเลือก",
    re.IGNORECASE,
)
_REQUIREMENT_FILTER_RESIDUE = re.compile(r"วิชาบังคับ(?!\s*ก่อน)", re.IGNORECASE)
_CREDIT_UNIT_FILTER_RESIDUE = re.compile(
    r"(?<!\d)\d+(?:\.\d+)?\s*หน่วยกิต", re.IGNORECASE
)
_LIST_FILTER_FALLBACK_CUE = re.compile(
    r"มีวิชา(?:[^?\n]{0,80})?(?:อะไร|ไหน)(?:บ้าง)?|"
    r"ลงเรียนวิชา[^?\n]{0,80}(?:อะไร|ไหน)(?:บ้าง)?",
    re.IGNORECASE,
)
_PLACEMENT_FALLBACK_CUE = re.compile(
    r"สามารถลงได้[^?\n]{0,50}(?:ช่วงไหน|ตอนไหน|ปีไหน|เทอมไหน)|"
    r"(?:อยู่ช่วงไหน|เรียนตอนไหน|ลงตอนไหน|ลงเรียนช่วงไหน|ลงเรียนตอนไหน|เรียนปีไหน|เทอมไหน)",
    re.IGNORECASE,
)
_COURSE_CREDIT_FALLBACK_CUE = re.compile(
    r"กี่\s*หน่วย(?:กิต)?(?:อะ|นะ|ครับ|คะ)?(?![ก-๙A-Za-z0-9_])|"
    r"กี่\s*เครดิต(?![ก-๙A-Za-z0-9_])|"
    r"(?<![A-Za-z0-9_])credits?(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_COURSE_CREDIT_EXCLUDE_CUE = re.compile(
    r"ลงทะเบียน|รวม|ทั้งหมด|ทุกวิชา|หมวด|วิชาบังคับ|วิชาเลือก|มีวิชา|กี่รายวิชา",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class StructuredParseCompleteness:
    """Conservative eligibility result for the future SQL fallback."""

    classification: str
    missing_filters: tuple[str, ...] = ()
    program: str | None = None
    plans: tuple[str, ...] = ()
    years: tuple[int, ...] = ()
    semesters: tuple[int, ...] = ()
    course_codes: tuple[str, ...] = ()
    course_name: str | None = None


def _classify_structured_parse_completeness(
    spec: Any,
    resolution: ResolutionOutcome,
    context: QueryContext | None = None,
) -> StructuredParseCompleteness:
    """Classify only bounded structured residue; never infer missing facts."""
    program = getattr(spec, "program", None) or getattr(context, "program", None)
    common = {
        "program": program,
        "plans": tuple(getattr(spec, "plans", ())),
        "years": tuple(getattr(spec, "years", ())),
        "semesters": tuple(getattr(spec, "semesters", ())),
        "course_codes": tuple(getattr(spec, "course_codes", ())),
        "course_name": getattr(spec, "course_name", None),
    }
    not_eligible = StructuredParseCompleteness(
        "not_eligible",
        **common,
    )
    if getattr(resolution, "action", None) != "answer":
        return not_eligible
    if not isinstance(program, str) or not program.strip():
        return not_eligible
    if getattr(spec, "judgement", None) == "unsupported":
        return not_eligible

    question = getattr(spec, "normalized_question", "")
    missing_filters: list[str] = []
    if (
        getattr(spec, "category", None) is None
        and _CATEGORY_FILTER_RESIDUE.search(question)
    ):
        missing_filters.append("category")
    if _REQUIREMENT_FILTER_RESIDUE.search(question):
        missing_filters.append("requirement_type")
    if _CREDIT_UNIT_FILTER_RESIDUE.search(question):
        missing_filters.append("credit_units")

    if missing_filters:
        return StructuredParseCompleteness(
            "partial",
            missing_filters=tuple(missing_filters),
            **common,
        )

    operations = tuple(getattr(spec, "operations", ()))
    if any(operation not in _STRUCTURED_FALLBACK_OPERATIONS for operation in operations):
        return not_eligible
    if getattr(spec, "topic", None) is not None:
        return not_eligible

    if operations:
        return StructuredParseCompleteness("complete", **common)

    has_scope = bool(
        common["plans"]
        or common["years"]
        or common["semesters"]
        or common["course_codes"]
        or common["course_name"]
    )
    if has_scope and (
        _STRUCTURED_FALLBACK_CUE.search(question)
        or _PLACEMENT_FALLBACK_CUE.search(question)
        or _COURSE_CREDIT_FALLBACK_CUE.search(question)
    ):
        return StructuredParseCompleteness(
            "unrecognized_structured",
            **common,
        )
    return not_eligible


def _is_course_list_fallback_candidate(
    spec: Any,
    completeness: StructuredParseCompleteness,
) -> bool:
    """Allow only bounded course-list/filter residue into the SQL seam."""
    if getattr(spec, "topic", None) is not None:
        return False
    operations = tuple(getattr(spec, "operations", ()))
    if operations:
        if operations in {("count",), ("existence",)}:
            return True
        return "list" in operations and set(operations) <= {
            "list",
            "sum_credits",
        }
    return bool(
        completeness.classification == "unrecognized_structured"
        and _LIST_FILTER_FALLBACK_CUE.search(
            getattr(spec, "normalized_question", "")
        )
    )


def _is_placement_fallback_candidate(
    spec: Any,
    completeness: StructuredParseCompleteness,
) -> bool:
    """Allow only bounded placement wording into the placement SQL seam."""
    if completeness.missing_filters:
        return False
    operations = tuple(getattr(spec, "operations", ()))
    if operations:
        return operations == ("placement",)
    return bool(
        completeness.classification == "unrecognized_structured"
        and _PLACEMENT_FALLBACK_CUE.search(
            getattr(spec, "normalized_question", "")
        )
    )


def _is_course_credit_fallback_candidate(
    spec: Any,
    completeness: StructuredParseCompleteness,
    resolution: ResolutionOutcome,
) -> bool:
    """Allow only one exact logical course into the credit SQL seam."""
    if completeness.missing_filters or tuple(getattr(spec, "operations", ())):
        return False

    question = getattr(spec, "normalized_question", "")
    if not _COURSE_CREDIT_FALLBACK_CUE.search(question):
        return False
    if _COURSE_CREDIT_EXCLUDE_CUE.search(question):
        return False

    course_codes: list[str] = []
    for code in completeness.course_codes:
        if isinstance(code, str) and code.strip() and code not in course_codes:
            course_codes.append(code)
    for reference in resolution.course_references:
        for candidate in reference.candidates:
            code = candidate.get("course_code")
            if isinstance(code, str) and code.strip() and code not in course_codes:
                course_codes.append(code)

    return len(resolution.course_references) == 1 and len(course_codes) == 1


def _should_use_intent_interpreter(
    spec: Any,
    completeness: StructuredParseCompleteness,
    resolution: ResolutionOutcome,
    context: QueryContext | None = None,
) -> bool:
    """Allow one bounded interpreter attempt for operation-free long-tail input."""
    if getattr(resolution, "action", None) != "answer":
        return False
    if completeness.classification not in {
        "unrecognized_structured",
        "not_eligible",
    }:
        return False
    if tuple(getattr(spec, "operations", ())) or getattr(
        spec, "judgement", None
    ) == "unsupported":
        return False

    program = getattr(spec, "program", None) or getattr(context, "program", None)
    if not isinstance(program, str) or not program.strip():
        return False

    # These are already bounded deterministic fields.  They make the input a
    # plausible structured long-tail question without adding a new phrase
    # catalog or allowing arbitrary prose to invoke a model.
    return bool(
        getattr(spec, "topic", None)
        or getattr(spec, "course_codes", ())
        or getattr(spec, "course_name", None)
        or getattr(spec, "plans", ())
        or getattr(spec, "years", ())
        or getattr(spec, "semesters", ())
        or getattr(spec, "judgement", None) in {"workload", "preference"}
    )


def _intent_authoritative_scope(
    spec: Any,
    context: QueryContext | None = None,
) -> dict[str, Any]:
    """Return only deterministic scope values permitted for intent validation."""
    plans = list(getattr(spec, "plans", ()))
    context_plan = getattr(context, "plan", None)
    if context_plan and context_plan not in plans:
        plans.append(context_plan)
    return {
        "authoritative_program": getattr(spec, "program", None)
        or getattr(context, "program", None),
        "authoritative_plans": tuple(plans),
        "authoritative_years": tuple(getattr(spec, "years", ())),
        "authoritative_semesters": tuple(getattr(spec, "semesters", ())),
        "allowed_course_codes": tuple(getattr(spec, "course_codes", ())),
    }


def _intent_failure_result(question: str) -> dict[str, Any]:
    grounded = compose_grounded_answer(
        resolution_status="insufficient_evidence",
    )
    return {
        "route": None,
        "result": render_grounded_answer(
            grounded,
            answer_model_callable=None,
            question=question,
        ),
    }


def _fallback_scope(
    completeness: StructuredParseCompleteness,
    resolution: ResolutionOutcome,
    *,
    placement_code_identity: bool = False,
) -> StructuredFallbackScope | None:
    """Build fallback scope only from deterministic parser/resolver state."""
    program = completeness.program or resolution.resolved_program
    if not isinstance(program, str) or not program.strip():
        return None

    plans = tuple(resolution.resolved_plans or completeness.plans)
    course_ids: list[int] = []
    course_codes: list[str] = list(completeness.course_codes)
    for reference in resolution.course_references:
        for candidate in reference.candidates:
            candidate_id = candidate.get("course_id")
            if isinstance(candidate_id, int) and not isinstance(candidate_id, bool):
                if candidate_id not in course_ids:
                    course_ids.append(candidate_id)
            candidate_code = candidate.get("course_code")
            if isinstance(candidate_code, str) and candidate_code.strip():
                if candidate_code not in course_codes:
                    course_codes.append(candidate_code)

    # A resolved course_id is catalog-local.  For placement fallback, an
    # exact course-code reference remains authoritative across all applicable
    # plan/catalog partitions unless no exact code was resolved.  Do not let
    # one resolver candidate narrow an otherwise unconstrained placement set.
    if placement_code_identity and course_codes:
        course_ids = []

    try:
        return StructuredFallbackScope(
            program=program,
            plans=plans,
            years=completeness.years,
            semesters=completeness.semesters,
            course_ids=tuple(course_ids),
            course_codes=tuple(course_codes),
        )
    except (TypeError, ValueError):
        return None


def _fallback_course_list_claim(
    grounded: GroundedCourseListResult,
    scope: StructuredFallbackScope,
    *,
    operation: str = "list",
) -> GroundedClaim:
    """Adapt canonical course-set facts to the requested relation claim."""
    if operation not in {"list", "count", "existence"}:
        raise ValueError(f"unsupported fallback relation operation: {operation!r}")

    effective_scope = StructuralScope(
        program=scope.program,
        plans=scope.plans,
        years=scope.years,
        semesters=scope.semesters,
    )
    if grounded.status == "insufficient_evidence":
        return GroundedClaim(
            claim_id=f"fallback_{operation}",
            operation=operation,
            effective_scope=effective_scope,
            status="insufficient_evidence",
        )

    evidence_complete = grounded.status in {"complete", "valid_empty"}
    try:
        aggregate = aggregate_course_set(
            grounded.records,
            evidence_complete=evidence_complete,
        )
    except (TypeError, ValueError, OverflowError):
        return GroundedClaim(
            claim_id=f"fallback_{operation}",
            operation=operation,
            effective_scope=effective_scope,
            status="insufficient_evidence",
        )

    value = {
        "list": aggregate.courses,
        "count": aggregate.count,
        "existence": aggregate.exists,
    }[operation]

    return GroundedClaim(
        claim_id=f"fallback_{operation}",
        operation=operation,
        effective_scope=effective_scope,
        status=aggregate.status,
        value=value,
        evidence=aggregate,
        provenance=_provenance_from_records(aggregate.courses),
    )


def _course_list_fallback_operation(spec: Any) -> str | None:
    """Return the relation operation selected by deterministic parsing only."""
    operations = tuple(getattr(spec, "operations", ()))
    if operations == ("count",):
        return "count"
    if operations == ("existence",):
        return "existence"
    if "list" in operations and set(operations) <= {"list", "sum_credits"}:
        return "list"
    if not operations:
        return "list"
    return None


def _is_filtered_sum_credits_fallback_candidate(
    spec: Any,
    completeness: StructuredParseCompleteness,
) -> bool:
    """Allow only bounded partial credit filters into the list selector."""
    return (
        completeness.classification == "partial"
        and tuple(getattr(spec, "operations", ())) == ("sum_credits",)
        and bool(completeness.missing_filters)
        and set(completeness.missing_filters)
        <= {"category", "requirement_type", "credit_units"}
        and getattr(spec, "topic", None) is None
    )


def _filtered_credit_plan(
    spec: Any,
    resolution: ResolutionOutcome,
    selected_targets: tuple[Mapping[str, Any], ...],
) -> EvidencePlan:
    """Build only the filtered credit request after selector grounding."""
    scope = replace(
        build_structural_scope(spec, resolution),
        course_targets=selected_targets,
    )
    request = EvidenceRequest(
        request_id="credit_facts",
        kind="credit_facts",
        scope=scope,
        course_targets=selected_targets,
        provenance_required=True,
    )
    return EvidencePlan(scope=scope, requests=(request,), group_by=scope.group_by)


def _filtered_credit_status_claim(
    scope: StructuralScope,
    status: str,
) -> GroundedClaim:
    """Represent selector-empty/failure states without executing broad credit facts."""
    if status == "valid_empty":
        aggregate = ComponentAggregation(
            operation="sum_credits",
            status="valid_empty",
            value=0,
        )
        return GroundedClaim(
            claim_id="fallback_sum_credits",
            operation="sum_credits",
            effective_scope=scope,
            status="valid_empty",
            value=0,
            evidence=aggregate,
        )
    return GroundedClaim(
        claim_id="fallback_sum_credits",
        operation="sum_credits",
        effective_scope=scope,
        status="insufficient_evidence",
    )


def _fallback_placement_claim(
    grounded: GroundedPlacementResult,
    scope: StructuredFallbackScope,
) -> GroundedClaim:
    """Adapt canonical fallback placement records to the typed claim shape."""
    effective_scope = StructuralScope(
        program=scope.program,
        plans=scope.plans,
        years=scope.years,
        semesters=scope.semesters,
    )
    if grounded.status == "insufficient_evidence":
        return GroundedClaim(
            claim_id="fallback_placement",
            operation="placement",
            effective_scope=effective_scope,
            status="insufficient_evidence",
        )
    if grounded.status == "valid_empty":
        return GroundedClaim(
            claim_id="fallback_placement",
            operation="placement",
            effective_scope=effective_scope,
            status="valid_empty",
        )
    return GroundedClaim(
        claim_id="fallback_placement",
        operation="placement",
        effective_scope=effective_scope,
        status="complete",
        value=grounded.records,
        evidence=grounded.records,
        provenance=_provenance_from_records(grounded.records),
    )


def _fallback_course_credit_claim(
    grounded: GroundedCourseCreditResult,
    scope: StructuredFallbackScope,
) -> GroundedClaim:
    """Adapt canonical fallback credit facts to the existing credit claim shape."""
    effective_scope = StructuralScope(
        program=scope.program,
        plans=scope.plans,
        years=scope.years,
        semesters=scope.semesters,
        course_targets=tuple(
            {"course_code": course_code} for course_code in scope.course_codes
        ),
    )
    if grounded.status == "insufficient_evidence":
        return GroundedClaim(
            claim_id="fallback_course_credit",
            operation="sum_credits",
            effective_scope=effective_scope,
            status="insufficient_evidence",
        )
    if grounded.status == "valid_empty":
        aggregate = ComponentAggregation(
            operation="sum_credits",
            status="valid_empty",
            value=0,
        )
    else:
        try:
            aggregate = ComponentAggregation(
                operation="sum_credits",
                status="complete",
                value=grounded.credit_units,
                components=grounded.records,
            )
        except (TypeError, ValueError, OverflowError):
            return GroundedClaim(
                claim_id="fallback_course_credit",
                operation="sum_credits",
                effective_scope=effective_scope,
                status="insufficient_evidence",
            )
    return GroundedClaim(
        claim_id="fallback_course_credit",
        operation="sum_credits",
        effective_scope=effective_scope,
        status=aggregate.status,
        value=aggregate.value,
        evidence=aggregate,
        provenance=grounded.provenance,
    )


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
    if isinstance(value, EarliestAggregation):
        return tuple(
            reference
            for partition in value.partitions
            for reference in partition.provenance
        )
    if isinstance(value, ComparisonAggregation):
        return _provenance_from_value(value.left) + _provenance_from_value(value.right)
    if isinstance(value, PlanComparisonAggregation):
        return tuple(value.provenance)
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
    effective_scope: Any = None,
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
        effective_scope=(
            result.effective_scope if effective_scope is None else effective_scope
        ),
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


def _missing_greatest_credit_claim() -> GroundedClaim:
    """Represent an unsupported or incomplete greatest-credit result."""
    return GroundedClaim(
        claim_id="pending",
        operation="sum_credits",
        status="insufficient_evidence",
    )


def _greatest_credit_claims(
    query_spec: Any,
    bundle: EvidenceBundle,
    results: tuple[EvidenceExecutionResult, ...],
) -> tuple[GroundedClaim, ...] | None:
    """Select maximum concrete-term credit claims independently per plan."""
    operations = set(getattr(query_spec, "operations", ()))
    if (
        operations != {"sum_credits", "compare"}
        or tuple(getattr(query_spec, "group_by", ())) != ("semester",)
        or getattr(query_spec, "years", ())
        or getattr(query_spec, "semesters", ())
        or getattr(query_spec, "course_codes", ())
        or getattr(query_spec, "course_name", None) is not None
    ):
        return None

    if not isinstance(bundle, EvidenceBundle) or not results:
        return (_missing_greatest_credit_claim(),)

    requested_program = getattr(query_spec, "program", None)
    candidates: dict[tuple[str, str], dict[tuple[int, int], ComponentAggregation]] = {}
    representatives: dict[tuple[str, str], EvidenceExecutionResult] = {}
    invalid_plans: set[tuple[str, str]] = set()
    global_invalid = False

    for result in results:
        scope = result.effective_scope
        program = getattr(scope, "program", None)
        plans = tuple(getattr(scope, "plans", ()))
        years = tuple(getattr(scope, "years", ()))
        semesters = tuple(getattr(scope, "semesters", ()))
        if (
            not isinstance(program, str)
            or not program.strip()
            or (isinstance(requested_program, str) and program != requested_program)
            or len(plans) != 1
            or not isinstance(plans[0], str)
            or not plans[0].strip()
            or len(years) != 1
            or isinstance(years[0], bool)
            or not isinstance(years[0], int)
            or len(semesters) != 1
            or isinstance(semesters[0], bool)
            or not isinstance(semesters[0], int)
        ):
            global_invalid = True
            continue

        plan_key = (program, plans[0])
        representatives.setdefault(plan_key, result)
        candidate_key = (years[0], semesters[0])
        if candidate_key in candidates.setdefault(plan_key, {}):
            invalid_plans.add(plan_key)
            continue

        aggregate = _credit_aggregate(result)
        if result.status == "complete":
            valid_status = aggregate is not None and aggregate.status == "complete"
        elif result.status == "valid_empty":
            valid_status = aggregate is not None and aggregate.status == "valid_empty"
        else:
            valid_status = False
        if not valid_status or aggregate is None or aggregate.value is None:
            invalid_plans.add(plan_key)
            continue

        provenance = _provenance_from_records(aggregate.components)
        if result.planned_request.provenance_required and aggregate.status == "complete" and not provenance:
            invalid_plans.add(plan_key)
            continue

        for component in aggregate.components:
            partition = component.get("partition")
            if (
                not isinstance(partition, Mapping)
                or partition.get("program") != program
                or tuple(partition.get("plans", ())) != (plans[0],)
                or tuple(partition.get("years", ())) != years
                or tuple(partition.get("semesters", ())) != semesters
            ):
                invalid_plans.add(plan_key)
                break
        else:
            candidates[plan_key][candidate_key] = aggregate

    if global_invalid or not candidates:
        return (_missing_greatest_credit_claim(),)

    expected: dict[tuple[str, str], set[tuple[int, int]]] = {}
    relation_results = _relation_results(query_spec, bundle)
    if relation_results:
        for result in relation_results:
            scope = result.effective_scope
            program = getattr(scope, "program", None)
            plans = tuple(getattr(scope, "plans", ()))
            years = tuple(getattr(scope, "years", ()))
            semesters = tuple(getattr(scope, "semesters", ()))
            if (
                not isinstance(program, str)
                or len(plans) != 1
                or not isinstance(plans[0], str)
                or not years
                or not semesters
            ):
                global_invalid = True
                continue
            plan_key = (program, plans[0])
            expected.setdefault(plan_key, set()).update(
                (year, semester)
                for year in years
                for semester in semesters
            )
    else:
        scope = getattr(bundle.plan, "scope", None)
        plans = tuple(getattr(scope, "plans", ()))
        years = tuple(getattr(scope, "years", ()))
        semesters = tuple(getattr(scope, "semesters", ()))
        program = getattr(scope, "program", None)
        if plans and years and semesters and isinstance(program, str):
            for plan in plans:
                expected[(program, plan)] = {
                    (year, semester)
                    for year in years
                    for semester in semesters
                }

    if global_invalid:
        return (_missing_greatest_credit_claim(),)

    plan_keys = set(candidates) | set(expected)
    claims: list[GroundedClaim] = []
    for plan_key in sorted(plan_keys):
        plan_candidates = candidates.get(plan_key, {})
        if (
            plan_key in invalid_plans
            or not plan_candidates
            or (
                plan_key in expected
                and set(plan_candidates) != expected[plan_key]
            )
        ):
            representative = representatives.get(plan_key)
            if representative is None:
                claims.append(_missing_greatest_credit_claim())
            else:
                claims.append(
                    _claim(
                        "sum_credits",
                        representative,
                        status="insufficient_evidence",
                        effective_scope=replace(
                            representative.effective_scope,
                            years=(),
                            semesters=(),
                            group_by=("plan",),
                        ),
                    )
                )
            continue

        maximum = max(aggregate.value for aggregate in plan_candidates.values())
        winners = sorted(
            (
                term,
                aggregate,
            )
            for term, aggregate in plan_candidates.items()
            if aggregate.value == maximum
        )
        for term, aggregate in winners:
            representative = next(
                result
                for result in results
                if result.effective_scope.program == plan_key[0]
                and tuple(result.effective_scope.plans) == (plan_key[1],)
                and tuple(result.effective_scope.years) == (term[0],)
                and tuple(result.effective_scope.semesters) == (term[1],)
            )
            claims.append(
                _claim(
                    "sum_credits",
                    representative,
                    value=aggregate.value,
                    evidence=aggregate,
                    provenance=_provenance_from_records(aggregate.components),
                    status=aggregate.status,
                )
            )
    return tuple(claims)


def _coarse_year_credit_claims(
    query_spec: Any,
    results: tuple[EvidenceExecutionResult, ...],
) -> tuple[GroundedClaim, ...] | None:
    """Roll semester credit facts up to one requested year per concrete plan."""
    years = tuple(getattr(query_spec, "years", ()))
    semesters = tuple(getattr(query_spec, "semesters", ()))
    group_by = tuple(getattr(query_spec, "group_by", ()))
    if len(years) != 1 or semesters or "semester" in group_by or not results:
        return None

    requested_year = years[0]
    grouped: dict[
        tuple[str, str, int],
        list[tuple[EvidenceExecutionResult, tuple[Mapping[str, Any], ...] | None]],
    ] = {}
    for result in results:
        scope = result.effective_scope
        program = getattr(scope, "program", None)
        plans = tuple(getattr(scope, "plans", ()))
        result_years = tuple(getattr(scope, "years", ()))
        result_semesters = tuple(getattr(scope, "semesters", ()))
        if (
            not isinstance(program, str)
            or not program.strip()
            or len(plans) != 1
            or not isinstance(plans[0], str)
            or not plans[0].strip()
            or len(result_years) != 1
            or result_years[0] != requested_year
            or len(result_semesters) != 1
        ):
            return None

        records: tuple[Mapping[str, Any], ...] | None
        if result.status == "insufficient_evidence":
            records = None
        elif result.status == "valid_empty":
            records = ()
        elif result.status == "complete":
            records = _payload_records(result, "components")
            if records is None:
                return None
        else:
            return None
        key = (program, plans[0], requested_year)
        grouped.setdefault(key, []).append((result, records))

    claims: list[GroundedClaim] = []
    for (_program, _plan, _year), entries in grouped.items():
        representative, _ = entries[0]
        rollup_scope = replace(
            representative.effective_scope,
            semesters=(),
            group_by=tuple(
                axis
                for axis in representative.effective_scope.group_by
                if axis != "semester"
            ),
        )
        if any(records is None for _, records in entries):
            claims.append(
                _claim(
                    "sum_credits",
                    representative,
                    status="insufficient_evidence",
                    effective_scope=rollup_scope,
                )
            )
            continue

        components = tuple(
            component
            for _, records in entries
            for component in records or ()
        )
        try:
            aggregate = aggregate_sum_credits(components, evidence_complete=True)
        except (TypeError, ValueError, OverflowError):
            claims.append(
                _claim(
                    "sum_credits",
                    representative,
                    status="insufficient_evidence",
                    effective_scope=rollup_scope,
                )
            )
            continue
        claims.append(
            _claim(
                "sum_credits",
                representative,
                value=aggregate.value,
                evidence=aggregate,
                provenance=_provenance_from_records(aggregate.components),
                status=aggregate.status,
                effective_scope=rollup_scope,
            )
        )
    return tuple(claims)


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


def _normalized_prerequisite_title(record: Mapping[str, Any]) -> str | None:
    for field in (
        "prerequisite_name_en",
        "prerequisite_name_th",
        "name_en",
        "name_th",
        "course_name",
    ):
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            return " ".join(value.casefold().split())
    return None


def _prerequisite_semantic_fingerprint(
    records: Iterable[Mapping[str, Any]],
) -> tuple[tuple[Any, ...], ...] | None:
    fingerprint: list[tuple[Any, ...]] = []
    for record in records:
        if not isinstance(record, Mapping):
            return None
        alternatives = record.get("alternative_courses")
        if isinstance(alternatives, Mapping):
            alternatives = (alternatives,)
        if isinstance(alternatives, (list, tuple)) and alternatives:
            member_titles: list[str] = []
            for member in alternatives:
                if not isinstance(member, Mapping):
                    return None
                title = _normalized_prerequisite_title(member)
                if title is None:
                    return None
                member_titles.append(title)
            group = record.get("alternative_group")
            minimum = record.get("minimum_choices")
            maximum = record.get("maximum_choices")
            if isinstance(group, Mapping):
                minimum = group.get("minimum_choices", minimum)
                maximum = group.get("maximum_choices", maximum)
            fingerprint.append(
                (
                    "alternative",
                    record.get("requirement_type"),
                    minimum,
                    maximum,
                    tuple(member_titles),
                )
            )
            continue
        title = _normalized_prerequisite_title(record)
        if title is None:
            return None
        fingerprint.append(("direct", record.get("requirement_type"), title))
    return tuple(fingerprint)


def _merge_consensus_prerequisites(
    record_sets: tuple[tuple[Mapping[str, Any], ...], ...],
) -> tuple[Mapping[str, Any], ...]:
    if not record_sets:
        return ()
    merged: list[dict[str, Any]] = [dict(record) for record in record_sets[0]]
    for index, record in enumerate(merged):
        references: list[Any] = []
        for records in record_sets:
            provenance = records[index].get("provenance", ())
            if isinstance(provenance, Mapping) or isinstance(provenance, str):
                provenance = (provenance,)
            if isinstance(provenance, (list, tuple)):
                for reference in provenance:
                    if reference not in references:
                        references.append(reference)
        record["provenance"] = tuple(references)
    return tuple(merged)


def _consensus_prerequisite_grounded_answer(
    db_path: str | Path,
    spec: Any,
    resolution: ResolutionOutcome,
    context: QueryContext | None,
) -> Any | None:
    """Answer only unanimous, program-free exact prerequisite references."""
    if spec.operations != ("prerequisite",):
        return None
    if getattr(spec, "program", None) is not None:
        return None
    if context is not None and context.program is not None:
        return None
    if len(resolution.course_references) != 1:
        return None
    reference = resolution.course_references[0]
    if spec.course_name is not None:
        candidates = exact_course_candidates(
            db_path,
            course_name=spec.course_name,
            exact_title=True,
        )
    else:
        candidates = list(reference.candidates)
    programs = {
        str(candidate.get("program"))
        for candidate in candidates
        if candidate.get("program") not in (None, "")
    }
    if len(programs) < 2:
        return None

    states: list[Mapping[str, Any]] = []
    fingerprints: list[tuple[tuple[Any, ...], ...] | None] = []
    for candidate in candidates:
        course_id = candidate.get("course_id")
        if isinstance(course_id, bool) or not isinstance(course_id, int):
            return None
        state = prerequisite_state(db_path, course_id)
        states.append(state)
        if state.get("state") == "explicit_none":
            fingerprints.append((('explicit_none',),))
        elif state.get("state") == "required":
            fingerprints.append(
                _prerequisite_semantic_fingerprint(state.get("records", ()))
            )
        else:
            fingerprints.append(None)
    if not fingerprints or any(fingerprint is None for fingerprint in fingerprints):
        return None
    first = fingerprints[0]
    if any(fingerprint != first for fingerprint in fingerprints[1:]):
        return None
    if first == (('explicit_none',),):
        merged_provenance: list[Any] = []
        for state in states:
            for reference in state.get("provenance", ()):
                if reference not in merged_provenance:
                    merged_provenance.append(reference)
        merged = (
            {
                "prerequisite_state": "explicit_none",
                "prerequisite_text": states[0].get("prerequisite_text"),
                "provenance": tuple(merged_provenance),
            },
        )
    else:
        merged = _merge_consensus_prerequisites(
            tuple(tuple(state.get("records", ())) for state in states)
        )
    provenance_values: list[Any] = list(_provenance_from_records(merged))
    for candidate in candidates:
        candidate_provenance = candidate.get("provenance", ())
        if isinstance(candidate_provenance, Mapping) or isinstance(candidate_provenance, str):
            candidate_provenance = (candidate_provenance,)
        if isinstance(candidate_provenance, (list, tuple)):
            for reference in candidate_provenance:
                if reference not in provenance_values:
                    provenance_values.append(reference)
    provenance = tuple(provenance_values)
    claim_status = "valid_empty" if first == (('explicit_none',),) else "complete"
    claim = GroundedClaim(
        claim_id="prerequisite_consensus",
        operation="prerequisite",
        status=claim_status,
        kind="deterministic_fact",
        value=merged,
        evidence=merged,
        provenance=provenance,
    )
    return compose_grounded_answer(composed_claims=(claim,))


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
    return _comparison_claim_from_payload(
        payload,
        effective_scope=result.effective_scope,
        provenance_required=result.planned_request.provenance_required,
    )


def _comparison_claim_from_payload(
    payload: ComparisonAggregation | PlanComparisonAggregation,
    *,
    effective_scope: Any = None,
    provenance_required: bool = True,
) -> tuple[GroundedClaim, ...]:
    """Compose one comparison payload while preserving its typed operands."""
    provenance = _provenance_from_value(payload)
    status = payload.status
    if status == "complete" and provenance_required and not provenance:
        status = "insufficient_evidence"
    value = payload if status != "insufficient_evidence" else None
    evidence = payload if status != "insufficient_evidence" else None
    retained_provenance = provenance if status != "insufficient_evidence" else ()
    return (
        GroundedClaim(
            claim_id="pending",
            operation="compare",
            effective_scope=effective_scope,
            status=status,
            kind="deterministic_fact",
            value=value,
            evidence=evidence,
            provenance=retained_provenance,
        ),
    )


def _whole_plan_comparison_payload(
    query_spec: Any,
    bundle: EvidenceBundle,
) -> PlanComparisonAggregation | None:
    """Build a typed comparison for exactly two complete plan partitions."""
    operations = tuple(getattr(query_spec, "operations", ()))
    if "compare" not in operations:
        return None
    if tuple(getattr(query_spec, "group_by", ())) != ("plan",):
        return None
    if getattr(query_spec, "course_codes", ()) or getattr(query_spec, "course_name", None):
        return None

    plans = tuple(getattr(query_spec, "plans", ()))
    if len(plans) != 2 or len(set(plans)) != 2:
        return None

    def results_by_plan(
        kind: str,
    ) -> dict[str, tuple[EvidenceExecutionResult, tuple[Mapping[str, Any], ...]]] | None:
        grouped: dict[
            str, tuple[EvidenceExecutionResult, tuple[Mapping[str, Any], ...]]
        ] = {}
        for result in _execution_results(bundle, kind):
            if result.status != "complete":
                return None
            scope = result.effective_scope
            result_plans = tuple(getattr(scope, "plans", ()))
            if len(result_plans) != 1 or result_plans[0] not in plans:
                return None
            plan = result_plans[0]
            if plan in grouped:
                return None
            records = _payload_records(result, "courses")
            if records is None:
                return None
            grouped[plan] = (result, records)
        if set(grouped) != set(plans):
            return None
        return grouped

    course_results = results_by_plan("course_set")
    placement_results = results_by_plan("placement_facts")
    if course_results is None or placement_results is None:
        return None

    inputs: list[PlanComparisonInput] = []
    for plan in plans:
        course_result, _course_records = course_results[plan]
        _placement_result, placement_records = placement_results[plan]
        course_aggregate = _course_set_aggregate(course_result)
        if course_aggregate is None or course_aggregate.status != "complete":
            return None
        inputs.append(
            PlanComparisonInput(
                plan=plan,
                course_set=course_aggregate,
                placements=placement_records,
                placements_complete=True,
            )
        )

    try:
        return aggregate_plan_comparison(inputs[0], inputs[1])
    except (TypeError, ValueError, OverflowError):
        return None


def _earliest_comparison_payload(
    query_spec: Any,
    bundle: EvidenceBundle,
) -> ComparisonAggregation | None:
    """Build the frozen two-plan earliest comparison from placement evidence."""
    operations = set(getattr(query_spec, "operations", ()))
    if not {"placement", "earliest", "compare"}.issubset(operations):
        return None
    if tuple(getattr(query_spec, "group_by", ())) != ("plan",):
        return None
    course_codes = tuple(getattr(query_spec, "course_codes", ()))
    if len(course_codes) != 1 or not isinstance(course_codes[0], str):
        return None
    requested_course_code = course_codes[0].strip()
    if not requested_course_code:
        return None

    grouped_records: dict[str, list[Mapping[str, Any]]] = {}
    grouped_statuses: dict[str, list[str]] = {}
    group_identity: tuple[str, str] | None = None
    placement_results = _execution_results(bundle, "placement_facts")
    if not placement_results:
        return None

    for result in placement_results:
        scope = result.effective_scope
        program = getattr(scope, "program", None)
        plans = tuple(getattr(scope, "plans", ()))
        if not isinstance(program, str) or not program.strip() or len(plans) != 1:
            return None
        plan = plans[0]
        if not isinstance(plan, str) or not plan.strip():
            return None

        request_targets = tuple(
            getattr(result.planned_request, "course_targets", ())
            or getattr(scope, "course_targets", ())
        )
        if len(request_targets) != 1:
            return None
        target_identity = _logical_target_key(request_targets[0])
        if (
            target_identity is None
            or target_identity[0] != program
            or target_identity[1] != requested_course_code
        ):
            return None
        if group_identity is None:
            group_identity = target_identity
        elif target_identity != group_identity:
            return None

        records = _payload_records(result, "courses")
        if records is None or result.status != "complete":
            return None
        for record in records:
            if _logical_target_key(record) != group_identity:
                return None
            record_plan = record.get("plan_key", record.get("plan"))
            if record_plan is not None and record_plan != plan:
                return None
            provenance = record.get("provenance")
            if (
                not isinstance(provenance, (list, tuple))
                or not provenance
                or any(not isinstance(reference, Mapping) for reference in provenance)
            ):
                return None
        grouped_records.setdefault(plan, []).extend(records)
        grouped_statuses.setdefault(plan, []).append(result.status)

    if group_identity is None or len(grouped_records) != 2:
        return None
    operands: dict[str, EarliestAggregation] = {}
    for plan, records in grouped_records.items():
        if not records or any(status != "complete" for status in grouped_statuses[plan]):
            return None
        try:
            aggregate = aggregate_earliest(records, evidence_complete=True)
        except (TypeError, ValueError, OverflowError):
            return None
        if (
            aggregate.status != "complete"
            or aggregate.value is None
            or len(aggregate.partitions) != 1
            or not _provenance_from_value(aggregate)
        ):
            return None
        operands[plan] = aggregate

    plan_scope = getattr(getattr(bundle, "plan", None), "scope", None)
    plan_order = tuple(
        plan
        for plan in getattr(plan_scope, "plans", ())
        if plan in operands
    )
    plan_order += tuple(plan for plan in operands if plan not in plan_order)
    if len(plan_order) != 2:
        return None
    try:
        return compare_aggregates(operands[plan_order[0]], operands[plan_order[1]])
    except (TypeError, ValueError, OverflowError):
        return None


def _placement_comparison_payloads(
    query_spec: Any,
    bundle: EvidenceBundle,
) -> tuple[ComparisonAggregation, ...]:
    """Compare complete exact-course placement operands within safe partitions."""
    operations = set(getattr(query_spec, "operations", ()))
    if (
        "compare" not in operations
        or "placement" not in operations
        or "earliest" in operations
    ):
        return ()

    placement_results = _execution_results(bundle, "placement_facts")
    if not placement_results:
        return ()

    requested_targets: set[tuple[str, str]] = set()
    grouped: dict[tuple[str, str], dict[tuple[str, str], list[Mapping[str, Any]]]] = {}
    for result in placement_results:
        if result.status != "complete":
            return ()
        scope = result.effective_scope
        program = getattr(scope, "program", None)
        plans = tuple(getattr(scope, "plans", ()))
        if not isinstance(program, str) or not program.strip() or len(plans) != 1:
            return ()
        plan = plans[0]
        if not isinstance(plan, str) or not plan.strip():
            return ()

        for target in tuple(
            getattr(result.planned_request, "course_targets", ())
            or getattr(scope, "course_targets", ())
        ):
            target_key = _logical_target_key(target)
            if target_key is not None:
                requested_targets.add(target_key)

        records = _payload_records(result, "courses")
        if not records:
            return ()
        identities = {_logical_target_key(record) for record in records}
        if None in identities or len(identities) != 1:
            return ()
        identity = next(iter(identities))
        if identity[0] != program or identity not in requested_targets:
            return ()
        for record in records:
            record_plan = record.get("plan_key", record.get("plan"))
            if record.get("program") != program or record_plan != plan:
                return ()
        grouped.setdefault((program, plan), {}).setdefault(identity, []).extend(records)

    if not requested_targets or len({program for program, _ in requested_targets}) != 1:
        return ()

    target_count = len(requested_targets)
    requested_plans = tuple(getattr(query_spec, "plans", ()))
    if target_count == 2:
        if any(set(targets) != requested_targets for targets in grouped.values()):
            return ()
        if len(grouped) == 0:
            return ()
    elif target_count == 1:
        if len(requested_plans) != 2 or set(requested_plans) != {
            plan for _, plan in grouped
        }:
            return ()
        if any(set(targets) != requested_targets for targets in grouped.values()):
            return ()
    else:
        return ()

    ordered_targets = tuple(sorted(requested_targets))
    if target_count == 1:
        plans = sorted(grouped)
        if len(plans) != 2:
            return ()
        target = ordered_targets[0]
        left_records = grouped[plans[0]][target]
        right_records = grouped[plans[1]][target]
        try:
            left = aggregate_earliest(left_records, evidence_complete=True)
            right = aggregate_earliest(right_records, evidence_complete=True)
            comparison = compare_aggregates(left, right)
        except (TypeError, ValueError, OverflowError):
            return ()
        if (
            left.status != "complete"
            or right.status != "complete"
            or left.value is None
            or right.value is None
            or len(left.partitions) != 1
            or len(right.partitions) != 1
            or not _provenance_from_value(left)
            or not _provenance_from_value(right)
        ):
            return ()
        return (comparison,)

    payloads: list[ComparisonAggregation] = []
    for target_records in grouped.values():
        aggregates: dict[tuple[str, str], EarliestAggregation] = {}
        for target in ordered_targets:
            try:
                aggregate = aggregate_earliest(
                    target_records[target],
                    evidence_complete=True,
                )
            except (TypeError, ValueError, OverflowError):
                return ()
            if (
                aggregate.status != "complete"
                or aggregate.value is None
                or len(aggregate.partitions) != 1
                or not _provenance_from_value(aggregate)
            ):
                return ()
            aggregates[target] = aggregate
        try:
            payloads.append(
                compare_aggregates(
                    aggregates[ordered_targets[0]],
                    aggregates[ordered_targets[1]],
                )
            )
        except (TypeError, ValueError, OverflowError):
            return ()
    return tuple(payloads)


def _missing_comparison_claim() -> GroundedClaim:
    """Represent an unexecuted comparison without inventing a relation."""
    return GroundedClaim(
        claim_id="pending",
        operation="compare",
        status="insufficient_evidence",
    )


def _similarity_provenance(
    evidence: SimilarityEvidence,
) -> tuple[Any, ...] | None:
    """Validate and retain first-seen provenance from both pair sides."""
    references: list[Any] = []
    for pair in evidence.pairs:
        for side in (pair.left, pair.right):
            provenance = side.get("provenance")
            if not isinstance(provenance, (list, tuple)) or not provenance:
                return None
            for reference in provenance:
                if not isinstance(reference, Mapping):
                    return None
                if reference not in references:
                    references.append(reference)
    return tuple(references)


def _similarity_claim(
    evidence: SimilarityEvidence | None,
    bundle: EvidenceBundle,
    request_ids: tuple[str, str] | None,
    *,
    selected_plan: str | None = None,
) -> GroundedClaim:
    """Wrap one exact similarity result without recomputing any evidence."""
    if not isinstance(evidence, SimilarityEvidence):
        return GroundedClaim(
            claim_id="pending",
            operation="similarity",
            status="insufficient_evidence",
        )

    scopes: list[Any] = []
    if request_ids is not None:
        for result in bundle.results:
            if (
                result.request_id in request_ids
                and result.kind == "description_evidence"
                and (
                    selected_plan is None
                    or tuple(getattr(result.effective_scope, "plans", ()))
                    == (selected_plan,)
                )
            ):
                if (
                    result.effective_scope is not None
                    and result.effective_scope not in scopes
                ):
                    scopes.append(result.effective_scope)
    pair_plans = {
        pair.partition.get("plan")
        for pair in evidence.pairs
        if isinstance(pair.partition, Mapping)
    }
    if len(pair_plans) == 1:
        matching_scopes = [
            scope
            for scope in scopes
            if tuple(getattr(scope, "plans", ())) == (next(iter(pair_plans)),)
        ]
        if matching_scopes:
            scopes = matching_scopes
    effective_scope = scopes[0] if len(scopes) == 1 else None
    provenance = _similarity_provenance(evidence)
    if provenance is None:
        return GroundedClaim(
            claim_id="pending",
            operation="similarity",
            effective_scope=effective_scope,
            status="insufficient_evidence",
        )
    return GroundedClaim(
        claim_id="pending",
        operation="similarity",
        effective_scope=effective_scope,
        status=evidence.status,
        kind="deterministic_fact",
        value=evidence,
        evidence=evidence,
        provenance=provenance,
    )


def _logical_target_key(target: Any) -> tuple[str, str] | None:
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
    return program, course_code


def _resolved_logical_targets(outcome: ResolutionOutcome) -> set[tuple[str, str]]:
    targets: set[tuple[str, str]] = set()
    for reference in outcome.course_references:
        for candidate in reference.candidates:
            key = _logical_target_key(candidate)
            if key is not None:
                targets.add(key)
    return targets


def _select_similarity_request_ids(
    plan: Any,
    outcome: ResolutionOutcome,
) -> tuple[str, str] | None:
    """Select exactly the planner-owned description requests for similarity."""
    requests = tuple(
        request
        for request in getattr(plan, "requests", ())
        if getattr(request, "kind", None) == "description_evidence"
        and isinstance(getattr(request, "request_id", None), str)
        and request.request_id.startswith("similarity_description_")
    )
    if len(requests) != 2:
        return None
    targets: list[tuple[str, str]] = []
    for request in requests:
        request_targets = getattr(request, "course_targets", ())
        if len(request_targets) != 1:
            return None
        key = _logical_target_key(request_targets[0])
        if key is None or key in targets:
            return None
        targets.append(key)
    scope_targets = tuple(
        key
        for key in (
            _logical_target_key(target)
            for target in getattr(getattr(plan, "scope", None), "course_targets", ())
        )
        if key is not None
    )
    resolved_targets = _resolved_logical_targets(outcome)
    if (
        len(scope_targets) != 2
        or len(set(scope_targets)) != 2
        or tuple(targets) != scope_targets
        or set(targets) != resolved_targets
    ):
        return None
    return requests[0].request_id, requests[1].request_id


def _compose_evidence_claims(
    query_spec: Any,
    bundle: EvidenceBundle,
    *,
    similarity_evidence: SimilarityEvidence | None = None,
    similarity_request_ids: tuple[str, str] | None = None,
    selected_plan: str | None = None,
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
    greatest_credit_claims = _greatest_credit_claims(
        query_spec,
        bundle,
        credit_results,
    )
    coarse_credit_claims = _coarse_year_credit_claims(query_spec, credit_results)
    placement_results = _execution_results(bundle, "placement_facts")
    prerequisite_results = _execution_results(bundle, "prerequisite_facts")
    description_results = _execution_results(bundle, "description_evidence")
    comparison_results = tuple(
        result
        for result in bundle.results
        if isinstance(result.payload, ComparisonAggregation)
    )
    generated_placement_comparisons: tuple[ComparisonAggregation, ...] = ()
    generated_plan_comparison: PlanComparisonAggregation | None = None
    generated_comparison = None
    if not comparison_results:
        generated_placement_comparisons = _placement_comparison_payloads(
            query_spec,
            bundle,
        )
    if not comparison_results and not generated_placement_comparisons:
        generated_plan_comparison = _whole_plan_comparison_payload(
            query_spec,
            bundle,
        )
    if (
        not comparison_results
        and not generated_placement_comparisons
        and generated_plan_comparison is None
    ):
        generated_comparison = _earliest_comparison_payload(query_spec, bundle)
    course_cache: dict[int, CourseSetAggregation | None] = {}
    claims: list[GroundedClaim] = []
    for operation in operations:
        if operation in {"list", "count", "existence"}:
            for result in relation_results:
                claim = _claim_for_relation_operation(operation, result, course_cache)
                if claim is not None:
                    claims.append(claim)
        elif operation == "sum_credits":
            if greatest_credit_claims is not None:
                claims.extend(greatest_credit_claims)
            elif coarse_credit_claims is not None:
                claims.extend(coarse_credit_claims)
            else:
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
            if comparison_results:
                for result in comparison_results:
                    claims.extend(_comparison_claims(result))
            elif generated_placement_comparisons:
                for payload in generated_placement_comparisons:
                    claims.extend(
                        _comparison_claim_from_payload(
                            payload,
                            effective_scope=None,
                        )
                    )
            elif generated_plan_comparison is not None:
                claims.extend(
                    _comparison_claim_from_payload(
                        generated_plan_comparison,
                        effective_scope=None,
                    )
                )
            elif generated_comparison is not None:
                claims.extend(
                    _comparison_claim_from_payload(
                        generated_comparison,
                        effective_scope=None,
                    )
                )
            elif greatest_credit_claims is not None:
                # The supported greatest-credit operation is represented by
                # its winning sum_credits claims, not a synthetic pairwise
                # ComparisonAggregation.
                pass
            else:
                claims.append(_missing_comparison_claim())
        elif operation == "similarity":
            claims.append(
                _similarity_claim(
                    similarity_evidence,
                    bundle,
                    similarity_request_ids,
                    selected_plan=selected_plan,
                )
            )

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


def _identity_result(
    outcome: ResolutionOutcome,
    *,
    operation: str = "identity",
) -> dict[str, Any]:
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
        "operation": operation,
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
    answer_model_callable: Callable[[str], str] | None = None,
    intent_model_callable: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """Run the typed evidence pipeline while retaining the legacy signature."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    spec = parse_query_spec(question)
    resolution = resolve_query_spec(spec, db_path, context=context)
    if resolution.action != "answer":
        if resolution.action == "clarify_program":
            consensus = _consensus_prerequisite_grounded_answer(
                db_path,
                spec,
                resolution,
                context,
            )
            if consensus is not None:
                return {
                    "route": None,
                    "result": render_grounded_answer(
                        consensus,
                        answer_model_callable=answer_model_callable,
                        question=question,
                    ),
                }
        return {"route": None, "result": _blocked_result(resolution)}

    if spec.operations == ("program_discovery",):
        grounded = compose_grounded_answer(
            identity_result=_identity_result(
                resolution,
                operation="program_discovery",
            ),
        )
        return {
            "route": None,
            "result": render_grounded_answer(
                grounded,
                answer_model_callable=answer_model_callable,
                question=question,
            ),
        }

    if spec.operations == ("identity",):
        grounded = compose_grounded_answer(
            identity_result=_identity_result(resolution),
        )
        return {
            "route": None,
            "result": render_grounded_answer(grounded),
        }

    completeness = _classify_structured_parse_completeness(
        spec,
        resolution,
        context,
    )
    filtered_credit_candidate = _is_filtered_sum_credits_fallback_candidate(
        spec,
        completeness,
    )
    if filtered_credit_candidate:
        filtered_scope = build_structural_scope(spec, resolution)
        if not callable(structured_model_callable):
            claim = _filtered_credit_status_claim(
                filtered_scope,
                "insufficient_evidence",
            )
        else:
            fallback_scope = _fallback_scope(completeness, resolution)
            if fallback_scope is None:
                claim = _filtered_credit_status_claim(
                    filtered_scope,
                    "insufficient_evidence",
                )
            else:
                fallback_result = run_structured_fallback(
                    db_path,
                    question,
                    fallback_scope,
                    structured_model_callable,
                )
                grounded_list = ground_course_list(
                    db_path,
                    fallback_result,
                    fallback_scope,
                )
                if grounded_list.status != "complete":
                    claim = _filtered_credit_status_claim(
                        filtered_scope,
                        grounded_list.status,
                    )
                elif not grounded_list.selected_targets:
                    claim = _filtered_credit_status_claim(
                        filtered_scope,
                        "insufficient_evidence",
                    )
                else:
                    filtered_plan = _filtered_credit_plan(
                        spec,
                        resolution,
                        grounded_list.selected_targets,
                    )
                    credit_bundle = execute_evidence_plan(
                        db_path,
                        filtered_plan,
                    )
                    claims = _compose_evidence_claims(spec, credit_bundle)
                    if not claims or any(
                        claim.operation != "sum_credits" for claim in claims
                    ):
                        claim = _filtered_credit_status_claim(
                            filtered_scope,
                            "insufficient_evidence",
                        )
                    else:
                        grounded = compose_grounded_answer(
                            composed_claims=claims,
                        )
                        return {
                            "route": None,
                            "result": render_grounded_answer(
                                grounded,
                                answer_model_callable=None,
                                question=question,
                            ),
                        }
        grounded = compose_grounded_answer(composed_claims=(claim,))
        return {
            "route": None,
            "result": render_grounded_answer(
                grounded,
                answer_model_callable=None,
                question=question,
            ),
        }
    if (
        completeness.classification in {"partial", "unrecognized_structured"}
        and callable(structured_model_callable)
        and _is_course_list_fallback_candidate(spec, completeness)
    ):
        fallback_operation = _course_list_fallback_operation(spec)
        fallback_scope = _fallback_scope(completeness, resolution)
        if fallback_scope is not None and fallback_operation is not None:
            fallback_result = run_structured_fallback(
                db_path,
                question,
                fallback_scope,
                structured_model_callable,
            )
            grounded_list = ground_course_list(
                db_path,
                fallback_result,
                fallback_scope,
            )
            claim = _fallback_course_list_claim(
                grounded_list,
                fallback_scope,
                operation=fallback_operation,
            )
            grounded = compose_grounded_answer(composed_claims=(claim,))
            return {
                "route": None,
                "result": render_grounded_answer(
                    grounded,
                    answer_model_callable=None,
                    question=question,
                ),
            }

    if (
        completeness.classification in {"partial", "unrecognized_structured"}
        and callable(structured_model_callable)
        and _is_placement_fallback_candidate(spec, completeness)
    ):
        fallback_scope = _fallback_scope(
            completeness,
            resolution,
            placement_code_identity=True,
        )
        if fallback_scope is not None:
            fallback_result = run_structured_fallback(
                db_path,
                question,
                fallback_scope,
                structured_model_callable,
                selector_mode="placement",
            )
            grounded_placement = ground_placement(
                db_path,
                fallback_result,
                fallback_scope,
            )
            claim = _fallback_placement_claim(
                grounded_placement,
                fallback_scope,
            )
            grounded = compose_grounded_answer(composed_claims=(claim,))
            return {
                "route": None,
                "result": render_grounded_answer(
                    grounded,
                    answer_model_callable=None,
                    question=question,
                ),
            }

    if (
        completeness.classification in {"partial", "unrecognized_structured"}
        and callable(structured_model_callable)
        and _is_course_credit_fallback_candidate(spec, completeness, resolution)
    ):
        fallback_scope = _fallback_scope(
            completeness,
            resolution,
            placement_code_identity=True,
        )
        if fallback_scope is not None:
            fallback_result = run_structured_fallback(
                db_path,
                question,
                fallback_scope,
                structured_model_callable,
                selector_mode="course_credit",
            )
            grounded_credit = ground_course_credit(
                db_path,
                fallback_result,
                fallback_scope,
            )
            claim = _fallback_course_credit_claim(
                grounded_credit,
                fallback_scope,
            )
            grounded = compose_grounded_answer(composed_claims=(claim,))
            return {
                "route": None,
                "result": render_grounded_answer(
                    grounded,
                    answer_model_callable=None,
                    question=question,
                ),
            }

    intent_interpreted = False
    if _should_use_intent_interpreter(
        spec,
        completeness,
        resolution,
        context,
    ):
        if not callable(intent_model_callable):
            return _intent_failure_result(question)

        try:
            interpretation = interpret_question_intent(
                question,
                intent_model_callable,
            )
        except Exception:
            # The interpreter is an optional proposal boundary.  A model or
            # validation failure must remain fail closed.
            return _intent_failure_result(question)
        try:
            compiled_spec = compile_intent_to_query_spec(
                spec,
                interpretation,
                **_intent_authoritative_scope(spec, context),
            )
        except (TypeError, ValueError):
            return _intent_failure_result(question)

        try:
            compiled_resolution = resolve_query_spec(
                compiled_spec,
                db_path,
                context=context,
            )
        except (FileNotFoundError, OSError, TypeError, ValueError, KeyError):
            return _intent_failure_result(question)

        if compiled_resolution.action != "answer":
            if compiled_resolution.action in {
                "clarify_program",
                "context_conflict",
                "no_data",
                "unsupported",
            }:
                return {"route": None, "result": _blocked_result(compiled_resolution)}
            return _intent_failure_result(question)

        compiled_completeness = _classify_structured_parse_completeness(
            compiled_spec,
            compiled_resolution,
            context,
        )
        if compiled_completeness.classification in {
            "partial",
            "unrecognized_structured",
        }:
            return _intent_failure_result(question)

        # Do not let an interpreted proposal re-enter any approved SQL seam.
        # It proceeds only through the existing deterministic evidence path.
        spec = compiled_spec
        resolution = compiled_resolution
        completeness = compiled_completeness
        intent_interpreted = True

    if completeness.classification in {"partial", "unrecognized_structured"}:
        grounded = compose_grounded_answer(
            resolution_status="insufficient_evidence",
        )
        return {
            "route": None,
            "result": render_grounded_answer(
                grounded,
                answer_model_callable=None,
                question=question,
            ),
        }

    try:
        plan = plan_evidence(spec, resolution)
        bundle = execute_evidence_plan(db_path, plan)
    except (FileNotFoundError, OSError, TypeError, ValueError, KeyError):
        grounded = compose_grounded_answer(
            resolution_status="insufficient_evidence",
        )
        return {
            "route": None,
            "result": render_grounded_answer(grounded),
        }

    similarity_evidence: SimilarityEvidence | None = None
    similarity_request_ids: tuple[str, str] | None = None
    selected_plan: str | None = None
    if "similarity" in spec.operations:
        similarity_request_ids = _select_similarity_request_ids(plan, resolution)
        if similarity_request_ids is not None:
            resolved_plans = tuple(resolution.resolved_plans)
            selected_plan = resolved_plans[0] if len(resolved_plans) == 1 else None
            try:
                similarity_evidence = execute_exact_similarity_from_bundle(
                    db_path,
                    bundle,
                    similarity_request_ids[0],
                    similarity_request_ids[1],
                    selected_plan=selected_plan,
                )
            except (FileNotFoundError, OSError, TypeError, ValueError, KeyError):
                similarity_evidence = None

    claims = _compose_evidence_claims(
        spec,
        bundle,
        similarity_evidence=similarity_evidence,
        similarity_request_ids=similarity_request_ids,
        selected_plan=selected_plan,
    )
    if "identity" in spec.operations:
        identity_claims = compose_grounded_answer(
            identity_result=_identity_result(resolution),
        ).claims
        claims = (*identity_claims, *claims)
    grounded = compose_grounded_answer(composed_claims=claims)
    return {
        "route": None,
        "result": render_grounded_answer(
            grounded,
            answer_model_callable=(
                None if intent_interpreted else answer_model_callable
            ),
            question=question,
        ),
    }


__all__ = ["ask"]
