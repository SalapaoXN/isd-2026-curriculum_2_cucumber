"""Immutable typed claim composition for the post-execution answer boundary."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any

from rag.aggregation import (
    ComparisonAggregation,
    ComponentAggregation,
    CourseSetAggregation,
    EarliestAggregation,
    PlanComparisonAggregation,
)
from rag.evidence_executor import EvidenceBundle, EvidenceExecutionResult
from rag.evidence_planner import StructuralScope
from rag.judgement import JudgementEvidence
from rag.retrieval.retrieve import SimilarityEvidence
from rag.resolution import ResolutionOutcome


ANSWER_STATUSES = (
    "answer",
    "no_data",
    "valid_empty",
    "insufficient_evidence",
    "unsupported",
    "clarify_program",
    "context_conflict",
)
ANSWER_MODES = ("deterministic", "grounded_synthesis", "mixed")
CLAIM_KINDS = ("deterministic_fact", "grounded_summary")
CLAIM_STATUSES = (
    "complete",
    "valid_empty",
    "insufficient_evidence",
    "descriptive_only",
)
CLAIM_OPERATIONS = frozenset(
    {
        "identity",
        "program_discovery",
        "list",
        "count",
        "course_set",
        "existence",
        "sum_credits",
        "placement",
        "prerequisite",
        "earliest",
        "compare",
        "describe",
        "similarity",
        "quantity",
        "workload",
        "preference",
        "topic_matches",
        "description_evidence",
    }
)
EMPTY_ANSWER = "ไม่พบข้อมูลนี้ในเล่มหลักสูตร"


class _InvalidComposition(ValueError):
    """Expected malformed composition input."""


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    return value


def _provenance_container(value: Any) -> tuple[Any, ...] | None:
    if value is None:
        return ()
    if isinstance(value, (Mapping, str)):
        return (_freeze(value),)
    if not isinstance(value, (list, tuple)):
        return None
    if any(not isinstance(item, (Mapping, str)) for item in value):
        return None
    return tuple(_freeze(item) for item in value)


def _provenance_from(value: Any) -> tuple[Any, ...] | None:
    """Collect only provenance already attached to typed evidence."""
    if isinstance(value, Mapping):
        if "provenance" in value:
            direct = _provenance_container(value.get("provenance"))
            if direct is None:
                return None
        else:
            direct = ()
        nested: list[Any] = list(direct)
        for item in value.values():
            if item is value.get("provenance"):
                continue
            found = _provenance_from(item)
            if found is None:
                return None
            nested.extend(found)
        return tuple(nested)
    if isinstance(value, (list, tuple)):
        nested: list[Any] = []
        for item in value:
            found = _provenance_from(item)
            if found is None:
                return None
            nested.extend(found)
        return tuple(nested)
    if isinstance(value, CourseSetAggregation):
        return _provenance_from(value.courses)
    if isinstance(value, ComponentAggregation):
        return _provenance_from(value.components)
    if isinstance(value, ComparisonAggregation):
        left = _provenance_from(value.left)
        right = _provenance_from(value.right)
        if left is None or right is None:
            return None
        return left + right
    if isinstance(value, PlanComparisonAggregation):
        return _provenance_container(value.provenance)
    if isinstance(value, EarliestAggregation):
        nested: list[Any] = []
        for partition in value.partitions:
            found = _provenance_from(partition.provenance)
            if found is None:
                return None
            nested.extend(found)
            found = _provenance_from(partition.placements)
            if found is None:
                return None
            nested.extend(found)
        return tuple(nested)
    if isinstance(value, JudgementEvidence):
        found = _provenance_from(value.provenance)
        if found is None:
            return None
        return found
    if isinstance(value, SimilarityEvidence):
        nested: list[Any] = []
        for pair in value.pairs:
            for evidence in (pair.left, pair.right):
                found = _provenance_from(evidence)
                if found is None:
                    return None
                nested.extend(found)
        return tuple(nested)
    return ()


def _stable_union(items: Sequence[Any]) -> tuple[Any, ...]:
    result: list[Any] = []
    for item in items:
        if not any(item == existing for existing in result):
            result.append(item)
    return tuple(result)


def _validate_numeric_distance(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


@dataclass(frozen=True, slots=True)
class GroundedClaim:
    claim_id: str
    operation: str
    effective_scope: StructuralScope | None = None
    status: str = "complete"
    kind: str = "deterministic_fact"
    value: Any = None
    evidence: Any = None
    provenance: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.claim_id, str) or not self.claim_id:
            raise ValueError("claim_id must be a non-empty string")
        if self.operation not in CLAIM_OPERATIONS:
            raise ValueError(f"unsupported claim operation: {self.operation!r}")
        if self.status not in CLAIM_STATUSES:
            raise ValueError(f"unsupported claim status: {self.status!r}")
        if self.kind not in CLAIM_KINDS:
            raise ValueError(f"unsupported claim kind: {self.kind!r}")
        provenance = _provenance_container(self.provenance)
        if provenance is None:
            raise ValueError("provenance must be an immutable sequence of references")
        object.__setattr__(self, "effective_scope", _freeze(self.effective_scope))
        object.__setattr__(self, "value", _freeze(self.value))
        object.__setattr__(self, "evidence", _freeze(self.evidence))
        object.__setattr__(self, "provenance", provenance)


@dataclass(frozen=True, slots=True)
class GroundedAnswerResult:
    status: str
    answer_mode: str
    final_answer: str = ""
    claims: tuple[GroundedClaim, ...] = ()
    provenance: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in ANSWER_STATUSES:
            raise ValueError(f"unsupported answer status: {self.status!r}")
        if self.answer_mode not in ANSWER_MODES:
            raise ValueError(f"unsupported answer mode: {self.answer_mode!r}")
        if not isinstance(self.final_answer, str):
            raise ValueError("final_answer must be a string")
        claims = tuple(self.claims)
        if any(not isinstance(claim, GroundedClaim) for claim in claims):
            raise ValueError("claims must contain GroundedClaim values")
        provenance = _provenance_container(self.provenance)
        if provenance is None:
            raise ValueError("provenance must be an immutable sequence of references")
        object.__setattr__(self, "claims", claims)
        object.__setattr__(self, "provenance", provenance)


def _operation_for(result: EvidenceExecutionResult, overrides: Mapping[str, str]) -> str:
    operation = overrides.get(result.request_id)
    payload = result.payload
    if operation is None and isinstance(payload, ComponentAggregation):
        operation = payload.operation
    if operation is None and isinstance(payload, ComparisonAggregation):
        operation = "compare"
    if operation is None and isinstance(payload, EarliestAggregation):
        operation = "earliest"
    if operation is None and isinstance(payload, SimilarityEvidence):
        operation = "similarity"
    if operation is None:
        operation = result.kind
    if operation not in CLAIM_OPERATIONS:
        raise _InvalidComposition(f"cannot identify claim operation: {operation!r}")
    return operation


def _claim_kind(operation: str, kind: str) -> str:
    if operation in {"describe", "preference", "topic_matches", "description_evidence"}:
        return "grounded_summary"
    if kind in {"topic_matches", "description_evidence"}:
        return "grounded_summary"
    return "deterministic_fact"


def _claim_from_execution(
    result: EvidenceExecutionResult,
    overrides: Mapping[str, str],
) -> GroundedClaim:
    if not isinstance(result, EvidenceExecutionResult):
        raise _InvalidComposition("results must contain EvidenceExecutionResult values")
    if result.status == "complete" and result.payload is None:
        raise _InvalidComposition("complete result cannot have a null payload")
    operation = _operation_for(result, overrides)
    provenance = _provenance_from(result.payload)
    if provenance is None:
        raise _InvalidComposition("malformed provenance")
    if result.status == "complete" and result.planned_request.provenance_required and not provenance:
        raise _InvalidComposition("complete result lacks required provenance")
    return GroundedClaim(
        claim_id="pending",
        operation=operation,
        effective_scope=result.effective_scope,
        status=result.status,
        kind=_claim_kind(operation, result.kind),
        value=result.payload,
        evidence=result.payload,
        provenance=provenance,
    )


def _identity_claim(identity_result: Mapping[str, Any]) -> GroundedClaim:
    if not isinstance(identity_result, Mapping):
        raise _InvalidComposition("identity result must be a mapping")
    if identity_result.get("status") != "answer":
        raise _InvalidComposition("identity result is not answerable")
    identities = identity_result.get("identities")
    if not isinstance(identities, (list, tuple)):
        raise _InvalidComposition("identity result identities must be a sequence")
    for identity in identities:
        if not isinstance(identity, Mapping):
            raise _InvalidComposition("identity evidence must contain mappings")
        if not identity.get("program") or not identity.get("course_code"):
            raise _InvalidComposition("identity evidence lacks logical identity")
    provenance = _provenance_from(identity_result)
    if provenance is None:
        raise _InvalidComposition("malformed identity provenance")
    operation = identity_result.get("operation", "identity")
    if operation not in CLAIM_OPERATIONS:
        raise _InvalidComposition("identity result has an unsupported operation")
    return GroundedClaim(
        claim_id="pending",
        operation=operation,
        status="complete",
        kind="deterministic_fact",
        value=tuple(identities),
        evidence=identity_result,
        provenance=provenance,
    )


def _external_claim(value: Any, operation: str) -> GroundedClaim:
    if operation == "similarity" and not isinstance(value, SimilarityEvidence):
        raise _InvalidComposition("similarity evidence has an unsupported type")
    if operation in {"quantity", "workload", "preference"} and not isinstance(
        value, JudgementEvidence
    ):
        raise _InvalidComposition("judgement evidence has an unsupported type")
    provenance = _provenance_from(value)
    if provenance is None:
        raise _InvalidComposition("malformed external provenance")
    if isinstance(value, JudgementEvidence):
        status = (
            value.status
            if value.status in {"descriptive_only", "insufficient_evidence"}
            else "complete"
        )
        kind = "grounded_summary" if operation == "preference" else "deterministic_fact"
    else:
        status = value.status
        kind = "deterministic_fact"
    return GroundedClaim(
        claim_id="pending",
        operation=operation,
        status=status,
        kind=kind,
        value=value,
        evidence=value,
        provenance=provenance,
    )


def _typed_sequence(value: Any, expected_type: type[Any]) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, expected_type):
        return (value,)
    if isinstance(value, (str, bytes, Mapping)):
        raise _InvalidComposition("typed evidence must be a value or sequence")
    try:
        values = tuple(value)
    except TypeError as error:
        raise _InvalidComposition("typed evidence must be a value or sequence") from error
    if any(not isinstance(item, expected_type) for item in values):
        raise _InvalidComposition("typed evidence sequence contains an invalid value")
    return values


def _number_claims(claims: Sequence[GroundedClaim]) -> tuple[GroundedClaim, ...]:
    return tuple(
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
        for index, claim in enumerate(claims, start=1)
    )


def _answer_mode(claims: Sequence[GroundedClaim]) -> str:
    kinds = {claim.kind for claim in claims if claim.status != "insufficient_evidence"}
    if kinds == {"grounded_summary"}:
        return "grounded_synthesis"
    if kinds == {"deterministic_fact"} or not kinds:
        return "deterministic"
    return "mixed"


def _status_for_claims(claims: Sequence[GroundedClaim]) -> str:
    if not claims:
        return "insufficient_evidence"
    statuses = [claim.status for claim in claims]
    if any(status == "insufficient_evidence" for status in statuses):
        return "insufficient_evidence"
    if any(status == "complete" or status == "descriptive_only" for status in statuses):
        return "answer"
    return "valid_empty"


def _blocked_result(status: str) -> GroundedAnswerResult:
    final_answer = EMPTY_ANSWER if status == "no_data" else ""
    return GroundedAnswerResult(status, "deterministic", final_answer)


def _build_claims(
    bundle: EvidenceBundle | None,
    *,
    operation_by_request: Mapping[str, str] | None,
    identity_result: Mapping[str, Any] | None,
    judgement_evidence: Sequence[JudgementEvidence] | None,
    similarity_evidence: Sequence[SimilarityEvidence] | None,
) -> tuple[GroundedClaim, ...]:
    if bundle is None and not any((identity_result, judgement_evidence, similarity_evidence)):
        raise _InvalidComposition("at least one typed evidence source is required")
    overrides = operation_by_request or {}
    if not isinstance(overrides, Mapping) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in overrides.items()
    ):
        raise _InvalidComposition("operation overrides must be a string mapping")
    claims: list[GroundedClaim] = []
    if bundle is not None:
        if not isinstance(bundle, EvidenceBundle):
            raise _InvalidComposition("bundle must be an EvidenceBundle")
        request_ids = {result.request_id for result in bundle.results}
        if any(request_id not in request_ids for request_id in overrides):
            raise _InvalidComposition("operation override references an unknown request")
        claims.extend(_claim_from_execution(result, overrides) for result in bundle.results)
    if identity_result is not None:
        claims.append(_identity_claim(identity_result))
    for evidence in _typed_sequence(judgement_evidence, JudgementEvidence):
        claims.append(_external_claim(evidence, evidence.judgement))
    for evidence in _typed_sequence(similarity_evidence, SimilarityEvidence):
        claims.append(_external_claim(evidence, "similarity"))
    return _number_claims(claims)


def _validate_composed_claims(
    composed_claims: Sequence[GroundedClaim],
) -> tuple[GroundedClaim, ...]:
    if isinstance(composed_claims, (str, bytes, Mapping, set, frozenset)):
        raise _InvalidComposition("composed_claims must be an ordered sequence")
    try:
        claims = tuple(composed_claims)
    except TypeError as error:
        raise _InvalidComposition("composed_claims must be an ordered sequence") from error
    for claim in claims:
        if not isinstance(claim, GroundedClaim):
            raise _InvalidComposition("composed_claims must contain GroundedClaim values")
        if claim.operation not in CLAIM_OPERATIONS or claim.status not in CLAIM_STATUSES:
            raise _InvalidComposition("composed claim has an invalid semantic field")
        if claim.kind not in CLAIM_KINDS:
            raise _InvalidComposition("composed claim has an invalid kind")
        if _provenance_container(claim.provenance) is None:
            raise _InvalidComposition("composed claim has malformed provenance")
    return claims


def compose_grounded_claims(
    bundle: EvidenceBundle | None = None,
    *,
    operation_by_request: Mapping[str, str] | None = None,
    identity_result: Mapping[str, Any] | None = None,
    judgement_evidence: Sequence[JudgementEvidence] | None = None,
    similarity_evidence: Sequence[SimilarityEvidence] | None = None,
) -> tuple[GroundedClaim, ...]:
    """Compose ordered typed claims, returning no claims for malformed input."""
    try:
        return _build_claims(
            bundle,
            operation_by_request=operation_by_request,
            identity_result=identity_result,
            judgement_evidence=judgement_evidence,
            similarity_evidence=similarity_evidence,
        )
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
        return ()


def compose_grounded_answer(
    bundle: EvidenceBundle | None = None,
    *,
    composed_claims: Sequence[GroundedClaim] | None = None,
    operation_by_request: Mapping[str, str] | None = None,
    identity_result: Mapping[str, Any] | None = None,
    judgement_evidence: Sequence[JudgementEvidence] | None = None,
    similarity_evidence: Sequence[SimilarityEvidence] | None = None,
    resolution: ResolutionOutcome | None = None,
    resolution_status: str | None = None,
) -> GroundedAnswerResult:
    """Compose a fail-closed immutable result without rendering or model calls."""
    if resolution is not None:
        if not isinstance(resolution, ResolutionOutcome):
            return _blocked_result("insufficient_evidence")
        resolution_status = resolution.action
    if resolution_status is not None:
        if resolution_status not in ANSWER_STATUSES:
            return _blocked_result("insufficient_evidence")
        if resolution_status != "answer":
            return _blocked_result(resolution_status)
    try:
        if composed_claims is not None:
            if any(
                value is not None
                for value in (
                    bundle,
                    operation_by_request,
                    identity_result,
                    judgement_evidence,
                    similarity_evidence,
                )
            ):
                raise _InvalidComposition(
                    "composed_claims cannot be combined with raw evidence sources"
                )
            claims = _validate_composed_claims(composed_claims)
        else:
            claims = _build_claims(
                bundle,
                operation_by_request=operation_by_request,
                identity_result=identity_result,
                judgement_evidence=judgement_evidence,
                similarity_evidence=similarity_evidence,
            )
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
        return _blocked_result("insufficient_evidence")
    status = _status_for_claims(claims)
    provenance = _stable_union(
        reference for claim in claims for reference in claim.provenance
    )
    return GroundedAnswerResult(status, _answer_mode(claims), "", claims, provenance)


__all__ = [
    "GroundedAnswerResult",
    "GroundedClaim",
    "compose_grounded_answer",
    "compose_grounded_claims",
]
