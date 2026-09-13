"""Deterministic QuerySpec resolution and ordered policy guards."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from rag.query_spec import QuerySpec
from rag.structured.queries import Database, exact_course_candidates


_PROGRAM_BLOCKING_AMBIGUITY = ("program",)


@dataclass(frozen=True, slots=True)
class CourseReferenceResolution:
    """Candidates returned for one explicit course code or name reference."""

    reference_type: str
    reference: str
    candidates: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True, slots=True)
class ResolutionOutcome:
    """Resolution state passed to later evidence planning."""

    action: str
    blocking_ambiguity: tuple[str, ...]
    resolved_program: str | None
    course_references: tuple[CourseReferenceResolution, ...]


def _requires_program_scope(spec: QuerySpec) -> bool:
    return bool(
        spec.plans
        or spec.years
        or spec.semesters
        or spec.category
        or spec.topic
        or spec.operations
        or spec.group_by
    )


def _explicit_references(spec: QuerySpec) -> tuple[tuple[str, str], ...]:
    references = [("course_code", code) for code in spec.course_codes]
    if spec.course_name is not None:
        references.append(("course_name", spec.course_name))
    return tuple(references)


def _outcome(
    action: str,
    *,
    resolved_program: str | None,
    course_references: tuple[CourseReferenceResolution, ...] = (),
    blocking_ambiguity: tuple[str, ...] = (),
) -> ResolutionOutcome:
    return ResolutionOutcome(
        action=action,
        blocking_ambiguity=blocking_ambiguity,
        resolved_program=resolved_program,
        course_references=course_references,
    )


def _candidate_programs(
    reference: CourseReferenceResolution,
) -> set[str]:
    return {
        str(candidate["program"])
        for candidate in reference.candidates
        if candidate.get("program") is not None
    }


def resolve_query_spec(spec: QuerySpec, db_path: Database) -> ResolutionOutcome:
    """Resolve exact entities and apply the Phase 3B guard order.

    The resolver only uses the relational exact-candidate primitive. It does
    not retrieve evidence, call a model, or decide any later answer content.
    """
    if spec.judgement == "unsupported":
        return _outcome("unsupported", resolved_program=spec.program)

    references: list[CourseReferenceResolution] = []
    for reference_type, reference in _explicit_references(spec):
        candidates = exact_course_candidates(
            db_path,
            course_code=reference if reference_type == "course_code" else None,
            course_name=reference if reference_type == "course_name" else None,
            program=spec.program,
        )
        references.append(
            CourseReferenceResolution(
                reference_type=reference_type,
                reference=reference,
                candidates=tuple(candidates),
            )
        )
    resolved_references = tuple(references)

    if any(not reference.candidates for reference in resolved_references):
        return _outcome(
            "no_data",
            resolved_program=spec.program,
            course_references=resolved_references,
        )

    resolved_program = spec.program
    if spec.program is None and resolved_references:
        programs = set().union(
            *(_candidate_programs(reference) for reference in resolved_references)
        )
        if len(programs) > 1:
            return _outcome(
                "clarify_program",
                resolved_program=None,
                course_references=resolved_references,
                blocking_ambiguity=_PROGRAM_BLOCKING_AMBIGUITY,
            )

        all_references_unique = all(
            len(reference.candidates) == 1
            for reference in resolved_references
        )
        if all_references_unique and len(programs) == 1:
            resolved_program = next(iter(programs))

    if (
        spec.program is None
        and resolved_program is None
        and _requires_program_scope(spec)
    ):
        return _outcome(
            "clarify_program",
            resolved_program=None,
            course_references=resolved_references,
            blocking_ambiguity=_PROGRAM_BLOCKING_AMBIGUITY,
        )

    return _outcome(
        "answer",
        resolved_program=resolved_program,
        course_references=resolved_references,
    )


__all__ = [
    "CourseReferenceResolution",
    "ResolutionOutcome",
    "resolve_query_spec",
]
