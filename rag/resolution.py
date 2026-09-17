"""Deterministic QuerySpec resolution and ordered policy guards."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping

from rag.query_spec import QuerySpec
from rag.structured.queries import Database, exact_course_candidates


_PROGRAM_BLOCKING_AMBIGUITY = ("program",)
_PROGRAM_REFERENCE_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])(?P<program>ait|bit|dsba|gened|it)"
    r"(?![A-Za-z0-9_])(?:\s+วิชา)?\s*(?P<course_code>\d{8})(?!\d)",
    re.IGNORECASE,
)
_CODE_FIRST_PROGRAM_REFERENCE_PATTERN = re.compile(
    r"(?<!\d)(?P<course_code>\d{8})(?!\d)\s*ของ\s*"
    r"(?P<program>ait|bit|dsba|gened|it)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_PROGRAM_ALIASES = {
    "ait": "AIT",
    "bit": "BIT",
    "dsba": "DSBA",
    "gened": "GENED",
    "it": "IT",
}


@dataclass(frozen=True, slots=True)
class QueryContext:
    """Immutable UI-provided scope, separate from question-derived entities."""

    program: str | None = None
    plan: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("program", "plan"):
            value = getattr(self, field_name)
            if value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                raise ValueError(f"{field_name} must be None or a non-empty string")


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
    resolved_plans: tuple[str, ...] = ()
    context_conflicts: tuple[str, ...] = ()


def _requires_program_scope(spec: QuerySpec, plans: tuple[str, ...]) -> bool:
    return bool(
        plans
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


def _explicit_reference_programs(question: str) -> dict[str, str]:
    programs: dict[str, str] = {}
    for pattern in (
        _PROGRAM_REFERENCE_PATTERN,
        _CODE_FIRST_PROGRAM_REFERENCE_PATTERN,
    ):
        for match in pattern.finditer(question):
            programs[match.group("course_code")] = _PROGRAM_ALIASES[
                match.group("program").casefold()
            ]
    return programs


def _outcome(
    action: str,
    *,
    resolved_program: str | None,
    course_references: tuple[CourseReferenceResolution, ...] = (),
    resolved_plans: tuple[str, ...] = (),
    blocking_ambiguity: tuple[str, ...] = (),
    context_conflicts: tuple[str, ...] = (),
) -> ResolutionOutcome:
    return ResolutionOutcome(
        action=action,
        blocking_ambiguity=blocking_ambiguity,
        resolved_program=resolved_program,
        course_references=course_references,
        resolved_plans=resolved_plans,
        context_conflicts=context_conflicts,
    )


def _same_scope_value(left: str, right: str) -> bool:
    return left.strip().casefold() == right.strip().casefold()


def _context_conflicts(spec: QuerySpec, context: QueryContext) -> tuple[str, ...]:
    conflicts: list[str] = []
    if (
        context.program is not None
        and spec.program is not None
        and not _same_scope_value(context.program, spec.program)
    ):
        conflicts.append("program")

    if context.plan is not None:
        if len(spec.plans) > 1:
            conflicts.append("plan")
        elif spec.plans and not _same_scope_value(context.plan, spec.plans[0]):
            conflicts.append("plan")
    return tuple(conflicts)


def _candidate_programs(
    reference: CourseReferenceResolution,
) -> set[str]:
    return {
        str(candidate["program"])
        for candidate in reference.candidates
        if candidate.get("program") is not None
    }


def resolve_query_spec(
    spec: QuerySpec,
    db_path: Database,
    context: QueryContext | None = None,
) -> ResolutionOutcome:
    """Resolve exact entities and apply the Phase 3B guard order.

    The resolver only uses the relational exact-candidate primitive. It does
    not retrieve evidence, call a model, or decide any later answer content.
    """
    if context is None:
        context = QueryContext()

    if spec.judgement == "unsupported":
        return _outcome("unsupported", resolved_program=spec.program)

    conflicts = _context_conflicts(spec, context)
    if conflicts:
        return _outcome(
            "context_conflict",
            resolved_program=None,
            context_conflicts=conflicts,
        )

    effective_program = spec.program or context.program
    effective_plans = spec.plans or (
        (context.plan,) if context.plan is not None else ()
    )
    explicit_reference_programs = _explicit_reference_programs(spec.original_question)

    references: list[CourseReferenceResolution] = []
    for reference_type, reference in _explicit_references(spec):
        reference_program = (
            explicit_reference_programs.get(reference)
            or effective_program
        )
        candidates = exact_course_candidates(
            db_path,
            course_code=reference if reference_type == "course_code" else None,
            course_name=reference if reference_type == "course_name" else None,
            program=reference_program,
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
            resolved_program=effective_program,
            course_references=resolved_references,
            resolved_plans=effective_plans,
        )

    resolved_program = effective_program
    if effective_program is None and resolved_references:
        programs = set().union(
            *(_candidate_programs(reference) for reference in resolved_references)
        )
        if len(programs) > 1:
            return _outcome(
                "clarify_program",
                resolved_program=None,
                course_references=resolved_references,
                resolved_plans=effective_plans,
                blocking_ambiguity=_PROGRAM_BLOCKING_AMBIGUITY,
            )

        all_references_unique = all(
            len(reference.candidates) == 1
            for reference in resolved_references
        )
        if all_references_unique and len(programs) == 1:
            resolved_program = next(iter(programs))

    if (
        effective_program is None
        and resolved_program is None
        and _requires_program_scope(spec, effective_plans)
    ):
        return _outcome(
            "clarify_program",
            resolved_program=None,
            course_references=resolved_references,
            resolved_plans=effective_plans,
            blocking_ambiguity=_PROGRAM_BLOCKING_AMBIGUITY,
        )

    return _outcome(
        "answer",
        resolved_program=resolved_program,
        course_references=resolved_references,
        resolved_plans=effective_plans,
    )


__all__ = [
    "CourseReferenceResolution",
    "QueryContext",
    "ResolutionOutcome",
    "resolve_query_spec",
]
