"""Deterministic conversation-context merge for semantic mode.

Precedence: explicit valid scope in the current turn > validated
conversation context > unknown. The LLM never decides precedence and its
output never mutates context directly: only validated scope fields flow in,
and explicitly changed program/catalog invalidates stale dependent
referents (focus course, retained result sets). Free-form prose from prior
turns is never copied as factual authority.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rag.semantic.schema import SemanticIntent


@dataclass(frozen=True, slots=True)
class MergedContext:
    program: str | None = None
    catalog_key: str | None = None
    plan: str | None = None
    plan_hint: str | None = None
    years: tuple[int, ...] = ()
    semesters: tuple[int, ...] = ()
    focus_course: dict[str, Any] | None = None
    result_courses: tuple[dict[str, Any], ...] = ()
    result_scope_program: str | None = None
    result_set_empty: bool = False
    invalidated: tuple[str, ...] = ()
    valid: bool = True
    reason: str | None = None


def _clean_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _clean_course_entry(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    code = _clean_text(value.get("course_code"))
    if code is None:
        return None
    entry: dict[str, Any] = {"course_code": code}
    for key in ("program", "catalog_key"):
        text = _clean_text(value.get(key))
        if text is not None:
            entry[key] = text
    return entry


def _validated_prior_context(
    conversation_context: dict[str, Any] | None,
) -> dict[str, Any]:
    """Extract only structurally valid carry-over fields from client context."""
    prior: dict[str, Any] = {}
    if not isinstance(conversation_context, dict):
        return prior
    for key in ("program", "catalog_key", "plan"):
        text = _clean_text(conversation_context.get(key))
        if text is not None:
            if key == "catalog_key" and len(text) > 128:
                continue
            prior[key] = text
    for key in ("years", "semesters"):
        raw = conversation_context.get(key)
        if isinstance(raw, (list, tuple)) and all(
            isinstance(item, int) and not isinstance(item, bool) for item in raw
        ):
            lo, hi = (1, 5) if key == "years" else (1, 2)
            values = tuple(item for item in raw if lo <= item <= hi)
            if values:
                prior[key] = values
    focus = _clean_course_entry(conversation_context.get("focus_course"))
    if focus is not None:
        prior["focus_course"] = focus
    raw_results = conversation_context.get("result_courses")
    if isinstance(raw_results, (list, tuple)):
        entries = tuple(
            entry
            for item in raw_results
            if (entry := _clean_course_entry(item)) is not None
        )
        if entries:
            prior["result_courses"] = entries[:50]
    scope_program = _clean_text(conversation_context.get("result_scope_program"))
    if scope_program is not None:
        prior["result_scope_program"] = scope_program
    if conversation_context.get("result_set_empty") is True:
        prior["result_set_empty"] = True
    operation = conversation_context.get("last_normal_operation")
    keys = {"kind", "program", "catalog_key", "plan", "years", "semesters"}
    if (
        isinstance(operation, dict)
        and set(operation) == keys
        and operation.get("kind") == "list_courses"
        and isinstance(operation.get("program"), str)
        and operation["program"] == prior.get("program")
        and operation.get("catalog_key") == prior.get("catalog_key")
        and operation.get("plan") == prior.get("plan")
        and isinstance(operation.get("years"), (list, tuple))
        and isinstance(operation.get("semesters"), (list, tuple))
        and all(type(value) is int for value in (*operation["years"], *operation["semesters"]))
        and tuple(operation["years"]) == prior.get("years", ())
        and tuple(operation["semesters"]) == prior.get("semesters", ())
        and (operation["years"] or operation["semesters"])
    ):
        prior["last_normal_operation"] = dict(operation)
    return prior


def merge_semantic_context(
    intent: SemanticIntent,
    conversation_context: dict[str, Any] | None,
) -> MergedContext:
    """Merge explicit turn scope over validated prior context deterministically."""
    if not isinstance(intent, SemanticIntent):
        return MergedContext(valid=False, reason="intent is not a SemanticIntent")
    prior = _validated_prior_context(conversation_context)
    scope = intent.scope

    program = scope.program.strip() if scope.program else prior.get("program")
    catalog_key = scope.catalog.strip() if scope.catalog else prior.get("catalog_key")
    plan = scope.plan.strip() if scope.plan else prior.get("plan")
    # A normalized plan hint is a candidate only: it never becomes
    # authoritative scope here and never invalidates retained context.
    plan_hint = scope.plan_hint.strip() if scope.plan_hint else None
    years = (scope.year,) if scope.year is not None else prior.get("years", ())
    semesters = (
        (scope.semester,) if scope.semester is not None else prior.get("semesters", ())
    )

    invalidated: list[str] = []
    focus_course = prior.get("focus_course")
    result_courses = prior.get("result_courses", ())
    result_scope_program = prior.get("result_scope_program")
    result_set_empty = bool(prior.get("result_set_empty", False))

    prior_program = prior.get("program")
    if (
        scope.program is not None
        and program is not None
        and prior_program is not None
        and program.casefold() != prior_program.casefold()
    ):
        # Only current-turn scope may cross a program boundary. Preserve
        # explicit replacements, never dependent defaults from the old program.
        for key, explicit in (
            ("catalog_key", scope.catalog),
            ("plan", scope.plan),
            ("years", scope.year),
            ("semesters", scope.semester),
        ):
            if explicit is None and key in prior:
                invalidated.append(key)
        catalog_key = scope.catalog.strip() if scope.catalog else None
        plan = scope.plan.strip() if scope.plan else None
        years = (scope.year,) if scope.year is not None else ()
        semesters = (scope.semester,) if scope.semester is not None else ()
        if result_scope_program is not None:
            invalidated.append("result_scope_program")
        invalidated.extend(["focus_course", "result_courses", "result_set_empty"])
        focus_course = None
        result_courses = ()
        result_set_empty = False
        result_scope_program = None
    prior_catalog = prior.get("catalog_key")
    if (
        catalog_key is not None
        and prior_catalog is not None
        and catalog_key.casefold() != prior_catalog.casefold()
    ):
        for key in ("focus_course", "result_courses", "result_set_empty"):
            if key not in invalidated:
                invalidated.append(key)
        focus_course = None
        result_courses = ()
        result_set_empty = False
    prior_plan = prior.get("plan")
    if (
        plan is not None
        and prior_plan is not None
        and plan.casefold() != prior_plan.casefold()
    ):
        # An explicit plan switch re-scopes every retained identity the same
        # way a program/edition switch does: stale dependents fail closed.
        for key in ("focus_course", "result_courses", "result_set_empty"):
            if key not in invalidated:
                invalidated.append(key)
        focus_course = None
        result_courses = ()
        result_set_empty = False

    return MergedContext(
        program=program,
        catalog_key=catalog_key,
        plan=plan,
        plan_hint=plan_hint,
        years=tuple(years),
        semesters=tuple(semesters),
        focus_course=focus_course,
        result_courses=tuple(result_courses),
        result_scope_program=result_scope_program,
        result_set_empty=result_set_empty,
        invalidated=tuple(invalidated),
    )


__all__ = ["MergedContext", "merge_semantic_context"]
