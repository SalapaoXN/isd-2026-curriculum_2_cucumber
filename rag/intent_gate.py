"""Shared validation mechanics for the bounded intent shadow gates.

This module contains no curriculum access and no answer composition.  It only
validates one-shot placement/count proposals against deterministic scope.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import re
from typing import Any

from rag.intent_compiler import compile_intent_to_query_spec
from rag.intent_interpreter import IntentValidationError, interpret_question_intent


_PLACEMENT_FALLBACK_CUE = re.compile(
    r"สามารถลงได้[^?\n]{0,50}(?:ช่วงไหน|ตอนไหน|ปีไหน|เทอมไหน)|"
    r"(?:อยู่ช่วงไหน|เรียนตอนไหน|ลงตอนไหน|ลงเรียนช่วงไหน|ลงเรียนตอนไหน|เรียนปีไหน|เทอมไหน)",
    re.IGNORECASE,
)
_COUNT_SHADOW_CUE = re.compile(
    r"มีรายวิชาทั้งหมดเท่าไหร่|มีรายวิชาทั้งหมดเท่าไร",
    re.IGNORECASE,
)
_PREREQUISITE_SHADOW_CUE = re.compile(
    r"วิชาบังคับก่อน|ต้องเรียนก่อน|prerequisite",
    re.IGNORECASE,
)
_DESCRIPTION_SHADOW_CUE = re.compile(
    r"รายละเอียด(?:ของ)?วิชา|คำอธิบายรายวิชา|course description",
    re.IGNORECASE,
)
_COURSE_CREDIT_SHADOW_CUE = re.compile(
    r"กี่\s*(?:หน่วย(?:กิต)?|เครดิต)|\bcredits?\b", re.IGNORECASE
)
_EXISTENCE_SHADOW_CUE = re.compile(
    r"(?:มี|อยู่|พบ)\s*(?:ไหม|มั้ย|หรือไม่)|\b(?:exists?|available)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class IntentShadowResult:
    """Validated bounded intent proposal and its diagnostic outcome."""

    attempted: bool
    eligible: bool
    family: str = "placement_query"
    status: str = "not_eligible"
    comparison: str = "not_eligible"
    compiled_spec: Any | None = None
    reason: str | None = None
    executed: bool = False


@dataclass(frozen=True, slots=True)
class CountShadowResult:
    """Diagnostic count proposal; it never supplies factual evidence."""

    attempted: bool
    eligible: bool
    family: str = "count_query"
    status: str = "not_eligible"
    comparison: str = "not_eligible"
    compiled_spec: Any | None = None
    reason: str | None = None
    executed: bool = False


def _exact_course_candidate(
    spec: Any,
    completeness: Any,
    resolution: Any,
    cue: re.Pattern[str],
) -> bool:
    """Identify only bounded, unresolved exact-course wording for shadowing."""
    if getattr(resolution, "action", None) != "answer":
        return False
    if completeness.classification not in {"partial", "unrecognized_structured"}:
        return False
    if completeness.missing_filters:
        return False
    if tuple(getattr(spec, "operations", ())) != ():
        return False
    if getattr(spec, "topic", None) is not None:
        return False
    if getattr(spec, "judgement", None) not in {None, "none"}:
        return False
    course_codes = tuple(getattr(spec, "course_codes", ()))
    course_name = getattr(spec, "course_name", None)
    if len(course_codes) > 1:
        return False
    if not course_codes and not (isinstance(course_name, str) and course_name.strip()):
        return False
    if not isinstance(completeness.program, str) or not completeness.program.strip():
        return False
    return bool(cue.search(getattr(spec, "normalized_question", "")))


def _exact_course_shadow_candidate(
    spec: Any,
    completeness: Any,
    resolution: Any,
    family: str,
) -> bool:
    cue = (
        _PREREQUISITE_SHADOW_CUE
        if family == "prerequisite_query"
        else _DESCRIPTION_SHADOW_CUE
        if family == "course_description"
        else _COURSE_CREDIT_SHADOW_CUE
        if family == "course_credit_query"
        else _EXISTENCE_SHADOW_CUE
    )
    return _exact_course_candidate(spec, completeness, resolution, cue)


def _exact_course_shadow_comparison(
    spec: Any,
    compiled_spec: Any,
    context: Any | None,
    operation: str,
) -> str:
    authoritative = authoritative_scope(spec, context)
    if tuple(getattr(compiled_spec, "operations", ())) != (operation,):
        return "conflict"
    if (getattr(compiled_spec, "program", None) or "").casefold() != (
        authoritative["authoritative_program"] or ""
    ).casefold():
        return "conflict"
    for field, scope_field in (
        ("plans", "authoritative_plans"),
        ("years", "authoritative_years"),
        ("semesters", "authoritative_semesters"),
    ):
        if tuple(getattr(compiled_spec, field, ())) != tuple(authoritative[scope_field]):
            return "conflict"
    if tuple(getattr(compiled_spec, "course_codes", ())) != tuple(
        getattr(spec, "course_codes", ())
    ):
        return "conflict"
    if getattr(compiled_spec, "course_name", None) != getattr(spec, "course_name", None):
        return "conflict"
    if getattr(compiled_spec, "topic", None) != getattr(spec, "topic", None):
        return "conflict"
    if getattr(compiled_spec, "judgement", None) != getattr(spec, "judgement", None):
        return "conflict"
    if getattr(compiled_spec, "category", None) != getattr(spec, "category", None):
        return "conflict"
    return "compatible_extension"


def run_exact_course_shadow(
    question: str,
    spec: Any,
    completeness: Any,
    resolution: Any,
    context: Any | None,
    intent_model_callable: Callable[[str], str] | None,
    *,
    family: str,
    interpret_callable: Callable[..., Any] = interpret_question_intent,
    compile_callable: Callable[..., Any] = compile_intent_to_query_spec,
) -> IntentShadowResult:
    """Validate an exact-course proposal diagnostically; never execute it."""
    if family not in {
        "prerequisite_query", "course_description",
        "course_credit_query", "existence_query",
    }:
        raise ValueError("unsupported exact-course shadow family")
    eligible = _exact_course_shadow_candidate(
        spec, completeness, resolution, family
    )
    if not eligible:
        return IntentShadowResult(
            attempted=False, eligible=False, family=family
        )
    if not callable(intent_model_callable):
        return IntentShadowResult(
            attempted=False, eligible=True, family=family, status="unavailable",
            comparison="unavailable", reason="model_unavailable",
        )
    try:
        interpretation = interpret_callable(question, intent_model_callable)
    except IntentValidationError:
        return IntentShadowResult(
            attempted=True, eligible=True, family=family,
            status="invalid_interpretation",
            comparison="invalid_interpretation",
            reason="malformed_or_unsupported_payload",
        )
    except Exception:
        return IntentShadowResult(
            attempted=True, eligible=True, family=family, status="unavailable",
            comparison="unavailable", reason="provider_failure",
        )

    expected_facts = {
        "prerequisite_query": ("prerequisite",),
        "course_description": ("course_description",),
        "course_credit_query": ("course_credit",),
        "existence_query": ("course_list",),
    }[family]
    if (
        interpretation.intent != family
        or tuple(interpretation.requested_facts) != expected_facts
    ):
        return IntentShadowResult(
            attempted=True, eligible=True, family=family,
            status="invalid_interpretation",
            comparison="invalid_interpretation", reason="wrong_intent_or_facts",
        )
    try:
        compiled_spec = compile_callable(
            spec, interpretation, **authoritative_scope(spec, context)
        )
    except (TypeError, ValueError):
        return IntentShadowResult(
            attempted=True, eligible=True, family=family,
            status="invalid_interpretation", comparison="conflict",
            reason="scope_or_compilation_rejected",
        )
    operation = {
        "prerequisite_query": "prerequisite",
        "course_description": "describe",
        "course_credit_query": "sum_credits",
        "existence_query": "existence",
    }[family]
    comparison = _exact_course_shadow_comparison(
        spec, compiled_spec, context, operation
    )
    if comparison == "conflict":
        return IntentShadowResult(
            attempted=True, eligible=True, family=family,
            status="invalid_interpretation", comparison=comparison,
            reason="compiled_scope_or_target_conflict",
        )
    return IntentShadowResult(
        attempted=True, eligible=True, family=family, status="validated",
        comparison=comparison, compiled_spec=compiled_spec,
    )


def authoritative_scope(spec: Any, context: Any | None = None) -> dict[str, Any]:
    """Return only deterministic scope values permitted for validation."""
    plans = list(getattr(spec, "plans", ()))
    context_plan = getattr(context, "plan", None)
    if context_plan and context_plan not in plans:
        plans.append(context_plan)
    return {
        "authoritative_program": getattr(spec, "program", None)
        or getattr(context, "program", None),
        "authoritative_plans": tuple(plans),
        "authoritative_years": tuple(getattr(spec, "years", ()))
        or tuple(getattr(context, "years", ())),
        "authoritative_semesters": tuple(getattr(spec, "semesters", ()))
        or tuple(getattr(context, "semesters", ())),
        "allowed_course_codes": tuple(getattr(spec, "course_codes", ()))
        or ((context.course_code,) if getattr(context, "course_code", None) else ()),
    }


def _placement_candidate(spec: Any, completeness: Any, resolution: Any) -> bool:
    return bool(
        getattr(resolution, "action", None) == "answer"
        and completeness.classification in {"partial", "unrecognized_structured"}
        and not completeness.missing_filters
        and tuple(getattr(spec, "operations", ())) == ()
        and getattr(spec, "topic", None) is None
        and getattr(spec, "judgement", None) != "unsupported"
        and len(tuple(getattr(spec, "course_codes", ()))) == 1
        and _is_placement_candidate(spec, completeness)
    )


def _is_placement_candidate(spec: Any, completeness: Any) -> bool:
    operations = tuple(getattr(spec, "operations", ()))
    if completeness.missing_filters:
        return False
    if operations:
        return operations == ("placement",)
    return bool(
        completeness.classification == "unrecognized_structured"
        and _PLACEMENT_FALLBACK_CUE.search(getattr(spec, "normalized_question", ""))
    )


def _placement_comparison(spec: Any, compiled_spec: Any, context: Any | None) -> str:
    authoritative = authoritative_scope(spec, context)
    if tuple(getattr(compiled_spec, "operations", ())) != ("placement",):
        return "conflict"
    if (getattr(compiled_spec, "program", None) or "").casefold() != (
        authoritative["authoritative_program"] or ""
    ).casefold():
        return "conflict"
    for field, scope_field in (
        ("plans", "authoritative_plans"),
        ("years", "authoritative_years"),
        ("semesters", "authoritative_semesters"),
    ):
        if tuple(getattr(compiled_spec, field, ())) != tuple(authoritative[scope_field]):
            return "conflict"
    if tuple(getattr(compiled_spec, "course_codes", ())) != tuple(
        authoritative["allowed_course_codes"]
    ):
        return "conflict"
    return "compatible_extension"


def run_placement_shadow(
    question: str,
    spec: Any,
    completeness: Any,
    resolution: Any,
    context: Any | None,
    intent_model_callable: Callable[[str], str] | None,
    *,
    interpret_callable: Callable[..., Any] = interpret_question_intent,
    compile_callable: Callable[..., Any] = compile_intent_to_query_spec,
) -> IntentShadowResult:
    """Run the existing one-shot placement proposal gate."""
    eligible = _placement_candidate(spec, completeness, resolution)
    if not eligible:
        return IntentShadowResult(attempted=False, eligible=False)
    if not callable(intent_model_callable):
        return IntentShadowResult(
            attempted=False, eligible=True, status="unavailable",
            comparison="unavailable", reason="model_unavailable",
        )
    try:
        interpretation = interpret_callable(question, intent_model_callable)
    except IntentValidationError:
        return IntentShadowResult(
            attempted=True, eligible=True, status="invalid_interpretation",
            comparison="invalid_interpretation", reason="malformed_or_unsupported_payload",
        )
    except Exception:
        return IntentShadowResult(
            attempted=True, eligible=True, status="unavailable",
            comparison="unavailable", reason="provider_failure",
        )
    if interpretation.intent != "placement_query":
        return IntentShadowResult(
            attempted=True, eligible=True, status="invalid_interpretation",
            comparison="invalid_interpretation", reason="wrong_intent_family",
        )
    try:
        compiled_spec = compile_callable(
            spec, interpretation, **authoritative_scope(spec, context)
        )
    except (TypeError, ValueError):
        return IntentShadowResult(
            attempted=True, eligible=True, status="invalid_interpretation",
            comparison="conflict", reason="scope_or_compilation_rejected",
        )
    comparison = _placement_comparison(spec, compiled_spec, context)
    if comparison == "conflict":
        return IntentShadowResult(
            attempted=True, eligible=True, status="invalid_interpretation",
            comparison=comparison, reason="compiled_scope_conflict",
        )
    return IntentShadowResult(
        attempted=True, eligible=True, status="validated",
        comparison=comparison, compiled_spec=compiled_spec,
    )


def _count_candidate(spec: Any, completeness: Any, resolution: Any) -> bool:
    if getattr(resolution, "action", None) != "answer":
        return False
    if completeness.classification != "unrecognized_structured":
        return False
    if completeness.missing_filters or tuple(getattr(spec, "operations", ())) != ():
        return False
    if getattr(spec, "topic", None) is not None:
        return False
    if getattr(spec, "judgement", None) not in {None, "none"}:
        return False
    if getattr(spec, "course_codes", ()) or getattr(spec, "course_name", None):
        return False
    if not isinstance(completeness.program, str) or not completeness.program.strip():
        return False
    return bool(_COUNT_SHADOW_CUE.search(getattr(spec, "normalized_question", "")))


def count_shadow_comparison(
    spec: Any, compiled_spec: Any, context: Any | None
) -> str:
    authoritative = authoritative_scope(spec, context)
    if tuple(getattr(compiled_spec, "operations", ())) != ("count",):
        return "conflict"
    if (getattr(compiled_spec, "program", None) or "").casefold() != (
        authoritative["authoritative_program"] or ""
    ).casefold():
        return "conflict"
    for field, scope_field in (
        ("plans", "authoritative_plans"),
        ("years", "authoritative_years"),
        ("semesters", "authoritative_semesters"),
    ):
        if tuple(getattr(compiled_spec, field, ())) != tuple(authoritative[scope_field]):
            return "conflict"
    if tuple(getattr(compiled_spec, "course_codes", ())) != tuple(
        authoritative["allowed_course_codes"]
    ):
        return "conflict"
    if getattr(compiled_spec, "category", None) != getattr(spec, "category", None):
        return "conflict"
    return "compatible_extension"


def run_count_shadow(
    question: str,
    spec: Any,
    completeness: Any,
    resolution: Any,
    context: Any | None,
    intent_model_callable: Callable[[str], str] | None,
    *,
    interpret_callable: Callable[..., Any] = interpret_question_intent,
    compile_callable: Callable[..., Any] = compile_intent_to_query_spec,
) -> CountShadowResult:
    """Run the existing one-shot count proposal gate."""
    eligible = _count_candidate(spec, completeness, resolution)
    if not eligible:
        return CountShadowResult(attempted=False, eligible=False)
    if not callable(intent_model_callable):
        return CountShadowResult(
            attempted=False, eligible=True, status="unavailable",
            comparison="unavailable", reason="model_unavailable",
        )
    try:
        interpretation = interpret_callable(question, intent_model_callable)
    except IntentValidationError:
        return CountShadowResult(
            attempted=True, eligible=True, status="invalid_interpretation",
            comparison="invalid_interpretation", reason="malformed_or_unsupported_payload",
        )
    except Exception:
        return CountShadowResult(
            attempted=True, eligible=True, status="unavailable",
            comparison="unavailable", reason="provider_failure",
        )
    if interpretation.intent != "count_query":
        return CountShadowResult(
            attempted=True, eligible=True, status="invalid_interpretation",
            comparison="invalid_interpretation", reason="wrong_intent_family",
        )
    try:
        compiled_spec = compile_callable(
            spec, interpretation, **authoritative_scope(spec, context)
        )
    except (TypeError, ValueError):
        return CountShadowResult(
            attempted=True, eligible=True, status="invalid_interpretation",
            comparison="conflict", reason="scope_or_compilation_rejected",
        )
    comparison = count_shadow_comparison(spec, compiled_spec, context)
    if comparison == "conflict":
        return CountShadowResult(
            attempted=True, eligible=True, status="invalid_interpretation",
            comparison=comparison, reason="compiled_scope_conflict",
        )
    return CountShadowResult(
        attempted=True, eligible=True, status="validated",
        comparison=comparison, compiled_spec=compiled_spec,
    )
