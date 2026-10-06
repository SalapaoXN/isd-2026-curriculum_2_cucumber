"""End-to-end semantic pipeline: interpret → validate → merge → resolve
→ plan → execute → verify → present, with trace, latency, and budgets.

One interpretation call per turn; at most one SQL-generation call and one
answerer call. Every stage fails closed with a typed category — a plausible
wrong answer is never preferable to a safe failure.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, replace as _replace_resolved
from pathlib import Path
from typing import Any

from rag.grounded_answer import GroundedAnswerResult
from rag.query_spec import QuerySpec
from rag.resolution import QueryContext
from rag.semantic.answerer import render_semantic_answer
from rag.semantic.compiler import compile_resolved_intent_to_query_spec
from rag.semantic.context import merge_semantic_context
from rag.semantic.executor import (
    execute_comparison,
    execute_deterministic,
    execute_policy,
    execute_sql_bridge,
)
from rag.semantic.interpreter import SemanticSchemaError, interpret_semantic_intent
from rag.semantic.planner import (
    EXECUTION_DETERMINISTIC,
    EXECUTION_POLICY,
    EXECUTION_SQL,
    plan_semantic_query,
)
from rag.semantic.resolver import resolve_semantic_intent
from rag.semantic.schema import (
    LEGACY_STATUS_FOR_INTERNAL,
    ResolvedIntent,
    SemanticIntent,
    VerifiedResult,
)
from rag.semantic.trace import SemanticTrace, StageTiming
from rag.semantic.validation import validate_semantic_intent


@dataclass(slots=True)
class SemanticPipelineResult:
    result: GroundedAnswerResult
    next_context: dict[str, Any] | None
    trace: SemanticTrace


def _intent_summary(intent: SemanticIntent) -> dict[str, Any]:
    """Trace-grade intent projection (nested shape matches dataset gold).

    Records the full linguistic structure the model returned — including
    fields dataset gold constrains (target/scope nesting, filters,
    requested fields, clarification, observed value) — so evaluation can
    distinguish model output from validator/resolver behavior.
    """
    return {
        "task": intent.task,
        "subject": intent.subject,
        "relation": intent.relation,
        "target": {
            "kind": intent.target.kind,
            "raw_text": intent.target.raw_text,
            "normalized_hint": intent.target.normalized_hint,
            "ordinal": intent.target.ordinal,
        },
        "scope": {
            "program": intent.scope.program,
            "catalog": intent.scope.catalog,
            "plan": intent.scope.plan,
            "plan_hint": intent.scope.plan_hint,
            "year": intent.scope.year,
            "semester": intent.scope.semester,
        },
        "filters": [
            {"field": item.field, "operator": item.operator} for item in intent.filters
        ],
        "has_aggregation": intent.aggregation is not None,
        "has_ranking": intent.ranking is not None,
        "has_comparison": intent.comparison is not None,
        "requested_fields": list(intent.requested_fields),
        "clarification": intent.clarification,
        "policy_topic": intent.policy_topic,
        "observed_value": intent.observed_value,
    }


def _resolved_summary(resolved: ResolvedIntent) -> dict[str, Any]:
    return {
        "program": resolved.scope.program,
        "catalog_key": resolved.scope.catalog_key,
        "plan": resolved.scope.plan,
        "target_kind": resolved.target.kind,
        "has_course_code": resolved.target.course_code is not None,
        "needs_clarification": resolved.needs_clarification,
        "clarification_reason": resolved.clarification_reason,
        "comparison_sides": [
            {"unresolved": side.unresolved, "reason": side.reason}
            for side in resolved.comparison_sides
        ],
    }


def _is_ambiguous_edition_scope(db_path: str | Path, resolved: ResolvedIntent) -> bool:
    """Return True when a multi-edition program lacks an authoritative catalog.

    Generic edition-safety rule (mirrors the legacy clarify_catalog
    contract): 0 catalogs is the resolver's own failure; 1 authoritative
    catalog (or none needed) proceeds; >1 plausible editions without a
    disambiguator fail closed. Never infers newest/oldest, never asks the
    model to choose.
    """
    program = resolved.scope.program
    if not isinstance(program, str) or not program.strip():
        return False
    if resolved.intent.comparison is not None and len(resolved.comparison_sides) == 2:
        # A comparison may explicitly bind each side to separate editions.
        # Those independent canonical scopes supersede the shared top-level
        # program context; require both sides to resolve their own catalog.
        return any(side.scope.catalog_key is None for side in resolved.comparison_sides)
    if resolved.scope.catalog_key is not None:
        return False
    try:
        from rag.structured.queries import edition_catalog_keys_for_program

        editions = edition_catalog_keys_for_program(db_path, program)
    except Exception:
        return True
    return len(tuple(editions)) > 1


def _fail_closed(
    trace: SemanticTrace,
    category: str,
    reason: str,
    internal_status: str,
    missing: tuple[str, ...] = (),
    *,
    started: float | None = None,
) -> SemanticPipelineResult:
    if started is not None:
        # On very fast safe failures Windows' monotonic clock can return the
        # same millisecond tick as the start. Preserve the invariant that a
        # completed request has a positive trace latency without fabricating
        # a meaningful duration (sub-tick values are represented as 0.001ms).
        elapsed_ms = (time.monotonic() - started) * 1000.0
        trace.timing.total_ms = max(elapsed_ms, 0.001)
    trace.failure_category = category
    trace.failure_reason = reason
    trace.verified_summary = {"status": internal_status, "missing": list(missing)}
    result = GroundedAnswerResult(
        status=LEGACY_STATUS_FOR_INTERNAL.get(internal_status, "insufficient_evidence"),
        answer_mode="deterministic",
        final_answer="",
        claims=(),
        provenance=(),
    )
    return SemanticPipelineResult(result=result, next_context=None, trace=trace)


def _next_context_for(
    resolved: ResolvedIntent, verified: VerifiedResult
) -> dict[str, Any] | None:
    if verified.status != "answer":
        return None
    context: dict[str, Any] = {}
    if resolved.scope.program is not None:
        context["program"] = resolved.scope.program
    if resolved.scope.catalog_key is not None:
        context["catalog_key"] = resolved.scope.catalog_key
    if resolved.scope.plan is not None:
        context["plan"] = resolved.scope.plan
    if resolved.scope.years:
        context["years"] = list(resolved.scope.years)
    if resolved.scope.semesters:
        context["semesters"] = list(resolved.scope.semesters)
    if resolved.target.course_code is not None:
        focus: dict[str, Any] = {"course_code": resolved.target.course_code}
        if resolved.target.program is not None:
            focus["program"] = resolved.target.program
        if resolved.target.catalog_key is not None:
            focus["catalog_key"] = resolved.target.catalog_key
        context["focus_course"] = focus
    if verified.result_courses:
        context["result_courses"] = [
            dict(entry) for entry in verified.result_courses
        ]
    if verified.result_scope_program is not None:
        context["result_scope_program"] = verified.result_scope_program
    return context or None


def semantic_answer(
    db_path: str | Path,
    question: str,
    conversation_context: dict[str, Any] | None = None,
    *,
    interpret_callable: Callable[..., str] | None = None,
    answer_callable: Callable[..., str] | None = None,
    sql_callable: Callable[..., str] | None = None,
    allow_hint_candidates: bool = False,
) -> SemanticPipelineResult:
    """Run the full semantic pipeline with dependency-injected providers."""
    started = time.monotonic()
    timing = StageTiming()
    trace = SemanticTrace(question=question if isinstance(question, str) else "")
    counts = {"llm": 0}
    raw_outputs: list[str] = []

    def counting(callable_object: Callable[..., str] | None) -> Callable[..., str] | None:
        if not callable(callable_object):
            return None

        def wrapped(*args: Any, **kwargs: Any) -> str:
            counts["llm"] += 1
            output = callable_object(*args, **kwargs)
            if isinstance(output, str):
                raw_outputs.append(output[:2000])
            return output

        return wrapped

    interpret = counting(interpret_callable)
    answer_provider = counting(answer_callable)
    sql_provider = counting(sql_callable)

    if not isinstance(question, str) or not question.strip():
        trace.timing = timing
        return _fail_closed(
            trace, "INTERPRETATION_ERROR", "empty question", "invalid_interpretation",
            started=started,
        )
    if interpret is None:
        trace.timing = timing
        return _fail_closed(
            trace,
            "INTERPRETATION_ERROR",
            "no interpreter provider",
            "invalid_interpretation",
            started=started,
        )

    stage_started = time.monotonic()
    try:
        intent, _ = interpret_semantic_intent(question, interpret)
    except SemanticSchemaError as error:
        timing.interpreter_ms = (time.monotonic() - stage_started) * 1000.0
        trace.timing = timing
        trace.llm_request_count = counts["llm"]
        if raw_outputs:
            trace.interpreter_raw = raw_outputs[0]
        return _fail_closed(
            trace, "INTERPRETATION_ERROR", str(error), "invalid_interpretation",
            started=started,
        )
    except Exception as error:
        timing.interpreter_ms = (time.monotonic() - stage_started) * 1000.0
        trace.timing = timing
        trace.llm_request_count = counts["llm"]
        return _fail_closed(
            trace,
            "INTERPRETATION_ERROR",
            f"interpreter provider failed: {type(error).__name__}",
            "invalid_interpretation",
            started=started,
        )
    timing.interpreter_ms = (time.monotonic() - stage_started) * 1000.0
    trace.timing = timing
    trace.llm_request_count = counts["llm"]
    if raw_outputs:
        trace.interpreter_raw = raw_outputs[0]
    trace.semantic_intent = _intent_summary(intent)

    validation = validate_semantic_intent(intent, question)
    trace.validation = {"valid": validation.valid, "reason": validation.reason}
    if not validation.valid:
        trace.timing = timing
        trace.llm_request_count = counts["llm"]
        return _fail_closed(
            trace, "VALIDATION_ERROR", validation.reason or "invalid", "invalid_interpretation",
            started=started,
        )

    stage_started = time.monotonic()
    merged = merge_semantic_context(
        intent, conversation_context if isinstance(conversation_context, dict) else None
    )
    if not merged.valid:
        trace.timing = timing
        trace.llm_request_count = counts["llm"]
        return _fail_closed(
            trace, "CONTEXT_ERROR", merged.reason or "invalid", "missing_scope",
            started=started,
        )
    trace.context_merge = {
        "program": merged.program,
        "catalog_key": merged.catalog_key,
        "invalidated": list(merged.invalidated),
    }

    resolved = resolve_semantic_intent(
        db_path, intent, merged, allow_hint_candidates=allow_hint_candidates
    )
    if (
        not resolved.needs_clarification
        and resolved.intent.comparison is not None
        and not resolved.comparison_sides
    ):
        # Comparison operands resolve independently per side (never merged):
        # shared program/catalog/year may coincide, plan and course identity
        # stay side-local so no cross-plan evidence leakage is possible.
        from rag.semantic.resolver import resolve_comparison_operand

        sides = tuple(
            resolve_comparison_operand(
                db_path,
                side,
                resolved.scope.program,
                resolved.scope.catalog_key,
            )
            for side in (
                resolved.intent.comparison.left,
                resolved.intent.comparison.right,
            )
        )
        resolved = _replace_resolved(resolved, comparison_sides=sides)
    timing.resolution_ms = (time.monotonic() - stage_started) * 1000.0
    trace.resolved_intent = _resolved_summary(resolved)
    if resolved.needs_clarification:
        trace.timing = timing
        trace.llm_request_count = counts["llm"]
        return _fail_closed(
            trace,
            "EXPECTED_SAFE_FAILURE",
            resolved.clarification_reason or "clarification required",
            "ambiguous_entity"
            if "ambiguous" in (resolved.clarification_reason or "")
            else "missing_scope",
            started=started,
        )
    if _is_ambiguous_edition_scope(db_path, resolved):
        # A multi-edition program without an authoritative catalog must not
        # aggregate across editions or silently pick one: fail closed exactly
        # like the legacy clarify_catalog contract (semantic mode reports it
        # as missing scope since there is no interactive edition picker).
        trace.timing = timing
        trace.llm_request_count = counts["llm"]
        return _fail_closed(
            trace,
            "EXPECTED_SAFE_FAILURE",
            "program scope matches multiple editions without an authoritative catalog",
            "missing_scope",
            started=started,
        )

    plan = plan_semantic_query(resolved)
    trace.query_plan = {"execution": plan.execution, "reason": plan.reason}

    stage_started = time.monotonic()
    if plan.execution == EXECUTION_POLICY:
        verified = execute_policy(db_path, resolved)
    elif plan.execution == EXECUTION_DETERMINISTIC and resolved.intent.task == "compare":
        verified = execute_comparison(db_path, resolved)
    elif plan.execution == EXECUTION_SQL:
        if sql_provider is None:
            verified = VerifiedResult(
                status="unsupported",
                missing_information=("ไม่มีผู้ให้บริการ SQL สำหรับคำถามเชิงประกอบ",),
                failure_category="EXPECTED_SAFE_FAILURE",
            )
        else:
            service_context: dict[str, Any] = {}
            if merged.program is not None:
                service_context["program"] = merged.program
            if merged.catalog_key is not None:
                service_context["catalog_key"] = merged.catalog_key
            verified = execute_sql_bridge(
                db_path,
                resolved,
                plan,
                question,
                sql_provider,
                answer_provider
                if answer_provider is not None
                else (lambda *args, **kwargs: ""),
                service_context,
            )
    elif plan.execution == EXECUTION_DETERMINISTIC:
        spec: QuerySpec = compile_resolved_intent_to_query_spec(resolved, question)
        resolution_context = QueryContext(
            program=resolved.scope.program,
            catalog_key=resolved.scope.catalog_key,
            plan=resolved.scope.plan,
            years=tuple(resolved.scope.years),
            semesters=tuple(resolved.scope.semesters),
        )
        verified = execute_deterministic(db_path, spec, resolution_context, question)
    else:
        verified = VerifiedResult(
            status="unsupported",
            missing_information=(plan.reason,),
            failure_category="EXPECTED_SAFE_FAILURE",
        )
    timing.execution_ms = (time.monotonic() - stage_started) * 1000.0
    trace.verified_summary = {
        "status": verified.status,
        "fact_count": len(verified.summary_facts),
        "claim_count": len(verified.claims),
        "provenance_count": len(verified.provenance),
        "missing": list(verified.missing_information),
    }
    if verified.status != "answer":
        trace.timing = timing
        trace.llm_request_count = counts["llm"]
        category = verified.failure_category
        if category == "NONE":
            category = "EXPECTED_SAFE_FAILURE"
        return _fail_closed(
            trace,
            category,
            "; ".join(verified.missing_information) or verified.status,
            verified.status if verified.status in LEGACY_STATUS_FOR_INTERNAL else "missing_data",
            verified.missing_information,
            started=started,
        )

    stage_started = time.monotonic()
    answer_text, answer_mode = render_semantic_answer(
        question, verified, answer_provider
    )
    timing.answerer_ms = (time.monotonic() - stage_started) * 1000.0
    timing.total_ms = (time.monotonic() - started) * 1000.0
    trace.timing = timing
    trace.llm_request_count = counts["llm"]
    trace.failure_category = "NONE"
    result = GroundedAnswerResult(
        status="answer",
        answer_mode=answer_mode,
        final_answer=answer_text,
        claims=tuple(verified.claims),
        provenance=tuple(verified.provenance),
    )
    return SemanticPipelineResult(
        result=result,
        next_context=_next_context_for(resolved, verified),
        trace=trace,
    )


__all__ = ["SemanticPipelineResult", "semantic_answer"]
