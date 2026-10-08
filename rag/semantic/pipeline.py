"""End-to-end semantic pipeline: interpret → validate → merge → resolve
→ plan → execute → verify → present, with trace, latency, and budgets.

One interpretation call per turn; at most one SQL-generation call and one
answerer call. Every stage fails closed with a typed category — a plausible
wrong answer is never preferable to a safe failure.
"""

from __future__ import annotations

import sqlite3
import re
import time
from copy import deepcopy
from collections.abc import Callable
from dataclasses import dataclass, replace as _replace_resolved
from pathlib import Path
from typing import Any

from rag.grounded_answer import GroundedAnswerResult
from rag.query_spec import QuerySpec, detect_surface_operations
from rag.resolution import QueryContext
from rag.semantic.answerer import render_semantic_answer
from rag.semantic.compiler import compile_resolved_intent_to_query_spec
from rag.semantic.context import merge_semantic_context, _validated_prior_context
from rag.semantic.errors import SemanticOperationalError, is_provider_error as _is_provider_error
from rag.semantic.executor import (
    _missing_answer_provider,
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
    MissingScopeRequirement,
    missing_comparison_plan,
    plan_semantic_query,
)
from rag.semantic.resolver import (
    canonical_catalog_key, canonical_program, resolve_semantic_intent, valid_plan,
)
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


def _canonical_program_code_candidates(db_path: str | Path) -> tuple[str, ...]:
    """Read current canonical program codes for interpreter role disambiguation.

    These are bounded scope candidates only. The resolver still validates any
    candidate selected from the current user question.
    """
    try:
        uri = f"{Path(db_path).resolve().as_uri()}?mode=ro"
        connection = sqlite3.connect(uri, uri=True)
        try:
            rows = connection.execute(
                """SELECT DISTINCT trim(program_code) AS program_code
                   FROM programs
                   WHERE program_code IS NOT NULL AND trim(program_code) != ''
                   ORDER BY lower(trim(program_code)), trim(program_code)"""
            ).fetchall()
        finally:
            connection.close()
    except (OSError, sqlite3.Error, TypeError, ValueError):
        return ()
    return tuple(
        row[0]
        for row in rows
        if isinstance(row[0], str) and row[0].strip()
    )


def _canonical_placement_category_candidates(db_path: str | Path) -> tuple[str, ...]:
    """Read canonical placement category labels as bounded filter vocabulary."""
    try:
        uri = f"{Path(db_path).resolve().as_uri()}?mode=ro"
        connection = sqlite3.connect(uri, uri=True)
        try:
            rows = connection.execute(
                """SELECT DISTINCT trim(category) AS category
                   FROM plan_placements
                   WHERE category IS NOT NULL AND trim(category) != ''
                   ORDER BY lower(trim(category)), trim(category)"""
            ).fetchall()
        finally:
            connection.close()
    except (OSError, sqlite3.Error, TypeError, ValueError):
        return ()
    return tuple(
        row[0] for row in rows if isinstance(row[0], str) and row[0].strip()
    )


def _canonical_plan_key_candidates(
    db_path: str | Path,
    question: str,
    conversation_context: dict[str, Any] | None,
    program_codes: tuple[str, ...],
) -> tuple[str, ...]:
    """Read plan keys for current-turn program mentions/validated context only."""
    selected_programs = [
        code
        for code in program_codes
        if re.search(
            r"(?<![A-Za-z0-9_])" + re.escape(code) + r"(?![A-Za-z0-9_])",
            question,
            re.IGNORECASE,
        )
    ]
    context_program = (
        conversation_context.get("program")
        if isinstance(conversation_context, dict)
        else None
    )
    context_catalog = (
        conversation_context.get("catalog_key")
        if isinstance(conversation_context, dict)
        else None
    )
    if isinstance(context_program, str) and context_program.strip():
        if not any(
            code.casefold() == context_program.strip().casefold()
            for code in selected_programs
        ):
            selected_programs.append(context_program.strip())
    if not selected_programs and not (
        isinstance(context_catalog, str) and context_catalog.strip()
    ):
        return ()

    plan_keys: list[str] = []
    try:
        uri = f"{Path(db_path).resolve().as_uri()}?mode=ro"
        connection = sqlite3.connect(uri, uri=True)
        try:
            for program in selected_programs:
                catalog_filter = ""
                parameters: list[str] = [program]
                if (
                    isinstance(context_program, str)
                    and context_program.strip().casefold() == program.strip().casefold()
                    and isinstance(context_catalog, str)
                    and context_catalog.strip()
                ):
                    catalog_filter = " AND lower(trim(c.catalog_key)) = ?"
                    parameters.append(context_catalog.strip())
                rows = connection.execute(
                    """SELECT DISTINCT trim(cp.plan_key) AS plan_key
                       FROM curriculum_plans cp
                       JOIN programs p ON p.program_id = cp.program_id
                       JOIN catalogs c ON c.catalog_id = cp.catalog_id
                       WHERE lower(trim(p.program_code)) = lower(trim(?))"""
                    + catalog_filter
                    + " ORDER BY lower(trim(cp.plan_key)), trim(cp.plan_key)",
                    tuple(parameters),
                ).fetchall()
                for row in rows:
                    value = row[0]
                    if isinstance(value, str) and value.strip() and value not in plan_keys:
                        plan_keys.append(value.strip())
            if not selected_programs and isinstance(context_catalog, str):
                rows = connection.execute(
                    """SELECT DISTINCT trim(cp.plan_key) AS plan_key
                       FROM curriculum_plans cp
                       JOIN catalogs c ON c.catalog_id = cp.catalog_id
                       WHERE lower(trim(c.catalog_key)) = lower(trim(?))
                       ORDER BY lower(trim(cp.plan_key)), trim(cp.plan_key)""",
                    (context_catalog.strip(),),
                ).fetchall()
                for row in rows:
                    value = row[0]
                    if isinstance(value, str) and value.strip() and value not in plan_keys:
                        plan_keys.append(value.strip())
        finally:
            connection.close()
    except (OSError, sqlite3.Error, TypeError, ValueError):
        return ()
    return tuple(plan_keys)


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
    if (
        resolved.intent.task == "list" and resolved.intent.subject == "course"
        and resolved.intent.target.kind == "none" and not resolved.intent.filters
        and resolved.intent.relation is None and resolved.scope.program is not None
        and (resolved.scope.years or resolved.scope.semesters)
    ):
        context["last_normal_operation"] = {
            "kind": "list_courses", "program": resolved.scope.program,
            "catalog_key": resolved.scope.catalog_key, "plan": resolved.scope.plan,
            "years": list(resolved.scope.years), "semesters": list(resolved.scope.semesters),
        }
    return context or None


def _scope_conflict(
    question: str, reason: str, context: dict[str, Any] | None,
) -> SemanticPipelineResult:
    trace = SemanticTrace(question=question)
    trace.failure_category = "EXPECTED_SAFE_FAILURE"
    trace.failure_reason = reason
    return SemanticPipelineResult(
        result=GroundedAnswerResult(
            status="context_conflict", answer_mode="deterministic", final_answer=reason,
        ),
        next_context=deepcopy(context), trace=trace,
    )


def _scope_clarification(
    trace: SemanticTrace, dimension: str, program: str | None = None,
    operand: str | None = None, *, started: float,
) -> SemanticPipelineResult:
    """Adapt a known scope failure using the existing clarification carrier."""
    label = {
        "program": "หลักสูตร", "catalog": "ปีหลักสูตร/ฉบับหลักสูตร",
        "plan": "แผนการเรียน", "comparison_operation": "ลักษณะการเปรียบเทียบที่ต้องการ เช่น ส่วนต่าง หรือความเท่ากัน",
    }[dimension]
    text = f"กรุณาระบุ{label}"
    if program is not None:
        text += f"ของ {program} ที่ต้องการ"
    if operand is not None:
        text += "ใช้ในการเปรียบเทียบฝั่ง" + ("ซ้าย" if operand == "left" else "ขวา")
    outcome = _fail_closed(
        trace, "EXPECTED_SAFE_FAILURE", text, "missing_scope", started=started,
    )
    trace.verified_summary.update({
        "scope_dimension": dimension, "program": program, "operand": operand,
    })
    outcome.result = GroundedAnswerResult(
        status="clarify_program", answer_mode="deterministic", final_answer=text,
    )
    return outcome


def _matches_operand_resolution(
    db_path: str | Path, requirement: MissingScopeRequirement | None, resolution: Any,
) -> bool:
    fields = {"dimension", "program", "operand", "value"}
    if (
        not isinstance(resolution, dict) or set(resolution) != fields
        or any(
            not isinstance(resolution[key], str) or not resolution[key]
            or len(resolution[key]) > 80 for key in fields
        )
        or requirement is None or resolution["dimension"] != requirement.dimension
        or resolution["operand"] not in {"left", "right"}
        or resolution["operand"] != requirement.operand
        or resolution["program"] != requirement.program
        or canonical_program(db_path, resolution["program"]) != requirement.program
    ):
        return False
    return True


def _apply_operand_plan_resolution(
    db_path: str | Path, resolved: ResolvedIntent,
    requirement: MissingScopeRequirement | None, resolution: Any,
) -> ResolvedIntent | None:
    """Bind a one-shot user selection only to the current missing operand."""
    if not _matches_operand_resolution(db_path, requirement, resolution):
        return None
    index = 0 if resolution["operand"] == "left" else 1
    side = resolved.comparison_sides[index]
    comparison = resolved.intent.comparison
    raw_side = dict(comparison.left if index == 0 else comparison.right)
    if side.unresolved or side.scope.plan is not None or raw_side.get("plan") is not None:
        return None
    plan = valid_plan(db_path, resolution["value"], side.scope.program, side.scope.catalog_key)
    if plan is None:
        return None
    sides = list(resolved.comparison_sides)
    sides[index] = _replace_resolved(side, scope=_replace_resolved(side.scope, plan=plan))
    return _replace_resolved(resolved, comparison_sides=tuple(sides))


def _missing_operand_catalog(db_path: str | Path, resolved: ResolvedIntent) -> MissingScopeRequirement | None:
    comparison = resolved.intent.comparison
    if comparison is None or len(resolved.comparison_sides) != 2:
        return None
    raw_sides = (comparison.left, comparison.right)
    for label, side, raw in zip(("left", "right"), resolved.comparison_sides, raw_sides):
        if side.unresolved and side.reason in {
            "ambiguous operand catalog", "unknown operand catalog", "unknown operand program",
        }:
            if side.reason != "ambiguous operand catalog" or dict(raw).get("catalog") is not None:
                return None
            program = canonical_program(db_path, dict(raw).get("program", resolved.scope.program))
            return MissingScopeRequirement("catalog", program, label)
    if _is_ambiguous_edition_scope(db_path, resolved):
        for label, side, raw in zip(("left", "right"), resolved.comparison_sides, raw_sides):
            if side.scope.catalog_key is None and dict(raw).get("catalog") is None:
                return MissingScopeRequirement("catalog", side.scope.program, label)
    return None


def _apply_operand_catalog_resolution(db_path: str | Path, resolved: ResolvedIntent, resolution: Any) -> ResolvedIntent | None:
    requirement = _missing_operand_catalog(db_path, resolved)
    if not _matches_operand_resolution(db_path, requirement, resolution):
        return None
    index = 0 if resolution["operand"] == "left" else 1
    comparison = resolved.intent.comparison
    raw = dict(comparison.left if index == 0 else comparison.right)
    if raw.get("catalog") is not None:
        return None
    catalog = canonical_catalog_key(db_path, resolution["value"], requirement.program)
    if catalog is None:
        return None
    raw["catalog"] = catalog
    from rag.semantic.resolver import resolve_comparison_operand
    side = resolve_comparison_operand(
        db_path, tuple(raw.items()), resolved.scope.program, resolved.scope.catalog_key,
        default_years=resolved.scope.years, default_semesters=resolved.scope.semesters,
    )
    if side.unresolved:
        return None
    sides = list(resolved.comparison_sides)
    sides[index] = side
    return _replace_resolved(resolved, comparison_sides=tuple(sides))


def _normalize_operand_resolutions(singular: Any, plural: Any) -> list[dict[str, str]]:
    if singular is not None and plural is not None:
        raise ValueError("ambiguous clarification resolution transport")
    items = plural if plural is not None else [singular] if singular is not None else []
    if not isinstance(items, list) or len(items) > 4:
        raise ValueError("clarification resolutions must be a bounded list")
    unique: dict[tuple[str, str], dict[str, str]] = {}
    for item in items:
        if (
            not isinstance(item, dict) or set(item) != {"dimension", "program", "operand", "value"}
            or any(not isinstance(value, str) or not value or len(value) > 80 for value in item.values())
            or item["dimension"] not in {"catalog", "plan"}
            or item["operand"] not in {"left", "right"}
        ):
            raise ValueError("invalid clarification resolution")
        key = (item["operand"], item["dimension"])
        if key in unique and unique[key] != item:
            raise ValueError("conflicting clarification resolutions")
        unique[key] = dict(item)
    return list(unique.values())


def _apply_operand_resolution_chain(
    db_path: str | Path, resolved: ResolvedIntent, resolutions: list[dict[str, str]],
) -> ResolvedIntent | None:
    remaining = list(resolutions)
    while remaining:
        requirement = _missing_operand_catalog(db_path, resolved) or missing_comparison_plan(resolved)
        if requirement is None:
            return None
        selected = next((item for item in remaining if _matches_operand_resolution(db_path, requirement, item)), None)
        if selected is None:
            return None
        retry = (
            _apply_operand_catalog_resolution(db_path, resolved, selected)
            if requirement.dimension == "catalog"
            else _apply_operand_plan_resolution(db_path, resolved, requirement, selected)
        )
        if retry is None:
            return None
        resolved = retry
        remaining.remove(selected)
    return resolved


def semantic_answer(
    db_path: str | Path,
    question: str,
    conversation_context: dict[str, Any] | None = None,
    *,
    home_program: str | None = None,
    clarification_resolution: dict[str, Any] | None = None,
    clarification_resolutions: list[dict[str, Any]] | None = None,
    interpret_callable: Callable[..., str] | None = None,
    answer_callable: Callable[..., str] | None = None,
    sql_callable: Callable[..., str] | None = None,
    allow_hint_candidates: bool = False,
) -> SemanticPipelineResult:
    """Apply request scope policy without mixing comparisons into normal state."""
    prior = deepcopy(conversation_context) if isinstance(conversation_context, dict) else None
    home = None
    normal = deepcopy(prior)
    if home_program is not None:
        home = canonical_program(db_path, home_program)
        if home is None:
            return _scope_conflict(question, "หลักสูตรประจำแชทไม่ตรงกับข้อมูลหลักสูตร", prior)
        normal = normal or {}
        raw_program = normal.get("program")
        catalog = normal.get("catalog_key")
        plan = normal.get("plan")
        if (
            (raw_program is not None and canonical_program(db_path, raw_program) != home)
            or (catalog is not None and canonical_catalog_key(db_path, catalog, home) is None)
            or (plan is not None and valid_plan(db_path, plan, home, catalog) is None)
        ):
            return _scope_conflict(question, "ขอบเขตการสนทนาไม่ตรงกับหลักสูตรประจำแชท", prior)
        normal["program"] = home
    outcome = _run_semantic_answer(
        db_path, question, normal, home_program=home,
        clarification_resolution=clarification_resolution,
        clarification_resolutions=clarification_resolutions,
        interpret_callable=interpret_callable, answer_callable=answer_callable,
        sql_callable=sql_callable, allow_hint_candidates=allow_hint_candidates,
    )
    # Comparison operands/results are temporary, including on safe failure.
    # Ordinary scoped failures must also leave the previous normal state intact.
    if outcome.trace.semantic_intent.get("task") == "compare" or (
        home is not None and outcome.result.status != "answer"
    ):
        outcome.next_context = deepcopy(normal)
        if isinstance(outcome.next_context, dict):
            # The immediately prior turn is no longer a successful ordinary list.
            outcome.next_context.pop("last_normal_operation", None)
    return outcome


def _run_semantic_answer(
    db_path: str | Path,
    question: str,
    conversation_context: dict[str, Any] | None = None,
    *,
    home_program: str | None = None,
    clarification_resolution: dict[str, Any] | None = None,
    clarification_resolutions: list[dict[str, Any]] | None = None,
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
        program_codes = _canonical_program_code_candidates(db_path)
        previous_operation = _validated_prior_context(conversation_context).get("last_normal_operation")
        intent, _ = interpret_semantic_intent(
            question,
            interpret,
            canonical_program_codes=program_codes,
            canonical_category_labels=_canonical_placement_category_candidates(db_path),
            last_normal_operation=previous_operation,
            canonical_plan_keys=_canonical_plan_key_candidates(
                db_path,
                question,
                conversation_context if isinstance(conversation_context, dict) else None,
                program_codes,
            ),
        )
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
        if _is_provider_error(error):
            raise SemanticOperationalError("provider_unavailable") from None
        raise SemanticOperationalError("error") from None
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
        if (
            intent.task == "compare" and intent.comparison is not None
            and intent.comparison.operation is None
            and validation.reason == "compare requires an explicit comparison operation"
        ):
            return _scope_clarification(trace, "comparison_operation", started=started)
        return _fail_closed(
            trace, "VALIDATION_ERROR", validation.reason or "invalid", "invalid_interpretation",
            started=started,
        )

    try:
        resolutions = _normalize_operand_resolutions(clarification_resolution, clarification_resolutions)
    except ValueError as error:
        return _fail_closed(
            trace, "VALIDATION_ERROR", str(error), "invalid_interpretation", started=started,
        )
    if resolutions and intent.task != "compare":
        return _fail_closed(
            trace, "VALIDATION_ERROR", "operand resolution requires a comparison",
            "invalid_interpretation", started=started,
        )

    if home_program is not None and intent.task != "compare" and intent.scope.program is not None:
        if canonical_program(db_path, intent.scope.program) != home_program:
            outcome = _scope_conflict(
                question,
                f"แชทนี้กำหนดไว้สำหรับหลักสูตร {home_program} หากต้องการถาม "
                f"{intent.scope.program} โดยตรง ให้ใช้แชทหลักสูตรนั้นหรือแชทที่ไม่ได้กำหนดหลักสูตร",
                conversation_context,
            )
            outcome.trace = trace
            trace.failure_category = "EXPECTED_SAFE_FAILURE"
            trace.failure_reason = outcome.result.final_answer
            return outcome

    if intent.task == "unknown":
        return _fail_closed(
            trace, "EXPECTED_SAFE_FAILURE", "unsupported judgement or intent",
            "unsupported", started=started,
        )

    if (
        intent.task == "list" and intent.subject == "course"
        and intent.target.kind == "none" and not intent.filters
        and intent.relation is None and (intent.scope.year is not None or intent.scope.semester is not None)
        and not detect_surface_operations(question)
    ):
        prior = _validated_prior_context(conversation_context)
        operation = prior.get("last_normal_operation")
        compatible = operation is not None and all(
            explicit is None or explicit.casefold() == str(prior.get(key, "")).casefold()
            for key, explicit in (("program", intent.scope.program),
                                  ("catalog_key", intent.scope.catalog), ("plan", intent.scope.plan))
        )
        if not compatible:
            return _fail_closed(
                trace, "EXPECTED_SAFE_FAILURE", "no compatible preceding normal list operation",
                "invalid_interpretation", started=started,
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
                default_years=resolved.scope.years,
                default_semesters=resolved.scope.semesters,
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
        reason = resolved.clarification_reason
        if reason in {"unknown program scope", "unknown catalog scope"}:
            return _scope_clarification(
                trace, "program" if reason == "unknown program scope" else "catalog",
                canonical_program(db_path, merged.program), started=started,
            )
        return _fail_closed(
            trace,
            "EXPECTED_SAFE_FAILURE",
            resolved.clarification_reason or "clarification required",
            "ambiguous_entity"
            if "ambiguous" in (resolved.clarification_reason or "")
            else "missing_scope",
            started=started,
        )
    if resolutions:
        retry = _apply_operand_resolution_chain(db_path, resolved, resolutions)
        if retry is None:
            return _fail_closed(
                trace, "VALIDATION_ERROR", "invalid or mismatched operand resolution chain",
                "invalid_interpretation", started=started,
            )
        resolved = retry
        trace.resolved_intent = _resolved_summary(resolved)
    if resolved.intent.comparison is not None:
        for label, side, raw_side in zip(
            ("left", "right"), resolved.comparison_sides,
            (resolved.intent.comparison.left, resolved.intent.comparison.right),
        ):
            if side.unresolved and side.reason in {
                "ambiguous operand catalog", "unknown operand catalog", "unknown operand program",
            }:
                dimension = "program" if side.reason == "unknown operand program" else "catalog"
                program = canonical_program(
                    db_path, dict(raw_side).get("program", resolved.scope.program),
                )
                return _scope_clarification(trace, dimension, program, label, started=started)
    if _is_ambiguous_edition_scope(db_path, resolved):
        # A multi-edition program without an authoritative catalog must not
        # aggregate across editions or silently pick one: fail closed exactly
        # like the legacy clarify_catalog contract (semantic mode reports it
        # as missing scope since there is no interactive edition picker).
        trace.timing = timing
        trace.llm_request_count = counts["llm"]
        if resolved.intent.comparison is not None:
            for label, side in zip(("left", "right"), resolved.comparison_sides):
                if side.scope.catalog_key is None:
                    return _scope_clarification(
                        trace, "catalog", side.scope.program, label, started=started,
                    )
        return _scope_clarification(trace, "catalog", resolved.scope.program, started=started)

    requirement = missing_comparison_plan(resolved)
    if requirement is not None:
        return _scope_clarification(
            trace, requirement.dimension, requirement.program, requirement.operand,
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
                else _missing_answer_provider,
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
        if verified.status == "missing_scope" and "resolution: clarify_program" in verified.missing_information:
            return _scope_clarification(trace, "program", started=started)
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
        question,
        verified,
        answer_provider,
        numeric_comparison=verified.numeric_comparison,
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


__all__ = ["SemanticOperationalError", "SemanticPipelineResult", "semantic_answer"]
