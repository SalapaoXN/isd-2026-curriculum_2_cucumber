"""Deterministic query planner over ResolvedIntent.

Chooses deterministic execution, guarded SQL execution, or typed
unsupported — from structure, never from raw wording. Rank/compare and
non-trivial aggregation/filter shapes route to the guarded SQL bridge;
anything outside supported canonical machinery fails cleanly with a typed
reason instead of unsafe free-form behavior.
"""

from __future__ import annotations

from dataclasses import dataclass

from rag.semantic.schema import ResolvedIntent

EXECUTION_DETERMINISTIC = "deterministic"
EXECUTION_SQL = "sql"
EXECUTION_POLICY = "policy"
EXECUTION_UNSUPPORTED = "unsupported"

_SIMPLE_FILTER_FIELDS = frozenset({"topic", "category", "has_prerequisite", "plan", "year", "semester"})


@dataclass(frozen=True, slots=True)
class SemanticPlan:
    execution: str
    reason: str
    needs_sql: bool = False


@dataclass(frozen=True, slots=True)
class MissingScopeRequirement:
    dimension: str
    program: str | None
    operand: str


def missing_comparison_plan(resolved: ResolvedIntent) -> MissingScopeRequirement | None:
    """Preflight only the single-plan shape already required by _side_credits."""
    comparison = resolved.intent.comparison
    if (
        resolved.needs_clarification
        or resolved.intent.task != "compare"
        or comparison is None
        or comparison.measure != "credits"
        or comparison.operation not in {"greater", "less", "equal", "difference"}
        or len(resolved.comparison_sides) != 2
        or any(side.unresolved for side in resolved.comparison_sides)
    ):
        return None
    for label, side in zip(("left", "right"), resolved.comparison_sides):
        if (
            side.target.course_code is None
            and (side.scope.years or side.scope.semesters)
            and side.scope.plan is None
        ):
            return MissingScopeRequirement("plan", side.scope.program, label)
    return None


def _filters_are_simple(resolved: ResolvedIntent) -> bool:
    for item in resolved.intent.filters:
        if item.field not in _SIMPLE_FILTER_FIELDS:
            return False
        if item.operator not in {"eq", "related_to"}:
            return False
    return True


def effective_aggregation_group_by(resolved: ResolvedIntent) -> tuple[str, ...]:
    """Remove grouping dimensions already fixed to one scoped value.

    A group-by over the exact year/semester/plan/program already selected
    by resolved scope cannot create multiple groups; it is redundant and
    must not force grouped/SQL execution. Other dimensions remain grouped.
    """
    aggregation = resolved.intent.aggregation
    if aggregation is None:
        return ()
    fixed: set[str] = set()
    scope = resolved.scope
    if len(tuple(scope.years)) == 1:
        fixed.add("year")
    if len(tuple(scope.semesters)) == 1:
        fixed.add("semester")
    if scope.plan is not None:
        fixed.add("plan")
    if scope.program is not None:
        fixed.add("program")
    categories = {
        item.value
        for item in resolved.intent.filters
        if item.field == "category" and item.operator == "eq"
    }
    if len(categories) == 1:
        fixed.add("category")
    return tuple(
        dimension
        for dimension in tuple(aggregation.group_by)
        if dimension not in fixed
    )


def plan_semantic_query(resolved: ResolvedIntent) -> SemanticPlan:
    """Route a resolved intent to an execution path with a typed reason."""
    intent = resolved.intent
    if resolved.needs_clarification:
        return SemanticPlan(
            EXECUTION_UNSUPPORTED,
            resolved.clarification_reason or "clarification required",
        )
    if intent.task == "unknown":
        return SemanticPlan(EXECUTION_UNSUPPORTED, "unsupported judgement or intent")
    if intent.task in {"policy", "requirement"}:
        return SemanticPlan(EXECUTION_POLICY, "canonical policy evidence")
    if intent.task == "lookup" and intent.subject == "program" and intent.relation == "credits":
        # Whole-program totals are authoritative program-requirement facts,
        # never placement-row sums. A requested year/semester is narrower
        # scope, however, and must use canonical placement aggregation.
        if not resolved.scope.years and not resolved.scope.semesters:
            return SemanticPlan(EXECUTION_POLICY, "authoritative program total")
        return SemanticPlan(EXECUTION_DETERMINISTIC, "scoped program credit aggregate")
    if intent.task == "lookup" and intent.subject in {"course", "program"}:
        if resolved.target.kind == "none" and intent.subject == "course":
            return SemanticPlan(EXECUTION_UNSUPPORTED, "course lookup without a target")
        return SemanticPlan(EXECUTION_DETERMINISTIC, "exact-entity lookup")
    if intent.task in {"list", "search"}:
        if _filters_are_simple(resolved):
            return SemanticPlan(EXECUTION_DETERMINISTIC, "scoped collection")
        return SemanticPlan(EXECUTION_SQL, "filtered collection needs SQL", needs_sql=True)
    if intent.task == "aggregate":
        aggregation = intent.aggregation
        if (
            aggregation is not None
            and not effective_aggregation_group_by(resolved)
            and (
                (aggregation.function == "sum" and aggregation.measure == "credits")
                or (
                    aggregation.function == "count"
                    and aggregation.measure == "course_count"
                )
            )
        ):
            return SemanticPlan(EXECUTION_DETERMINISTIC, "single total aggregate")
        return SemanticPlan(EXECUTION_SQL, "compositional query needs SQL", needs_sql=True)
    if intent.task == "compare":
        return SemanticPlan(EXECUTION_DETERMINISTIC, "verified comparison")
    if intent.task == "rank":
        return SemanticPlan(EXECUTION_SQL, "compositional query needs SQL", needs_sql=True)
    return SemanticPlan(EXECUTION_UNSUPPORTED, "unsupported task shape")


__all__ = [
    "MissingScopeRequirement",
    "missing_comparison_plan",
    "EXECUTION_DETERMINISTIC",
    "EXECUTION_POLICY",
    "EXECUTION_SQL",
    "EXECUTION_UNSUPPORTED",
    "SemanticPlan",
    "effective_aggregation_group_by",
    "plan_semantic_query",
]
