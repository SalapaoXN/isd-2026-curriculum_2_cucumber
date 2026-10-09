"""Strict typed schema for Semantic QA vNext interpreter output.

Authority boundary (see docs/semantic-qa-vnext.md):

- LANGUAGE (what the student means) ............ LLM interpreter output
- IDENTITY / SCOPE (which entities are valid) .. deterministic resolver
- FACTS (what is true) ......................... canonical SQLite
- TRUST (what may be claimed) .................. evidence + provenance
- PRESENTATION (how to say it) ................. LLM answerer (bounded)

Nothing in this module touches the database, the network, or legacy
natural-language parsing. All types are immutable value objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


SEMANTIC_INTENT_VERSION = "semantic-intent/v4"

TASKS = frozenset(
    {
        "lookup",
        "compose",
        "list",
        "search",
        "aggregate",
        "compare",
        "rank",
        "policy",
        "requirement",
        "unknown",
    }
)

SUBJECTS = frozenset(
    {
        "course",
        "program",
        "semester",
        "curriculum",
        "requirement",
        "policy",
    }
)

RELATIONS = frozenset(
    {
        "identity",
        "description",
        "credits",
        "prerequisite",
        "placement",
        "existence",
        "alternative_selection",
    }
)

TARGET_KINDS = frozenset(
    {
        "literal",
        "literal_set",
        "current_course",
        "result_ordinal",
        "previous_result_set",
        "none",
    }
)

FILTER_FIELDS = frozenset(
    {
        "topic",
        "category",
        "has_prerequisite",
        "credits",
        "year",
        "semester",
        "plan",
    }
)

FILTER_OPERATORS = frozenset(
    {
        "eq",
        "ne",
        "lt",
        "lte",
        "gt",
        "gte",
        "between",
        "related_to",
        "contains",
    }
)

AGGREGATION_FUNCTIONS = frozenset({"count", "sum", "average", "minimum", "maximum"})

MEASURES = frozenset({"course_count", "credits", "prerequisite_count"})
COMPARISON_MEASURES = MEASURES | {"placement"}
PLAN_SELECTORS = frozenset({"available_plans"})

GROUP_DIMENSIONS = frozenset({"year", "semester", "plan", "program", "category"})

RANK_DIRECTIONS = frozenset({"ascending", "descending"})

COMPARISON_OPERATIONS = frozenset(
    {
        "greater",
        "less",
        "equal",
        "difference",
        "set_difference",
        "overlap",
        "earliest_placement",
    }
)

# Structured operand dimensions for one comparison side. Course operands
# carry user-written text pending deterministic resolution, exactly like
# top-level literal targets; catalog carries an edition key.
COMPARISON_OPERAND_KEYS = frozenset(
    {
        "course",
        "plan",
        "plan_hint",
        "semester",
        "year",
        "catalog",
        "program",
    }
)

REQUESTED_FIELDS = frozenset(
    {
        "code",
        "name",
        "credits",
        "placement",
        "prerequisites",
        "prerequisite_placement",
        "description",
        "alternative_selection",
        "placement_sequence",
    }
)

POLICY_TOPICS = frozenset(
    {
        "honors",
        "probation",
        "graduation",
        "graduation_gpa",
        "english_exit",
        "registration",
        "leave",
        "resignation",
        "transfer",
        "conduct",
        "appeals",
        "reentry",
        "assessment",
        "grading",
    }
)

# Allowed (task, subject) pairs. Relation compatibility is judged
# separately by _relation_compatible rules in validation: lookup requires
# a relation, list/search/aggregate/compare/rank tolerate a compatible
# extra relation, policy/requirement/unknown require none.
VALID_TASK_SUBJECTS = frozenset(
    {
        ("lookup", "course"),
        ("compose", "course"),
        ("lookup", "program"),
        ("list", "course"),
        ("list", "semester"),
        ("list", "curriculum"),
        ("search", "course"),
        ("aggregate", "semester"),
        ("aggregate", "course"),
        ("aggregate", "program"),
        ("aggregate", "curriculum"),
        ("compare", "semester"),
        ("compare", "course"),
        ("compare", "program"),
        ("rank", "semester"),
        ("rank", "course"),
        ("policy", "policy"),
        ("policy", "requirement"),
        ("requirement", "requirement"),
        ("requirement", "policy"),
        ("unknown", "course"),
        ("unknown", "program"),
        ("unknown", "semester"),
        ("unknown", "curriculum"),
        ("unknown", "requirement"),
        ("unknown", "policy"),
    }
)

# Internal stage/failure taxonomy for trace and evaluation.
# EVAL_INFRA_ERROR is evaluation-harness only (provider quota/transport):
# such rows are excluded from accuracy denominators, never scored.
FAILURE_CATEGORIES = frozenset(
    {
        "INTERPRETATION_ERROR",
        "VALIDATION_ERROR",
        "CONTEXT_ERROR",
        "RESOLUTION_ERROR",
        "QUERY_ERROR",
        "DATA_ERROR",
        "GROUNDING_ERROR",
        "ANSWER_ERROR",
        "EXPECTED_SAFE_FAILURE",
        "EVAL_INFRA_ERROR",
        "NONE",
    }
)

# Internal pipeline statuses; mapped to the legacy public answer statuses.
INTERNAL_STATUSES = frozenset(
    {
        "answer",
        "unsupported",
        "missing_scope",
        "ambiguous_entity",
        "missing_data",
        "invalid_interpretation",
    }
)

LEGACY_STATUS_FOR_INTERNAL = {
    "answer": "answer",
    "unsupported": "unsupported",
    "missing_scope": "insufficient_evidence",
    "ambiguous_entity": "insufficient_evidence",
    "missing_data": "insufficient_evidence",
    "invalid_interpretation": "insufficient_evidence",
}

MAX_TEXT_LEN = 80
MAX_HINT_LEN = 80
MAX_ORDINAL = 50
MAX_COURSE_SET_MEMBERS = 20


class SemanticSchemaError(ValueError):
    """Raised when interpreter output violates the closed intent schema."""


@dataclass(frozen=True, slots=True)
class ScopeMention:
    """Linguistic scope as mentioned in the CURRENT user question only.

    ``plan_hint`` is an optional normalized spelling proposal for the plan
    (canonical plan keys only); like course normalized hints it is never
    identity until the deterministic resolver confirms it against canonical
    plan data.
    """

    program: str | None = None
    catalog: str | None = None
    plan: str | None = None
    plan_hint: str | None = None
    year: int | None = None
    semester: int | None = None


@dataclass(frozen=True, slots=True)
class LiteralCourseReference:
    """One current-turn course mention, not a canonical identity."""

    raw_text: str
    normalized_hint: str | None = None


@dataclass(frozen=True, slots=True)
class SemanticTarget:
    """What the user referred to — never a canonical database identity."""

    kind: str = "none"
    raw_text: str | None = None
    normalized_hint: str | None = None
    ordinal: int | None = None
    members: tuple[LiteralCourseReference, ...] = ()


@dataclass(frozen=True, slots=True)
class SemanticFilter:
    field: str = ""
    operator: str = "eq"
    value: Any = None


@dataclass(frozen=True, slots=True)
class AggregationSpec:
    function: str = "count"
    measure: str = "course_count"
    group_by: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RankingSpec:
    metric: str = "credits"
    direction: str = "descending"
    limit: int = 1


@dataclass(frozen=True, slots=True)
class ComparisonSpec:
    """Placement compares root course targets across left/right plan operands."""
    left: tuple[tuple[str, Any], ...] = ()
    right: tuple[tuple[str, Any], ...] = ()
    measure: str = "credits"
    operation: str | None = None
    plan_selector: str | None = None


@dataclass(frozen=True, slots=True)
class SemanticIntent:
    """Structured linguistic interpretation. Contains zero database facts."""

    version: str = SEMANTIC_INTENT_VERSION
    task: str = "unknown"
    subject: str = "course"
    relation: str | None = None
    target: SemanticTarget = field(default_factory=SemanticTarget)
    scope: ScopeMention = field(default_factory=ScopeMention)
    filters: tuple[SemanticFilter, ...] = ()
    aggregation: AggregationSpec | None = None
    ranking: RankingSpec | None = None
    comparison: ComparisonSpec | None = None
    requested_fields: tuple[str, ...] = ()
    clarification: str | None = None
    policy_topic: str | None = None
    observed_value: str | None = None


@dataclass(frozen=True, slots=True)
class ResolvedScope:
    """Deterministically validated canonical scope (no raw text)."""

    program: str | None = None
    catalog_key: str | None = None
    plan: str | None = None
    years: tuple[int, ...] = ()
    semesters: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class ResolvedTarget:
    """Deterministically resolved canonical target (DB-verified only)."""

    kind: str = "none"
    course_code: str | None = None
    course_name: str | None = None
    program: str | None = None
    catalog_key: str | None = None
    via_hint: bool = False
    members: tuple[ResolvedTarget, ...] = ()


@dataclass(frozen=True, slots=True)
class ResolvedOperand:
    """One independently resolved comparison side (no shared scope)."""

    scope: ResolvedScope = field(default_factory=ResolvedScope)
    target: ResolvedTarget = field(default_factory=ResolvedTarget)
    unresolved: bool = True
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class ResolvedIntent:
    """Linguistic intent bound to canonical identity/scope, still fact-free."""

    intent: SemanticIntent = field(default_factory=SemanticIntent)
    scope: ResolvedScope = field(default_factory=ResolvedScope)
    target: ResolvedTarget = field(default_factory=ResolvedTarget)
    hint_candidates: tuple[str, ...] = ()
    needs_clarification: bool = False
    clarification_reason: str | None = None
    comparison_sides: tuple[ResolvedOperand, ...] = ()


@dataclass(frozen=True, slots=True)
class VerifiedResult:
    """Facts the answerer may present — evidence-bound only."""

    status: str = "missing_data"
    summary_facts: tuple[str, ...] = ()
    claims: tuple[Any, ...] = ()
    provenance: tuple[Any, ...] = ()
    missing_information: tuple[str, ...] = ()
    failure_category: str = "NONE"
    result_courses: tuple[dict[str, Any], ...] = ()
    result_scope_program: str | None = None
    numeric_comparison: VerifiedNumericComparison | None = None
    alternative_selections: tuple[VerifiedAlternativeSelection, ...] = ()
    explicit_course_set: bool = False
    scoped_results: tuple[VerifiedScopedResult, ...] = ()
    placement_comparison: VerifiedPlacementComparison | None = None


@dataclass(frozen=True, slots=True)
class VerifiedPlacementCell:
    course_code: str
    course_name: str | None
    scope: ResolvedScope
    placements: tuple[tuple[int, int], ...]
    provenance: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class VerifiedEarliestPlacement:
    course_code: str
    plans: tuple[str, str]
    earliest: tuple[tuple[int, int], tuple[int, int]]
    earlier_plan: str | None
    tie: bool


@dataclass(frozen=True, slots=True)
class VerifiedPlacementComparison:
    cells: tuple[VerifiedPlacementCell, ...]
    conclusions: tuple[VerifiedEarliestPlacement, ...] = ()


@dataclass(frozen=True, slots=True)
class VerifiedScopedResult:
    """One complete evidence result with its explicit factual ownership."""

    scope_kind: str
    scope: ResolvedScope
    course_code: str | None = None
    parent_course_code: str | None = None
    total_credits: int | float | None = None
    summary_facts: tuple[str, ...] = ()
    claims: tuple[Any, ...] = ()
    provenance: tuple[Any, ...] = ()


@dataclass(frozen=True, slots=True)
class VerifiedAlternativeSelection:
    """Complete canonical group membership and selection bounds with sources."""

    program: str
    catalog_key: str
    plan: str
    alternative_group_id: int
    member_course_codes: tuple[str, ...]
    minimum_choices: int
    maximum_choices: int
    provenance: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class VerifiedNumericComparisonSide:
    """One canonically resolved side and its deterministically verified value."""

    label: str
    course_code: str | None
    course_name: str | None
    value: int | float


@dataclass(frozen=True, slots=True)
class VerifiedNumericComparison:
    """Verified numeric comparison facts; no presentation or model inference."""

    measure: str
    requested_operation: str
    actual_relation: str
    left: VerifiedNumericComparisonSide
    right: VerifiedNumericComparisonSide
    absolute_difference: int | float


__all__ = [
    "AGGREGATION_FUNCTIONS",
    "COMPARISON_OPERAND_KEYS",
    "COMPARISON_OPERATIONS",
    "COMPARISON_MEASURES",
    "PLAN_SELECTORS",
    "FAILURE_CATEGORIES",
    "FILTER_FIELDS",
    "FILTER_OPERATORS",
    "GROUP_DIMENSIONS",
    "INTERNAL_STATUSES",
    "LEGACY_STATUS_FOR_INTERNAL",
    "MAX_HINT_LEN",
    "MAX_COURSE_SET_MEMBERS",
    "MAX_ORDINAL",
    "MAX_TEXT_LEN",
    "MEASURES",
    "POLICY_TOPICS",
    "RANK_DIRECTIONS",
    "RELATIONS",
    "REQUESTED_FIELDS",
    "SEMANTIC_INTENT_VERSION",
    "SUBJECTS",
    "TARGET_KINDS",
    "TASKS",
    "VALID_TASK_SUBJECTS",
    "AggregationSpec",
    "ComparisonSpec",
    "RankingSpec",
    "ResolvedIntent",
    "ResolvedOperand",
    "ResolvedScope",
    "ResolvedTarget",
    "ScopeMention",
    "SemanticFilter",
    "SemanticIntent",
    "SemanticSchemaError",
    "SemanticTarget",
    "LiteralCourseReference",
    "VerifiedAlternativeSelection",
    "VerifiedResult",
    "VerifiedScopedResult",
    "VerifiedPlacementCell",
    "VerifiedEarliestPlacement",
    "VerifiedPlacementComparison",
    "VerifiedNumericComparison",
    "VerifiedNumericComparisonSide",
]
