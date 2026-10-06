"""Deterministic aggregate-evidence verification for SQL rescue (HSQL-6).

Verifies one narrow supported aggregate shape — a single-term credit total —
by recomputing it from canonical placements with the same counting semantics
owned by ``rag.structured.queries``. Never treats the LLM-generated aggregate
value as authority: the recomputed canonical total must equal the SQL value,
every counted contributor must have determinable credits, and every
contributor must carry canonical provenance.

Model-free, read-only, and isolated from routing and answer prose.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping

from rag.structured.queries import (
    _alternative_members,
    _credits_for_placement,
    _decimal_credits,
    _merge_provenance,
    _placement_rows,
    _provenance_for,
)


@dataclass(frozen=True, slots=True)
class VerifiedAggregateEvidence:
    """Immutable outcome of aggregate-evidence verification."""

    status: str
    aggregate_kind: str | None
    value: int | float | None
    provenance: tuple[dict[str, Any], ...]
    contributing_rows: int


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_non_empty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _known_placement_credits(
    connection: sqlite3.Connection, placement: Mapping[str, Any]
) -> Decimal | None:
    """Mirror canonical counting, returning None for unknown credits.

    Branches match ``_credits_for_placement`` exactly; the difference is
    that indeterminable credits fail closed instead of counting as zero.
    """
    override = placement.get("credits_override")
    if override not in (None, ""):
        return _decimal_credits(override)
    if placement.get("alternative_group_id") is None:
        return _decimal_credits(placement.get("credits"))
    members = _alternative_members(
        connection, int(placement["alternative_group_id"])
    )
    choices = int(placement.get("minimum_choices") or 1)
    subtotal = Decimal(0)
    for member in members[:choices]:
        value = _decimal_credits(member.get("credits"))
        if value is None:
            return None
        subtotal += value
    return subtotal


def _component_references(
    connection: sqlite3.Connection, placement: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Collect canonical provenance for one counted placement."""
    placement_references = _provenance_for(
        connection,
        "plan_placement_provenance",
        "placement_id",
        int(placement["placement_id"]),
    )
    group_id = placement.get("alternative_group_id")
    if group_id is None:
        course_id = placement.get("course_id")
        course_references = (
            _provenance_for(connection, "course_provenance", "course_id", int(course_id))
            if isinstance(course_id, int) and not isinstance(course_id, bool)
            else []
        )
        return _merge_provenance(placement_references, course_references)
    alternative_courses = _alternative_members(connection, int(group_id))
    return _merge_provenance(
        placement_references,
        _provenance_for(
            connection,
            "alternative_group_provenance",
            "alternative_group_id",
            int(group_id),
        ),
        *(member["provenance"] for member in alternative_courses),
    )


def verify_aggregate_evidence(
    db_path: str | Path,
    *,
    rows: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
    columns: list[str] | tuple[str, ...],
    program: str | None,
    catalog_key: str | None,
    plan: str | None,
    years: tuple[int, ...] | list[int],
    semesters: tuple[int, ...] | list[int],
    query_spec: Any | None,
) -> VerifiedAggregateEvidence:
    """Verify a single-term credit-total aggregate row against canonical data."""
    _ = columns
    row_list = list(rows or [])
    if len(row_list) != 1 or not isinstance(row_list[0], Mapping):
        return VerifiedAggregateEvidence(
            status="insufficient_evidence",
            aggregate_kind=None,
            value=None,
            provenance=(),
            contributing_rows=0,
        )
    row = row_list[0]
    if "total_credits" not in row or not _is_number(row["total_credits"]):
        return VerifiedAggregateEvidence(
            status="insufficient_evidence",
            aggregate_kind=None,
            value=None,
            provenance=(),
            contributing_rows=0,
        )
    if not (
        _is_non_empty_text(program)
        and _is_non_empty_text(catalog_key)
        and _is_non_empty_text(plan)
    ):
        return VerifiedAggregateEvidence(
            status="insufficient_evidence",
            aggregate_kind=None,
            value=None,
            provenance=(),
            contributing_rows=0,
        )
    years = tuple(years) if isinstance(years, (tuple, list)) else ()
    semesters = tuple(semesters) if isinstance(semesters, (tuple, list)) else ()
    if (
        len(years) != 1
        or len(semesters) != 1
        or not _is_number(years[0])
        or isinstance(years[0], bool)
        or not _is_number(semesters[0])
        or isinstance(semesters[0], bool)
        or not 1 <= int(years[0]) <= 5
        or not 1 <= int(semesters[0]) <= 2
    ):
        return VerifiedAggregateEvidence(
            status="insufficient_evidence",
            aggregate_kind=None,
            value=None,
            provenance=(),
            contributing_rows=0,
        )
    year, semester = int(years[0]), int(semesters[0])
    operations = tuple(getattr(query_spec, "operations", ()) or ())
    if "sum_credits" not in operations:
        normalized = getattr(query_spec, "normalized_question", "") or ""
        credit_word = any(
            token in normalized.casefold()
            for token in ("หน่วย", "เครดิต", "credit")
        )
        if not (
            credit_word
            and "list" not in operations
            and "count" not in operations
        ):
            return VerifiedAggregateEvidence(
                status="insufficient_evidence",
                aggregate_kind=None,
                value=None,
                provenance=(),
                contributing_rows=0,
            )

    database_uri = f"file:{Path(db_path)}?mode=ro"
    with closing(sqlite3.connect(database_uri, uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        plan_rows = connection.execute(
            """SELECT plan_id FROM curriculum_plans
               JOIN programs ON programs.program_id = curriculum_plans.program_id
               JOIN catalogs ON catalogs.catalog_id = curriculum_plans.catalog_id
               WHERE programs.program_code = ?
                 AND curriculum_plans.plan_key = ?
                 AND lower(trim(catalogs.catalog_key)) = ?""",
            (
                str(program).strip().upper(),
                str(plan).strip().lower(),
                str(catalog_key).strip().casefold(),
            ),
        ).fetchall()
        if len(plan_rows) != 1:
            return VerifiedAggregateEvidence(
                status="insufficient_evidence",
                aggregate_kind=None,
                value=None,
                provenance=(),
                contributing_rows=0,
            )
        placements = [
            dict(placement)
            for placement in _placement_rows(
                connection, int(plan_rows[0]["plan_id"]), year, semester
            )
        ]
        if not placements:
            return VerifiedAggregateEvidence(
                status="insufficient_evidence",
                aggregate_kind=None,
                value=None,
                provenance=(),
                contributing_rows=0,
            )
        total = Decimal(0)
        reference_groups: list[list[dict[str, Any]]] = []
        for placement in placements:
            counted = _credits_for_placement(connection, placement)
            known = _known_placement_credits(connection, placement)
            if known is None or known != counted:
                return VerifiedAggregateEvidence(
                    status="insufficient_evidence",
                    aggregate_kind=None,
                    value=None,
                    provenance=(),
                    contributing_rows=0,
                )
            references = _component_references(connection, placement)
            if not references:
                return VerifiedAggregateEvidence(
                    status="insufficient_evidence",
                    aggregate_kind=None,
                    value=None,
                    provenance=(),
                    contributing_rows=0,
                )
            total += counted
            reference_groups.append(references)

    if total != Decimal(str(row["total_credits"])):
        return VerifiedAggregateEvidence(
            status="insufficient_evidence",
            aggregate_kind=None,
            value=None,
            provenance=(),
            contributing_rows=0,
        )
    merged: list[dict[str, Any]] = []
    seen: set[int] = set()
    for references in reference_groups:
        for reference in references:
            provenance_id = reference["provenance_id"]
            if provenance_id in seen:
                continue
            seen.add(provenance_id)
            merged.append(dict(reference))
    value: int | float = (
        int(total) if total == total.to_integral_value() else float(total)
    )
    return VerifiedAggregateEvidence(
        status="complete",
        aggregate_kind="semester_credit_total",
        value=value,
        provenance=tuple(merged),
        contributing_rows=len(placements),
    )


__all__ = ["VerifiedAggregateEvidence", "verify_aggregate_evidence"]
