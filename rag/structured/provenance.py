"""Deterministic provenance hydration for SQL result rows (HSQL-1).

Derives canonical provenance from integer entity IDs already present in
executed SQL rows. Model-free: it only reads canonical provenance link
tables and never treats row-supplied values as authority.

This module changes no routing or answer behavior on its own; it only
establishes the evidence boundary for a future optional rescue step.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from rag.structured.queries import _provenance_for


@dataclass(frozen=True, slots=True)
class SqlRowProvenanceResult:
    """Immutable outcome of hydrating provenance for one SQL row set."""

    status: str
    provenance: tuple[dict[str, Any], ...]
    covered_rows: int
    total_rows: int


# Recognized canonical identity columns in deterministic first-seen order,
# each paired with its canonical provenance link table and entity column.
# Only these fixed pairs are ever queried; table/column names never come
# from row data.
_ID_SOURCES: tuple[tuple[str, str, str], ...] = (
    ("course_id", "course_provenance", "course_id"),
    ("placement_id", "plan_placement_provenance", "placement_id"),
    ("prerequisite_id", "prerequisite_provenance", "prerequisite_id"),
    ("requirement_id", "program_requirement_provenance", "requirement_id"),
    ("fact_id", "policy_fact_provenance", "fact_id"),
    ("alternative_group_id", "alternative_group_provenance", "alternative_group_id"),
)


def _valid_entity_id(value: Any) -> int | None:
    """Return the canonical ID, or None when the value is not one.

    Only positive plain integers qualify; bools, floats, strings, and
    non-positive values are never canonical entity IDs.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if value < 1:
        return None
    return value


def hydrate_sql_row_provenance(
    db_path: str | Path,
    rows: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
) -> SqlRowProvenanceResult:
    """Hydrate canonical provenance for executed SQL rows.

    A row is covered only when at least one recognized canonical ID in
    that row resolves to non-empty canonical provenance. All rows must
    be covered for ``complete``; an empty row set is ``valid_empty``.
    Row-supplied ``provenance`` fields are never read.
    """
    row_list = list(rows)
    total_rows = len(row_list)
    if total_rows == 0:
        return SqlRowProvenanceResult(
            status="valid_empty", provenance=(), covered_rows=0, total_rows=0
        )

    merged: list[dict[str, Any]] = []
    seen: set[int] = set()
    covered_rows = 0
    connection = sqlite3.connect(f"file:{Path(db_path)}?mode=ro", uri=True)
    try:
        connection.row_factory = sqlite3.Row
        for row in row_list:
            row_covered = False
            if isinstance(row, Mapping):
                for column, link_table, entity_column in _ID_SOURCES:
                    entity_id = _valid_entity_id(row.get(column))
                    if entity_id is None:
                        continue
                    references = _provenance_for(
                        connection, link_table, entity_column, entity_id
                    )
                    if not references:
                        continue
                    row_covered = True
                    for reference in references:
                        provenance_id = reference["provenance_id"]
                        if provenance_id in seen:
                            continue
                        seen.add(provenance_id)
                        merged.append(dict(reference))
            if row_covered:
                covered_rows += 1
    finally:
        connection.close()

    if covered_rows == total_rows:
        return SqlRowProvenanceResult(
            status="complete",
            provenance=tuple(merged),
            covered_rows=covered_rows,
            total_rows=total_rows,
        )
    return SqlRowProvenanceResult(
        status="insufficient_evidence",
        provenance=tuple(merged),
        covered_rows=covered_rows,
        total_rows=total_rows,
    )


__all__ = ["SqlRowProvenanceResult", "hydrate_sql_row_provenance"]
