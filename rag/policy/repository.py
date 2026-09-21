"""Read-only repository for policy facts and their canonical provenance."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from contextlib import closing
from pathlib import Path
from typing import Any


_REQUIRED_TABLES = frozenset(
    {
        "policy_facts",
        "policy_fact_provenance",
        "program_requirements",
        "program_requirement_provenance",
        "provenance",
    }
)


def _provenance(row: sqlite3.Row) -> dict[str, Any]:
    return {
        key: row[key]
        for key in (
            "provenance_id",
            "source_document_key",
            "program",
            "source_filename",
            "source_page",
            "document_page",
            "document_category",
            "source_uri",
            "source_locator",
            "excerpt",
        )
    }


def _ensure_tables(connection: sqlite3.Connection) -> None:
    found = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    missing = _REQUIRED_TABLES - found
    if missing:
        raise ValueError(f"policy database is missing tables: {sorted(missing)}")


def _with_provenance(
    connection: sqlite3.Connection,
    rows: Iterable[sqlite3.Row],
    join_table: str,
    key_name: str,
    expected_category: str,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        record = dict(row)
        provenance_rows = connection.execute(
            f"""
            SELECT p.*
            FROM {join_table} AS link
            JOIN provenance AS p ON p.provenance_id = link.provenance_id
            WHERE link.{key_name} = ?
            ORDER BY p.provenance_id
            """,
            (record[key_name],),
        )
        references = [_provenance(provenance_row) for provenance_row in provenance_rows]
        if not references or any(
            reference["document_category"] != expected_category for reference in references
        ):
            raise ValueError(f"policy record {record[key_name]} has no provenance")
        record["provenance"] = tuple(references)
        result.append(record)
    return result


def _open(db_path: str | Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{Path(db_path).resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    _ensure_tables(connection)
    return connection


def fetch_policy_facts(
    db_path: str | Path,
    *,
    category: str,
    fact_key_contains: str | None = None,
    condition: str | None = None,
    context: str | None = None,
    context_any: bool = False,
    fact_key: str | None = None,
) -> tuple[dict[str, Any], ...]:
    with closing(_open(db_path)) as connection:
        clauses = ["category = ?"]
        params: list[Any] = [category]
        if fact_key_contains is not None:
            clauses.append("fact_key LIKE ?")
            params.append(f"%{fact_key_contains}%")
        if fact_key is not None:
            clauses.append("fact_key = ?")
            params.append(fact_key)
        if condition is not None:
            clauses.append("condition = ?")
            params.append(condition)
        if context_any:
            pass
        elif context is None:
            clauses.append("context IS NULL")
        else:
            clauses.append("context = ?")
            params.append(context)
        rows = connection.execute(
            "SELECT * FROM policy_facts WHERE " + " AND ".join(clauses) + " ORDER BY fact_id",
            params,
        )
        return tuple(
            _with_provenance(
                connection,
                rows,
                "policy_fact_provenance",
                "fact_id",
                "rule",
            )
        )


def fetch_program_requirement(db_path: str | Path, program: str) -> dict[str, Any] | None:
    with closing(_open(db_path)) as connection:
        rows = connection.execute(
            """
            SELECT * FROM program_requirements
            WHERE program_code = ? AND requirement_type = 'total_program_credits'
            """,
            (program,),
        )
        records = _with_provenance(
            connection,
            rows,
            "program_requirement_provenance",
            "requirement_id",
            "program_requirement",
        )
        if len(records) > 1:
            raise ValueError("program requirement is ambiguous")
        return records[0] if records else None


__all__ = ["fetch_policy_facts", "fetch_program_requirement"]
