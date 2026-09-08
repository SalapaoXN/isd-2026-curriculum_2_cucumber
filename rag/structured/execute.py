"""Read-only execution of guarded structured SQLite queries."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from .guard_sql import guard_sql


def validate_readonly_sql(
    db_path: str | Path,
    sql: str,
) -> None:
    """Validate one guarded read-only query without executing it."""
    safe_sql = guard_sql(sql)
    database_path = Path(db_path)
    if not database_path.exists():
        raise FileNotFoundError(database_path)

    read_only_uri = f"{database_path.resolve().as_uri()}?mode=ro"
    with closing(sqlite3.connect(read_only_uri, uri=True)) as connection:
        connection.execute(f"EXPLAIN QUERY PLAN {safe_sql}").fetchall()


def execute_readonly(
    db_path: str | Path,
    sql: str,
) -> tuple[list[str], list[tuple[Any, ...]]]:
    """Execute one guarded SELECT/WITH query without opening a writable DB."""
    safe_sql = guard_sql(sql)
    database_path = Path(db_path)
    if not database_path.exists():
        raise FileNotFoundError(database_path)

    read_only_uri = f"{database_path.resolve().as_uri()}?mode=ro"
    with closing(sqlite3.connect(read_only_uri, uri=True)) as connection:
        cursor = connection.execute(safe_sql)
        columns = [description[0] for description in cursor.description or ()]
        rows = cursor.fetchall()
    return columns, rows


__all__ = ["execute_readonly", "validate_readonly_sql"]
