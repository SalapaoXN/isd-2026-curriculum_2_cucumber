"""Read-only execution of guarded structured SQLite queries."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from contextlib import closing
from pathlib import Path
from typing import Any

from .guard_sql import guard_sql


def _install_course_scope(
    connection: sqlite3.Connection,
    course_identities: Iterable[
        tuple[str, str] | tuple[str | None, str, str]
    ],
) -> None:
    identities = list(course_identities)
    if len(identities) > 50:
        raise ValueError("course scope exceeds the context row bound")
    normalized: list[tuple[str | None, str, str]] = []
    for identity in identities:
        if not isinstance(identity, tuple) or len(identity) not in {2, 3}:
            raise ValueError("course scope identities must contain program and course code")
        if len(identity) == 2:
            raw_catalog_key = None
            raw_program, raw_course_code = identity
        else:
            raw_catalog_key, raw_program, raw_course_code = identity
        if (
            (raw_catalog_key is not None and not isinstance(raw_catalog_key, str))
            or not isinstance(raw_program, str)
            or not raw_program.strip()
            or not isinstance(raw_course_code, str)
            or not raw_course_code.strip()
        ):
            raise ValueError("course scope identities must contain valid catalog, program, and course code")
        catalog_key = raw_catalog_key.strip() if raw_catalog_key is not None else None
        if catalog_key == "" or (catalog_key is not None and len(catalog_key) > 128):
            raise ValueError("course scope catalog_key is invalid")
        identity_key = (
            catalog_key.casefold() if catalog_key is not None else None,
            raw_program.strip().casefold(),
            raw_course_code.strip().casefold(),
        )
        if identity_key not in normalized:
            normalized.append(identity_key)
    if not normalized:
        raise ValueError("course scope must contain at least one identity")

    connection.execute(
        "CREATE TEMP TABLE _qa_requested_scope ("
        "scope_id INTEGER PRIMARY KEY, catalog_key TEXT, "
        "program TEXT NOT NULL, course_code TEXT NOT NULL)"
    )
    connection.executemany(
        "INSERT INTO _qa_requested_scope (scope_id, catalog_key, program, course_code) "
        "VALUES (?, ?, ?, ?)",
        [
            (scope_id, catalog_key, program, course_code)
            for scope_id, (catalog_key, program, course_code) in enumerate(normalized)
        ],
    )
    candidates = connection.execute(
        "SELECT requested.scope_id, course.catalog_id, catalog.catalog_key, "
        "course.course_id, lower(trim(view.program)), lower(trim(view.course_code)) "
        "FROM _qa_requested_scope AS requested "
        "JOIN main.v_plan_courses AS view "
        "ON lower(trim(view.program)) = requested.program "
        "AND lower(trim(view.course_code)) = requested.course_code "
        "JOIN main.courses AS course ON course.course_id = view.course_id "
        "JOIN main.catalogs AS catalog ON catalog.catalog_id = course.catalog_id"
    ).fetchall()
    candidates_by_scope: dict[int, list[tuple[Any, ...]]] = {}
    for row in candidates:
        candidates_by_scope.setdefault(row[0], []).append(row)

    resolved_scope: list[tuple[int, str | None, str, str]] = []
    allowed_course_ids: set[int] = set()
    for scope_id, (requested_key, program, course_code) in enumerate(normalized):
        matches = candidates_by_scope.get(scope_id, [])
        if requested_key is not None:
            matches = [
                row
                for row in matches
                if isinstance(row[2], str) and row[2].strip().casefold() == requested_key
            ]
        catalog_ids = {row[1] for row in matches}
        if not matches:
            raise ValueError("course scope contains an unresolved catalog/program/course identity")
        if requested_key is None and len(catalog_ids) > 1:
            raise ValueError("legacy course identity is ambiguous across catalogs")
        if len(catalog_ids) != 1:
            raise ValueError("catalog_key does not resolve to one canonical catalog")
        catalog_id = next(iter(catalog_ids))
        matching_catalog = [row for row in matches if row[1] == catalog_id]
        catalog_keys = {row[2] for row in matching_catalog}
        if len(catalog_keys) != 1:
            raise ValueError("canonical catalog has conflicting catalog keys")
        resolved_scope.append((catalog_id, next(iter(catalog_keys)), program, course_code))
        allowed_course_ids.update(row[3] for row in matching_catalog)

    connection.execute(
        "CREATE TEMP TABLE _qa_scope_pairs ("
        "catalog_id INTEGER NOT NULL, catalog_key TEXT, program TEXT NOT NULL, "
        "course_code TEXT NOT NULL, PRIMARY KEY (catalog_id, program, course_code))"
    )
    connection.executemany(
        "INSERT INTO _qa_scope_pairs (catalog_id, catalog_key, program, course_code) "
        "VALUES (?, ?, ?, ?)",
        resolved_scope,
    )

    connection.execute(
        "CREATE TEMP TABLE _qa_scope_ids (course_id INTEGER PRIMARY KEY)"
    )
    connection.executemany(
        "INSERT OR IGNORE INTO _qa_scope_ids (course_id) VALUES (?)",
        [(course_id,) for course_id in allowed_course_ids],
    )

    # TEMP views shadow canonical relations only for this query connection.
    # The main database remains opened read-only, and the predicates apply
    # before row filtering, joins, and aggregates in model-generated SQL.
    views = (
        (
            "v_plan_courses",
            "SELECT v.*, catalog.catalog_key FROM main.v_plan_courses AS v "
            "JOIN main.courses AS course ON course.course_id = v.course_id "
            "JOIN main.catalogs AS catalog ON catalog.catalog_id = course.catalog_id "
            "JOIN _qa_scope_pairs AS requested ON requested.catalog_id = course.catalog_id "
            "AND lower(trim(v.program)) = requested.program "
            "AND lower(trim(v.course_code)) = requested.course_code",
        ),
        (
            "courses",
            "SELECT c.*, catalog.catalog_key FROM main.courses AS c "
            "JOIN main.catalogs AS catalog ON catalog.catalog_id = c.catalog_id "
            "WHERE c.course_id IN (SELECT course_id FROM _qa_scope_ids)",
        ),
        (
            "catalogs",
            "SELECT c.* FROM main.catalogs AS c "
            "WHERE c.catalog_id IN (SELECT catalog_id FROM _qa_scope_pairs)",
        ),
        (
            "curriculum_plans",
            "SELECT p.* FROM main.curriculum_plans AS p "
            "WHERE p.plan_id IN (SELECT v.plan_id FROM main.v_plan_courses AS v "
            "JOIN main.courses AS course ON course.course_id = v.course_id "
            "JOIN _qa_scope_pairs AS requested ON requested.catalog_id = course.catalog_id "
            "AND lower(trim(v.program)) = requested.program "
            "AND lower(trim(v.course_code)) = requested.course_code)",
        ),
        (
            "programs",
            "SELECT p.* FROM main.programs AS p "
            "WHERE p.program_id IN (SELECT v.program_id FROM main.v_plan_courses AS v "
            "JOIN main.courses AS course ON course.course_id = v.course_id "
            "JOIN _qa_scope_pairs AS requested ON requested.catalog_id = course.catalog_id "
            "AND lower(trim(v.program)) = requested.program "
            "AND lower(trim(v.course_code)) = requested.course_code)",
        ),
        (
            "alternative_course_group_members",
            "SELECT m.* FROM main.alternative_course_group_members AS m "
            "WHERE m.course_id IN (SELECT course_id FROM _qa_scope_ids)",
        ),
        (
            "alternative_course_groups",
            "SELECT g.* FROM main.alternative_course_groups AS g "
            "WHERE EXISTS (SELECT 1 FROM main.alternative_course_group_members AS m "
            "JOIN _qa_scope_ids AS allowed ON allowed.course_id = m.course_id "
            "WHERE m.alternative_group_id = g.alternative_group_id)",
        ),
        (
            "plan_placements",
            "SELECT p.* FROM main.plan_placements AS p "
            "WHERE p.course_id IN (SELECT course_id FROM _qa_scope_ids) "
            "OR p.alternative_group_id IN (SELECT m.alternative_group_id "
            "FROM main.alternative_course_group_members AS m "
            "JOIN _qa_scope_ids AS allowed ON allowed.course_id = m.course_id)",
        ),
        (
            "prerequisites",
            "SELECT p.* FROM main.prerequisites AS p "
            "WHERE p.course_id IN (SELECT course_id FROM _qa_scope_ids)",
        ),
        (
            "v_prerequisite_edges",
            "SELECT e.* FROM main.v_prerequisite_edges AS e "
            "WHERE e.source_course_id IN (SELECT course_id FROM _qa_scope_ids)",
        ),
        (
            "v_semester_credits",
            "WITH counted_courses AS (SELECT * FROM v_plan_courses "
            "WHERE alternative_group_id IS NULL OR "
            "(alternative_member_order IS NOT NULL "
            "AND alternative_member_order <= minimum_choices)) "
            "SELECT plan_id, program, plan, year, semester, "
            "COALESCE(SUM(credit_units), 0) AS total_credits "
            "FROM counted_courses GROUP BY plan_id, program, plan, year, semester",
        ),
    )
    for name, query in views:
        connection.execute(f"CREATE TEMP VIEW {name} AS {query}")


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
    *,
    course_scope: Iterable[
        tuple[str, str] | tuple[str | None, str, str]
    ] | None = None,
) -> tuple[list[str], list[tuple[Any, ...]]]:
    """Execute one guarded SELECT/WITH query without opening a writable DB."""
    safe_sql = guard_sql(sql)
    database_path = Path(db_path)
    if not database_path.exists():
        raise FileNotFoundError(database_path)

    read_only_uri = f"{database_path.resolve().as_uri()}?mode=ro"
    with closing(sqlite3.connect(read_only_uri, uri=True)) as connection:
        if course_scope is not None:
            _install_course_scope(connection, course_scope)
        cursor = connection.execute(safe_sql)
        columns = [description[0] for description in cursor.description or ()]
        rows = cursor.fetchall()
    return columns, rows


__all__ = ["execute_readonly", "validate_readonly_sql"]
