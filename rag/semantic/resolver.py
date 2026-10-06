"""Deterministic entity/scope resolution for semantic mode.

Only deterministic resolvers create canonical fields. The interpreter's
raw strings are evidence of what the student wrote, never identity:

- 0 canonical matches → unresolved / clarification / insufficient
- 1 canonical match → accept
- N matches → clarify, or disambiguate through authoritative scope only

Normalized hints only GENERATE lookup candidates; in v1 they are never
accepted as identity (see allow_hint_candidates=False default). Programs,
catalogs, and plans are validated against canonical tables. A user-typed
8-digit code is user-provided text until the database confirms it.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any

from rag.semantic.context import MergedContext
from rag.semantic.schema import (
    ResolvedIntent,
    ResolvedOperand,
    ResolvedScope,
    ResolvedTarget,
    SemanticIntent,
)
from rag.structured.queries import exact_course_candidates

_COURSE_CODE_RE = re.compile(r"^\d{8}$")


def _connect_ro(db_path: str | Path) -> sqlite3.Connection:
    uri = f"{Path(db_path).resolve().as_uri()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def canonical_program(db_path: str | Path, raw: str | None) -> str | None:
    """Return the canonical program code for user text, or None."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        connection = _connect_ro(db_path)
        try:
            rows = connection.execute(
                "SELECT program_code FROM programs"
            ).fetchall()
        finally:
            connection.close()
    except sqlite3.Error:
        return None
    wanted = raw.strip().casefold()
    for row in rows:
        code = row["program_code"]
        if isinstance(code, str) and code.strip().casefold() == wanted:
            return code.strip()
    return None


def canonical_catalog_key(
    db_path: str | Path, raw: str | None, program: str | None = None
) -> str | None:
    """Resolve an explicit catalog key or academic-year token to one edition.

    Exact catalog_key wins. A year token (e.g. ``2560``) is accepted only
    when the canonical DB maps it to exactly one catalog for the supplied
    program. No newest/oldest preference is inferred.
    """
    if not isinstance(raw, str) or not raw.strip() or len(raw.strip()) > 128:
        return None
    try:
        connection = _connect_ro(db_path)
        try:
            rows = connection.execute(
                "SELECT catalog_key FROM catalogs WHERE lower(trim(catalog_key)) = ?",
                (raw.strip().casefold(),),
            ).fetchall()
            if not rows and raw.strip().isdigit() and program is not None:
                rows = connection.execute(
                    """SELECT DISTINCT c.catalog_key FROM catalogs c
                       JOIN programs p ON p.catalog_id = c.catalog_id
                       WHERE trim(c.academic_year) = ?
                         AND lower(trim(p.program_code)) = ?""",
                    (raw.strip(), program.strip().casefold()),
                ).fetchall()
        finally:
            connection.close()
    except sqlite3.Error:
        return None
    if len(rows) != 1 or not rows[0]["catalog_key"]:
        return None
    return str(rows[0]["catalog_key"]).strip()


def valid_plan(
    db_path: str | Path,
    plan: str | None,
    program: str | None,
    catalog_key: str | None,
) -> str | None:
    """Return the plan key when canonical data confirms it, else None."""
    if not isinstance(plan, str) or not plan.strip():
        return None
    try:
        connection = _connect_ro(db_path)
        try:
            if catalog_key is not None:
                rows = connection.execute(
                    """SELECT cp.plan_key FROM curriculum_plans cp
                       JOIN catalogs c ON c.catalog_id = cp.catalog_id
                       WHERE lower(trim(c.catalog_key)) = ?
                         AND lower(trim(cp.plan_key)) = ?""",
                    (catalog_key.strip().casefold(), plan.strip().casefold()),
                ).fetchall()
            elif program is not None:
                rows = connection.execute(
                    """SELECT DISTINCT cp.plan_key FROM curriculum_plans cp
                       JOIN programs p ON p.program_id = cp.program_id
                       WHERE lower(trim(p.program_code)) = ?
                         AND lower(trim(cp.plan_key)) = ?""",
                    (program.strip().casefold(), plan.strip().casefold()),
                ).fetchall()
            else:
                return None
        finally:
            connection.close()
    except sqlite3.Error:
        return None
    if len(rows) != 1 or not rows[0]["plan_key"]:
        return None
    return str(rows[0]["plan_key"]).strip()


def _candidate_identity(candidate: dict[str, Any]) -> tuple[str | None, str | None]:
    code = candidate.get("course_code")
    name = candidate.get("name_en") or candidate.get("name_th")
    code = code.strip() if isinstance(code, str) and code.strip() else None
    name = name.strip() if isinstance(name, str) and name.strip() else None
    return code, name


def _lookup_literal(
    db_path: str | Path,
    raw_text: str,
    hint: str | None,
    program: str | None,
    catalog_key: str | None,
    *,
    allow_hint_candidates: bool = False,
) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    """Return (accepted_candidates, hint_candidates_for_trace)."""
    hint_candidates: tuple[str, ...] = ()
    try:
        if _COURSE_CODE_RE.fullmatch(raw_text.strip()):
            direct = exact_course_candidates(
                db_path,
                course_code=raw_text.strip(),
                program=program,
                catalog_key=catalog_key,
            )
        else:
            direct = exact_course_candidates(
                db_path,
                course_name=raw_text.strip(),
                program=program,
                catalog_key=catalog_key,
                exact_title=True,
            )
    except Exception:
        return [], ()
    if hint and allow_hint_candidates:
        hint_candidates = (hint.strip(),)
    return list(direct), hint_candidates


def _resolve_ordinal(
    ordinal: int, result_courses: tuple[dict[str, Any], ...]
) -> dict[str, Any] | None:
    if not isinstance(ordinal, int) or ordinal < 1:
        return None
    if not result_courses or ordinal > len(result_courses):
        return None
    entry = result_courses[ordinal - 1]
    code = entry.get("course_code") if isinstance(entry, dict) else None
    if not isinstance(code, str) or not code.strip():
        return None
    return {"course_code": code.strip()}


def _side_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def resolve_comparison_operand(
    db_path: str | Path,
    side: tuple[tuple[str, Any], ...],
    default_program: str | None,
    default_catalog_key: str | None,
) -> ResolvedOperand:
    """Resolve ONE comparison side independently (never merged across sides).

    Each side validates its own program/catalog/plan against canonical
    tables and its own course mention against canonical identities scoped
    to that side. Shared program/catalog/year values may coincide across
    sides, but plan scope and course identity are never shared: no
    cross-plan evidence leakage is possible at resolution time.
    """
    fields = dict(side)
    program = canonical_program(db_path, fields.get("program", default_program))
    if fields.get("program", default_program) is not None and program is None:
        return ResolvedOperand(unresolved=True, reason="unknown operand program")
    catalog_key = (
            canonical_catalog_key(
                db_path,
                fields.get("catalog", default_catalog_key),
                program,
            )
        if (fields.get("catalog", default_catalog_key) is not None)
        else None
    )
    if fields.get("catalog", default_catalog_key) is not None and catalog_key is None:
        return ResolvedOperand(unresolved=True, reason="unknown operand catalog")
    plan = valid_plan(db_path, fields.get("plan"), program, catalog_key)
    if fields.get("plan") is not None and plan is None:
        return ResolvedOperand(unresolved=True, reason="unknown operand plan")
    years: tuple[int, ...] = ()
    semesters: tuple[int, ...] = ()
    year = _side_int(fields.get("year"))
    semester = _side_int(fields.get("semester"))
    if fields.get("year") is not None and not (isinstance(year, int) and 1 <= year <= 5):
        return ResolvedOperand(unresolved=True, reason="bad operand year")
    if fields.get("semester") is not None and not (
        isinstance(semester, int) and 1 <= semester <= 2
    ):
        return ResolvedOperand(unresolved=True, reason="bad operand semester")
    if year is not None:
        years = (year,)
    if semester is not None:
        semesters = (semester,)
    scope = ResolvedScope(
        program=program, catalog_key=catalog_key, plan=plan,
        years=years, semesters=semesters,
    )
    course_raw = fields.get("course")
    if course_raw is None:
        return ResolvedOperand(scope=scope, unresolved=False)
    if not isinstance(course_raw, str) or not course_raw.strip():
        return ResolvedOperand(scope=scope, unresolved=True, reason="empty operand course")
    try:
        if _COURSE_CODE_RE.fullmatch(course_raw.strip()):
            candidates = exact_course_candidates(
                db_path, course_code=course_raw.strip(),
                program=program, catalog_key=catalog_key)
        else:
            candidates = exact_course_candidates(
                db_path, course_name=course_raw.strip(),
                program=program, catalog_key=catalog_key, exact_title=True)
    except Exception:
        return ResolvedOperand(scope=scope, unresolved=True, reason="operand lookup failed")
    identities = {_candidate_identity(c) for c in candidates}
    identities.discard((None, None))
    if len(identities) != 1:
        return ResolvedOperand(
            scope=scope, unresolved=True,
            reason="operand course unresolved or ambiguous")
    code, name = next(iter(identities))
    return ResolvedOperand(
        scope=scope,
        target=ResolvedTarget(kind="literal", course_code=code, course_name=name,
                              program=program, catalog_key=catalog_key),
        unresolved=False,
    )


def resolve_semantic_intent(
    db_path: str | Path,
    intent: SemanticIntent,
    merged: MergedContext,
    *,
    allow_hint_candidates: bool = False,
) -> ResolvedIntent:
    """Bind linguistic intent to canonical scope/identity, fact-free."""
    program = canonical_program(db_path, merged.program)
    if merged.program is not None and program is None:
        return ResolvedIntent(
            intent=intent,
            needs_clarification=True,
            clarification_reason="unknown program scope",
        )
    catalog_key = (
        canonical_catalog_key(db_path, merged.catalog_key, program)
        if merged.catalog_key is not None
        else None
    )
    if merged.catalog_key is not None and catalog_key is None:
        return ResolvedIntent(
            intent=intent,
            needs_clarification=True,
            clarification_reason="unknown catalog scope",
        )
    plan = valid_plan(db_path, merged.plan, program, catalog_key)
    if plan is None and merged.plan_hint is not None:
        # Normalized plan-hint path: the interpreter's canonical-key
        # proposal is accepted only on exact deterministic match against
        # canonical plan data (0 → fail, 1 → accept, ambiguous → fail).
        # No Thai alias table exists anywhere in this path.
        plan = valid_plan(db_path, merged.plan_hint, program, catalog_key)
    if merged.plan is not None and plan is None:
        return ResolvedIntent(
            intent=intent,
            needs_clarification=True,
            clarification_reason="unknown plan scope",
        )
    scope = ResolvedScope(
        program=program,
        catalog_key=catalog_key,
        plan=plan,
        years=tuple(merged.years),
        semesters=tuple(merged.semesters),
    )

    target = intent.target
    if (
        intent.subject == "program"
        and target.kind == "literal"
        and program is not None
        and isinstance(target.raw_text, str)
        and target.raw_text.strip().casefold() == program.casefold()
    ):
        # Program-as-target rescue: the explicit program mention IS the
        # subject being asked about (e.g. "<program> มีกี่หน่วยกิต"), not a
        # course. The program lives in scope; no course lookup is attempted.
        # Context-only programs never match (no raw text), and course
        # questions never enter (subject must be program).
        return ResolvedIntent(intent=intent, scope=scope)
    if target.kind == "none":
        return ResolvedIntent(intent=intent, scope=scope)
    if target.kind == "literal":
        raw = (target.raw_text or "").strip()
        if not raw:
            return ResolvedIntent(
                intent=intent,
                needs_clarification=True,
                clarification_reason="empty target reference",
            )
        candidates, hint_candidates = _lookup_literal(
            db_path,
            raw,
            target.normalized_hint,
            program,
            catalog_key,
            allow_hint_candidates=allow_hint_candidates,
        )
        identities = {_candidate_identity(c) for c in candidates}
        identities.discard((None, None))
        if not identities:
            return ResolvedIntent(
                intent=intent,
                scope=scope,
                hint_candidates=hint_candidates,
                needs_clarification=True,
                clarification_reason="target does not resolve to canonical data",
            )
        if len(identities) > 1:
            if program is None and catalog_key is None:
                return ResolvedIntent(
                    intent=intent,
                    scope=scope,
                    hint_candidates=hint_candidates,
                    needs_clarification=True,
                    clarification_reason="ambiguous target across scopes",
                )
            return ResolvedIntent(
                intent=intent,
                scope=scope,
                hint_candidates=hint_candidates,
                needs_clarification=True,
                clarification_reason="ambiguous target within scope",
            )
        code, name = next(iter(identities))
        return ResolvedIntent(
            intent=intent,
            scope=scope,
            target=ResolvedTarget(
                kind="literal",
                course_code=code,
                course_name=name,
                program=program,
                catalog_key=catalog_key,
            ),
            hint_candidates=hint_candidates,
        )
    if target.kind == "current_course":
        focus = merged.focus_course
        if focus is None:
            return ResolvedIntent(
                intent=intent,
                scope=scope,
                needs_clarification=True,
                clarification_reason="no validated current course in context",
            )
        code = focus.get("course_code")
        try:
            confirmed = exact_course_candidates(
                db_path,
                course_code=code,
                program=program,
                catalog_key=catalog_key or focus.get("catalog_key"),
            )
        except Exception:
            confirmed = []
        if len(confirmed) != 1:
            return ResolvedIntent(
                intent=intent,
                scope=scope,
                needs_clarification=True,
                clarification_reason="context course no longer resolves uniquely",
            )
        resolved_code, resolved_name = _candidate_identity(confirmed[0])
        return ResolvedIntent(
            intent=intent,
            scope=scope,
            target=ResolvedTarget(
                kind="current_course",
                course_code=resolved_code,
                course_name=resolved_name,
                program=program,
                catalog_key=catalog_key,
            ),
        )
    if target.kind == "result_ordinal":
        entry = _resolve_ordinal(target.ordinal or 0, merged.result_courses)
        if entry is None:
            return ResolvedIntent(
                intent=intent,
                scope=scope,
                needs_clarification=True,
                clarification_reason="ordinal has no validated referent",
            )
        try:
            confirmed = exact_course_candidates(
                db_path,
                course_code=entry["course_code"],
                program=program,
                catalog_key=catalog_key,
            )
        except Exception:
            confirmed = []
        if len(confirmed) != 1:
            return ResolvedIntent(
                intent=intent,
                scope=scope,
                needs_clarification=True,
                clarification_reason="ordinal referent no longer resolves uniquely",
            )
        resolved_code, resolved_name = _candidate_identity(confirmed[0])
        return ResolvedIntent(
            intent=intent,
            scope=scope,
            target=ResolvedTarget(
                kind="result_ordinal",
                course_code=resolved_code,
                course_name=resolved_name,
                program=program,
                catalog_key=catalog_key,
            ),
        )
    if target.kind == "previous_result_set":
        if not merged.result_courses:
            return ResolvedIntent(
                intent=intent,
                scope=scope,
                needs_clarification=True,
                clarification_reason="no validated previous result set in context",
            )
        return ResolvedIntent(intent=intent, scope=scope)
    return ResolvedIntent(
        intent=intent,
        needs_clarification=True,
        clarification_reason="unsupported target reference",
    )


__all__ = [
    "canonical_catalog_key",
    "canonical_program",
    "resolve_semantic_intent",
    "valid_plan",
]
