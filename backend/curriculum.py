"""Read-only listing and filter options for indexed curriculum subjects."""

import json
import logging
import sqlite3
import unicodedata
from contextlib import closing
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from .rag_service import _db_path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_COURSE_NAME_ALIASES_PATH = (
    _PROJECT_ROOT / "data" / "corrections" / "source_verified_course_name_aliases.json"
)
_logger = logging.getLogger(__name__)


def _normalized_search_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


@lru_cache(maxsize=1)
def _course_name_alias_entries() -> tuple[dict[str, Any], ...]:
    try:
        document = json.loads(_COURSE_NAME_ALIASES_PATH.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return ()
    except (OSError, ValueError):
        _logger.warning("source-verified course-name alias artifact could not be read")
        return ()
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        _logger.warning("source-verified course-name alias artifact has an unsupported schema")
        return ()
    entries = document.get("aliases")
    if not isinstance(entries, list):
        _logger.warning("source-verified course-name alias artifact has no aliases list")
        return ()
    return tuple(entry for entry in entries if isinstance(entry, dict))


def _verified_course_name_aliases(search: str) -> list[dict[str, Any]]:
    normalized = _normalized_search_text(search)
    if not normalized:
        return []
    matches = []
    for entry in _course_name_alias_entries():
        observed_aliases = entry.get("observed_aliases")
        if not isinstance(observed_aliases, list):
            observed_aliases = []
        candidates = [entry.get("canonical_name"), *observed_aliases]
        if any(
            isinstance(candidate, str)
            and _normalized_search_text(candidate) == normalized
            for candidate in candidates
        ):
            if all(
                isinstance(entry.get(field), str) and entry[field].strip()
                for field in ("catalog_key", "program", "academic_year", "course_code")
            ):
                matches.append(entry)
    return matches


def matches_verified_course_name_alias(search: str, course: dict[str, Any]) -> bool:
    course_code = str(course.get("course_code") or "").strip().casefold()
    program = str(course.get("program_code") or "").strip().casefold()
    academic_year = str(course.get("academic_year") or "").strip()
    plans = {str(value).strip().casefold() for value in (course.get("plans") or [])}
    for entry in _verified_course_name_aliases(search):
        if (course_code != str(entry["course_code"]).strip().casefold()
                or program != str(entry["program"]).strip().casefold()
                or academic_year != str(entry["academic_year"]).strip()):
            continue
        plan = entry.get("plan")
        if isinstance(plan, str) and plan.strip() and plan.strip().casefold() not in plans:
            continue
        return True
    return False


def verified_course_name_in_text(text: str, course: dict[str, Any]) -> str | None:
    """Return a source-verified canonical name only when its full alias occurs in text."""
    text_key = _normalized_search_text(text)
    if not text_key:
        return None
    for entry in _course_name_alias_entries():
        canonical = entry.get("canonical_name")
        observed_aliases = entry.get("observed_aliases")
        candidates = [canonical, *(observed_aliases if isinstance(observed_aliases, list) else [])]
        for candidate in candidates:
            if (isinstance(candidate, str) and len(_normalized_search_text(candidate)) >= 5
                    and _normalized_search_text(candidate) in text_key
                    and matches_verified_course_name_alias(candidate, course)):
                return canonical if isinstance(canonical, str) and canonical.strip() else candidate
    return None


def _filter_clauses(
    values: dict[str, str], *, exclude: str = "",
    search_aliases: list[dict[str, Any]] | None = None,
) -> tuple[list[str], list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    for key, expr in (("program", "p.program_code"), ("academic_year", "cat.academic_year"),
                      ("category", "c.category"), ("course_type", "c.course_type")):
        if key != exclude and values.get(key):
            clauses.append(f"{expr}=? COLLATE NOCASE" if key == "program" else f"{expr}=?")
            params.append(values[key])
    if values.get("course_code") and exclude != "course_code":
        clauses.append("lower(c.course_code)=lower(?)")
        params.append(values["course_code"].strip())
    if values.get("search") and exclude != "search":
        search_terms = [
            "instr(lower(c.course_code),lower(?))>0",
            "instr(lower(COALESCE(c.name_th,'')),lower(?))>0",
            "instr(lower(COALESCE(c.name_en,'')),lower(?))>0",
            "instr(lower(COALESCE(c.description_th,'')),lower(?))>0",
            "instr(lower(COALESCE(c.description_en,'')),lower(?))>0",
        ]
        params.extend([values["search"]] * len(search_terms))
        for alias in search_aliases or []:
            alias_clause = (
                "(c.course_code_normalized=? AND cat.catalog_key=? "
                "AND p.program_code=? COLLATE NOCASE AND cat.academic_year=?)"
            )
            params.extend([
                alias["course_code"].strip().casefold(),
                alias["catalog_key"],
                alias["program"],
                alias["academic_year"],
            ])
            plan = alias.get("plan")
            if isinstance(plan, str) and plan.strip():
                alias_clause = (
                    "(c.course_code_normalized=? AND cat.catalog_key=? "
                    "AND p.program_code=? COLLATE NOCASE AND cat.academic_year=? "
                    "AND EXISTS (SELECT 1 FROM curriculum_plans alias_plan "
                    "JOIN plan_placements alias_placement ON alias_placement.plan_id=alias_plan.plan_id "
                    "WHERE alias_placement.course_id=c.course_id "
                    "AND alias_plan.catalog_id=c.catalog_id AND alias_plan.program_id=p.program_id "
                    "AND (alias_plan.plan_key=? COLLATE NOCASE OR alias_plan.plan_code=? COLLATE NOCASE)))"
                )
                params.extend([plan, plan])
            search_terms.append(alias_clause)
        clauses.append("(" + " OR ".join(search_terms) + ")")
    if values.get("plan") and exclude != "plan":
        clauses.append("EXISTS (SELECT 1 FROM curriculum_plans pf "
                       "JOIN plan_placements pfp ON pfp.plan_id=pf.plan_id "
                       "WHERE pfp.course_id=c.course_id AND pf.catalog_id=c.catalog_id "
                       "AND pf.program_id=p.program_id AND (pf.plan_key=? COLLATE NOCASE "
                       "OR pf.plan_code=? COLLATE NOCASE))")
        params.extend([values["plan"], values["plan"]])
    for key, normal, flexible in (("study_year", "year_number", "flexible_year_number"),
                                  ("semester", "semester_number", "flexible_semester_number")):
        if values.get(key) and exclude != key:
            try:
                number = int(values[key])
            except ValueError:
                raise HTTPException(422, f"{key.replace('_', ' ').capitalize()} must be a number") from None
            clauses.append(f"EXISTS (SELECT 1 FROM plan_placements ps "
                           "JOIN curriculum_plans pls ON pls.plan_id=ps.plan_id "
                           f"WHERE ps.course_id=c.course_id AND pls.program_id=p.program_id AND pls.catalog_id=c.catalog_id "
                           f"AND (ps.{normal}=? OR ps.{flexible}=?))")
            params.extend([number, number])
    return clauses, params


def filters(*, search: str = "", program: str = "", academic_year: str = "", plan: str = "",
            category: str = "", course_type: str = "", study_year: str = "", semester: str = "") -> dict[str, list[str]]:
    path = _db_path()
    if not path.is_file():
        raise HTTPException(503, "Curriculum index is not built yet.")
    try:
        with closing(sqlite3.connect(str(path))) as db:
            values = {"search": search, "program": program, "academic_year": academic_year, "plan": plan,
                      "category": category, "course_type": course_type,
                      "study_year": study_year, "semester": semester}
            base = "courses c JOIN catalogs cat ON cat.catalog_id=c.catalog_id JOIN programs p ON p.catalog_id=c.catalog_id"

            def options(expression: str, *, exclude: str, joins: str = "", order: str | None = None) -> list[str]:
                clauses, params = _filter_clauses(values, exclude=exclude)
                where = " AND ".join(clauses) if clauses else "1=1"
                ordering = order or expression
                sql = f"SELECT DISTINCT {expression} FROM {base} {joins} WHERE {where} AND {expression} IS NOT NULL ORDER BY {ordering}"
                return [str(row[0]) for row in db.execute(sql, params)]

            plan_join = ("JOIN curriculum_plans cp ON cp.catalog_id=c.catalog_id "
                         "AND cp.program_id=p.program_id")
            placement_join = (plan_join + " JOIN plan_placements pp ON pp.plan_id=cp.plan_id "
                              "AND pp.course_id=c.course_id")
            programs = options("p.program_code", exclude="program")
            years = options("cat.academic_year", exclude="academic_year", order="cat.academic_year DESC")
            categories = options("c.category", exclude="category")
            course_types = options("c.course_type", exclude="course_type")
            plans = options("COALESCE(NULLIF(cp.plan_code,''),cp.plan_key)", exclude="plan", joins=plan_join)
            study_years = options("COALESCE(pp.year_number,pp.flexible_year_number)", exclude="study_year", joins=placement_join)
            semesters = options("COALESCE(pp.semester_number,pp.flexible_semester_number)", exclude="semester", joins=placement_join)
    except sqlite3.Error:
        raise HTTPException(503, "Curriculum filters are unavailable.") from None
    return {"programs": programs, "academic_years": years, "categories": categories,
            "course_types": course_types, "plans": plans, "study_years": study_years, "semesters": semesters}


def list_courses(*, search: str = "", program: str = "", academic_year: str = "",
                 plan: str = "", category: str = "", course_type: str = "",
                 study_year: str = "", semester: str = "", course_code: str = "",
                 limit: int = 50, offset: int = 0) -> dict[str, Any]:
    path = _db_path()
    if not path.is_file():
        raise HTTPException(503, "Curriculum index is not built yet. Run `python -m rag.build_index` first.")
    clauses, params = _filter_clauses({"search": search, "program": program, "academic_year": academic_year,
                                       "plan": plan, "category": category, "course_type": course_type,
                                       "study_year": study_year, "semester": semester,
                                       "course_code": course_code},
                                      search_aliases=_verified_course_name_aliases(search))
    if not clauses:
        clauses = ["1=1"]
    where = " AND ".join(clauses)
    count_sql = f"""
        SELECT count(*) FROM courses c
        JOIN catalogs cat ON cat.catalog_id=c.catalog_id
        JOIN programs p ON p.catalog_id=c.catalog_id
        WHERE {where}
    """
    list_sql = f"""
        SELECT DISTINCT c.course_id, c.course_code, c.name_th, c.name_en, c.credits,
            c.credit_units, c.description_th, c.description_en, c.category, c.course_type,
            c.prerequisite_text, c.notes, p.program_code, cat.catalog_key, cat.academic_year,
            (SELECT group_concat(DISTINCT COALESCE(NULLIF(pl.plan_code,''),pl.plan_key))
             FROM curriculum_plans pl JOIN plan_placements plan_pp ON plan_pp.plan_id=pl.plan_id
             WHERE plan_pp.course_id=c.course_id AND pl.catalog_id=c.catalog_id
               AND pl.program_id=p.program_id) AS plans,
            (SELECT group_concat(DISTINCT CASE WHEN pp.year_number IS NOT NULL
                    THEN 'Year ' || pp.year_number || CASE WHEN pp.semester_number IS NOT NULL THEN ' · Semester ' || pp.semester_number ELSE '' END
                    ELSE pp.flexible_year_semester_raw END)
             FROM plan_placements pp JOIN curriculum_plans pl2 ON pl2.plan_id=pp.plan_id
             WHERE pp.course_id=c.course_id AND pl2.program_id=p.program_id
               AND pl2.catalog_id=c.catalog_id) AS placements,
            (SELECT group_concat(DISTINCT pv.document_category)
             FROM course_provenance cpf JOIN provenance pv USING(provenance_id)
             WHERE cpf.course_id=c.course_id) AS document_types,
            (SELECT group_concat(DISTINCT COALESCE(pv.source_filename,'') || CASE WHEN pv.source_page IS NOT NULL THEN ' · p.' || pv.source_page ELSE '' END)
             FROM course_provenance cpf JOIN provenance pv USING(provenance_id)
             WHERE cpf.course_id=c.course_id) AS sources
        FROM courses c
        JOIN catalogs cat ON cat.catalog_id=c.catalog_id
        JOIN programs p ON p.catalog_id=c.catalog_id
        WHERE {where}
        ORDER BY p.program_code, cat.academic_year DESC, c.course_code
        LIMIT ? OFFSET ?
    """
    try:
        with closing(sqlite3.connect(str(path))) as db:
            db.row_factory = sqlite3.Row
            total = db.execute(count_sql, params).fetchone()[0]
            rows = db.execute(list_sql, [*params, limit, offset]).fetchall()
    except sqlite3.Error:
        raise HTTPException(503, "Curriculum subjects are unavailable.") from None
    items = []
    for row in rows:
        item = dict(row)
        item["plans"] = [value for value in (item.get("plans") or "").split(",") if value]
        item["placements"] = [value for value in (item.get("placements") or "").split(",") if value]
        item["document_types"] = [value for value in (item.get("document_types") or "").split(",") if value]
        item["sources"] = [value for value in (item.get("sources") or "").split(",") if value]
        items.append(item)
    return {"items": items, "total": total, "limit": limit, "offset": offset}


def courses_requiring(*, prerequisite_code: str, program: str = "", academic_year: str = "",
                      plan: str = "") -> dict[str, Any]:
    """Return courses that list the supplied course as a prerequisite.

    Both directly resolved prerequisite edges and alternative-group membership
    are considered. Unresolved raw prerequisite text is intentionally excluded
    because it cannot be attributed safely to a course code.
    """
    path = _db_path()
    if not path.is_file():
        raise HTTPException(503, "Curriculum index is not built yet.")
    query = """
        SELECT DISTINCT dependent.course_code, cat.academic_year, p.program_code
        FROM prerequisites AS edge
        JOIN courses AS dependent ON dependent.course_id = edge.course_id
        JOIN catalogs AS cat ON cat.catalog_id = dependent.catalog_id
        JOIN programs AS p ON p.catalog_id = cat.catalog_id
        JOIN curriculum_plans AS cp ON cp.catalog_id = cat.catalog_id
                                   AND cp.program_id = p.program_id
        JOIN plan_placements AS dependent_placement
             ON dependent_placement.plan_id = cp.plan_id
            AND dependent_placement.course_id = dependent.course_id
        LEFT JOIN alternative_course_group_members AS members
               ON members.alternative_group_id = edge.alternative_group_id
        LEFT JOIN courses AS direct_prerequisite
               ON direct_prerequisite.course_id = edge.prerequisite_course_id
        LEFT JOIN courses AS alternative_prerequisite
               ON alternative_prerequisite.course_id = members.course_id
        WHERE lower(COALESCE(direct_prerequisite.course_code,
                             alternative_prerequisite.course_code, '')) = lower(?)
          AND EXISTS (
              SELECT 1 FROM plan_placements AS target_placement
              WHERE target_placement.plan_id = cp.plan_id
                AND target_placement.course_id = COALESCE(
                    direct_prerequisite.course_id, alternative_prerequisite.course_id
                )
          )
    """
    params: list[Any] = [prerequisite_code.strip()]
    if program:
        query += " AND p.program_code=? COLLATE NOCASE"
        params.append(program.strip())
    if academic_year:
        query += " AND cat.academic_year=?"
        params.append(academic_year.strip())
    if plan:
        query += " AND (cp.plan_key=? COLLATE NOCASE OR cp.plan_code=? COLLATE NOCASE)"
        params.extend([plan.strip(), plan.strip()])
    query += " ORDER BY p.program_code, cat.academic_year DESC, dependent.course_code"
    try:
        with closing(sqlite3.connect(str(path))) as db:
            rows = db.execute(query, params).fetchall()
    except sqlite3.Error:
        raise HTTPException(503, "Curriculum prerequisite data is unavailable.") from None

    courses: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for code, year, program_code in rows:
        identity = (str(code), str(year or ""), str(program_code or ""))
        if identity in seen:
            continue
        seen.add(identity)
        result = list_courses(course_code=str(code), program=str(program_code or ""),
                              academic_year=str(year or ""), plan=plan, limit=1000)
        courses.extend(result["items"])
    return {"items": courses, "total": len(courses)}


def course_prerequisite_requirements(*, course_id: int, plan: str = "") -> list[dict[str, Any]]:
    """Return normalized prerequisite edges, including alternative choice groups."""
    path = _db_path()
    if not path.is_file():
        raise HTTPException(503, "Curriculum index is not built yet.")
    query = """
        SELECT edge.prerequisite_id, edge.requirement_type, edge.raw_text,
               edge.prerequisite_course_id, direct.course_code AS direct_course_code,
               group_row.alternative_group_id, group_row.minimum_choices,
               group_row.maximum_choices, group_row.plan_id,
               COALESCE(NULLIF(plan.plan_code, ''), plan.plan_key) AS plan_label,
               members.course_id AS member_course_id,
               member_course.course_code AS member_course_code
        FROM prerequisites AS edge
        LEFT JOIN courses AS direct
               ON direct.course_id = edge.prerequisite_course_id
        LEFT JOIN alternative_course_groups AS group_row
               ON group_row.alternative_group_id = edge.alternative_group_id
        LEFT JOIN curriculum_plans AS plan
               ON plan.plan_id = group_row.plan_id
        LEFT JOIN alternative_course_group_members AS members
               ON members.alternative_group_id = group_row.alternative_group_id
        LEFT JOIN courses AS member_course
               ON member_course.course_id = members.course_id
        WHERE edge.course_id = ?
          AND (group_row.alternative_group_id IS NULL OR ? = ''
               OR group_row.plan_id IS NULL
               OR plan.plan_key = ? COLLATE NOCASE
               OR plan.plan_code = ? COLLATE NOCASE)
        ORDER BY edge.prerequisite_id, members.member_order, members.course_id
    """
    normalized_plan = plan.strip()
    try:
        with closing(sqlite3.connect(str(path))) as db:
            db.row_factory = sqlite3.Row
            rows = [dict(row) for row in db.execute(
                query, (course_id, normalized_plan, normalized_plan, normalized_plan)
            )]
    except sqlite3.Error:
        raise HTTPException(503, "Curriculum prerequisite data is unavailable.") from None

    requirements: list[dict[str, Any]] = []
    indexed: dict[tuple[Any, ...], dict[str, Any]] = {}
    alternative_members: dict[tuple[Any, ...], dict[int, str]] = {}
    for row in rows:
        group_id = row["alternative_group_id"]
        if group_id is not None and row["minimum_choices"] is not None:
            key = (
                "alternative", row["requirement_type"], row["raw_text"],
                row["minimum_choices"], row["maximum_choices"], group_id,
            )
            # Per-plan group ids differ, so merge only matching raw requirements,
            # choice bounds, and member sets.
            merge_key = key[:-1] + (
                tuple(sorted((
                    int(item["member_course_id"]), str(item["member_course_code"])
                ) for item in rows
                 if item["alternative_group_id"] == group_id
                    and item["member_course_id"] is not None)),
            )
            requirement = indexed.get(merge_key)
            if requirement is None:
                requirement = {
                    "kind": "alternative",
                    "requirement_type": row["requirement_type"],
                    "raw_text": row["raw_text"],
                    "minimum_choices": int(row["minimum_choices"]),
                    "maximum_choices": int(row["maximum_choices"]),
                    "members": [],
                    "plans": [],
                }
                indexed[merge_key] = requirement
                requirements.append(requirement)
                alternative_members[merge_key] = {}
            member_id = row["member_course_id"]
            member_code = row["member_course_code"]
            if member_id is not None and member_code:
                members = alternative_members[merge_key]
                members[int(member_id)] = str(member_code)
                requirement["members"] = [
                    {"course_id": value, "course_code": code}
                    for value, code in members.items()
                ]
            plan_label = row["plan_label"]
            if plan_label and plan_label not in requirement["plans"]:
                requirement["plans"].append(str(plan_label))
            continue

        if row["prerequisite_course_id"] is not None and row["direct_course_code"]:
            key = (
                "course", int(row["prerequisite_course_id"]),
                row["requirement_type"], row["raw_text"],
            )
            if key not in indexed:
                indexed[key] = {
                    "kind": "course",
                    "requirement_type": row["requirement_type"],
                    "raw_text": row["raw_text"],
                    "course_id": int(row["prerequisite_course_id"]),
                    "course_code": str(row["direct_course_code"]),
                    "plans": [],
                }
                requirements.append(indexed[key])
            continue

        raw_text = str(row["raw_text"] or "").strip()
        if raw_text:
            key = ("raw", row["requirement_type"], raw_text)
            if key not in indexed:
                indexed[key] = {
                    "kind": "raw",
                    "requirement_type": row["requirement_type"],
                    "raw_text": raw_text,
                    "plans": [],
                }
                requirements.append(indexed[key])
    return requirements


def program_requirements(*, program: str, academic_year: str,
                         requirement_type: str = "total_program_credits") -> dict[str, Any]:
    """Read a program requirement and its source provenance from the index."""
    path = _db_path()
    if not path.is_file():
        raise HTTPException(503, "Curriculum index is not built yet.")
    query = """
        SELECT req.requirement_id, req.program_code, cat.academic_year,
               req.requirement_type, req.operator, req.value, req.unit,
               prov.source_filename, prov.source_page, prov.source_uri,
               prov.source_locator
        FROM program_requirements AS req
        JOIN catalogs AS cat ON cat.catalog_id=req.catalog_id
        LEFT JOIN program_requirement_provenance AS links
               ON links.requirement_id=req.requirement_id
        LEFT JOIN provenance AS prov ON prov.provenance_id=links.provenance_id
        WHERE req.program_code=? COLLATE NOCASE AND cat.academic_year=?
          AND req.requirement_type=?
        ORDER BY req.requirement_id, prov.provenance_id
    """
    try:
        with closing(sqlite3.connect(str(path))) as db:
            db.row_factory = sqlite3.Row
            rows = [dict(row) for row in db.execute(
                query, (program.strip(), academic_year.strip(), requirement_type)
            )]
    except sqlite3.Error:
        raise HTTPException(503, "Program requirement data is unavailable.") from None

    results: list[dict[str, Any]] = []
    by_id: dict[int, dict[str, Any]] = {}
    for row in rows:
        requirement_id = int(row["requirement_id"])
        item = by_id.get(requirement_id)
        if item is None:
            item = {key: row[key] for key in (
                "requirement_id", "program_code", "academic_year", "requirement_type",
                "operator", "value", "unit",
            )}
            item["sources"] = []
            by_id[requirement_id] = item
            results.append(item)
        if row.get("source_filename") or row.get("source_uri"):
            item["sources"].append({key: row[key] for key in (
                "source_filename", "source_page", "source_uri", "source_locator",
            )})
    return {"items": results, "total": len(results)}


def latest_program_requirement_year(*, program: str,
                                    requirement_type: str = "total_program_credits") -> str | None:
    """Return the newest edition with a sourced, usable program requirement."""
    path = _db_path()
    if not path.is_file():
        raise HTTPException(503, "Curriculum index is not built yet.")
    query = """
        SELECT cat.academic_year
        FROM program_requirements AS req
        JOIN catalogs AS cat ON cat.catalog_id=req.catalog_id
        WHERE req.program_code=? COLLATE NOCASE
          AND req.requirement_type=?
          AND req.value IS NOT NULL
          AND NULLIF(TRIM(CAST(req.value AS TEXT)), '') IS NOT NULL
          AND NULLIF(TRIM(req.unit), '') IS NOT NULL
          AND CAST(cat.academic_year AS INTEGER) > 0
          AND EXISTS (
              SELECT 1
              FROM program_requirement_provenance AS links
              JOIN provenance AS prov ON prov.provenance_id=links.provenance_id
              WHERE links.requirement_id=req.requirement_id
                AND LOWER(TRIM(COALESCE(prov.document_category, '')))='program_requirement'
                AND (NULLIF(TRIM(prov.source_filename), '') IS NOT NULL
                     OR NULLIF(TRIM(prov.source_uri), '') IS NOT NULL)
          )
        ORDER BY CAST(cat.academic_year AS INTEGER) DESC, cat.academic_year DESC
        LIMIT 1
    """
    try:
        with closing(sqlite3.connect(str(path))) as db:
            row = db.execute(query, (program.strip(), requirement_type)).fetchone()
    except sqlite3.Error:
        raise HTTPException(503, "Program requirement data is unavailable.") from None
    return str(row[0]) if row and row[0] is not None else None
