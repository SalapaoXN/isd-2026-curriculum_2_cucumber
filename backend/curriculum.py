"""Read-only listing and filter options for indexed curriculum subjects."""

import sqlite3
from contextlib import closing
from typing import Any

from fastapi import HTTPException

from .rag_service import _db_path


def _filter_clauses(values: dict[str, str], *, exclude: str = "") -> tuple[list[str], list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    for key, expr in (("program", "p.program_code"), ("academic_year", "cat.academic_year"),
                      ("category", "c.category"), ("course_type", "c.course_type")):
        if key != exclude and values.get(key):
            clauses.append(f"{expr}=?")
            params.append(values[key])
    if values.get("search") and exclude != "search":
        clauses.append("(instr(lower(c.course_code),lower(?))>0 OR instr(lower(COALESCE(c.name_th,'')),lower(?))>0 "
                       "OR instr(lower(COALESCE(c.name_en,'')),lower(?))>0 "
                       "OR instr(lower(COALESCE(c.description_th,'')),lower(?))>0 "
                       "OR instr(lower(COALESCE(c.description_en,'')),lower(?))>0)")
        params.extend([values["search"]] * 5)
    if values.get("plan") and exclude != "plan":
        clauses.append("EXISTS (SELECT 1 FROM curriculum_plans pf WHERE pf.catalog_id=c.catalog_id "
                       "AND pf.program_id=p.program_id AND (pf.plan_key=? OR pf.plan_code=?))")
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
                 study_year: str = "", semester: str = "", limit: int = 50, offset: int = 0) -> dict[str, Any]:
    path = _db_path()
    if not path.is_file():
        raise HTTPException(503, "Curriculum index is not built yet. Run `python -m rag.build_index` first.")
    clauses, params = _filter_clauses({"search": search, "program": program, "academic_year": academic_year,
                                       "plan": plan, "category": category, "course_type": course_type,
                                       "study_year": study_year, "semester": semester})
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
             FROM curriculum_plans pl WHERE pl.catalog_id=c.catalog_id AND pl.program_id=p.program_id) AS plans,
            (SELECT group_concat(DISTINCT CASE WHEN pp.year_number IS NOT NULL
                    THEN 'Year ' || pp.year_number || CASE WHEN pp.semester_number IS NOT NULL THEN ' · Semester ' || pp.semester_number ELSE '' END
                    ELSE pp.flexible_year_semester_raw END)
             FROM plan_placements pp JOIN curriculum_plans pl2 ON pl2.plan_id=pp.plan_id
             WHERE pp.course_id=c.course_id AND pl2.program_id=p.program_id) AS placements,
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
