"""FastAPI bridge for the CUCUMBER curriculum QA system."""

import re
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from rag.hybrid_demo import (
    DEFAULT_CURRICULUM_DB_PATH,
    answer_question_once,
    parse_conversation_context,
)
from rag.policy.query import parse_policy_question
from rag.policy.routing import route_policy_question
from rag.providers.gemini import make_gemini_callable
from rag.query_spec import parse_query_spec
from rag.structured.queries import (
    catalog_keys_for_program,
    edition_catalog_keys_for_program,
)
from .hard_qa import HARD_INTERPRETATION_RESPONSE_JSON_SCHEMA, answer_hard_question
from .llm_sql_qa import (
    ask_sql,
    parse_focus_course_context,
    parse_result_courses_context,
)
from .schemas import (
    AskRequest,
    AskResponse,
    CourseDetailResponse,
    CurriculumResponse,
    ProgramsResponse,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent.parent / "frontend"
DIST_DIR = STATIC_DIR / "dist"

load_dotenv(PROJECT_ROOT / ".env")

app = FastAPI(
    title="CUCUMBER Curriculum API",
    description="FastAPI frontend bridge for the CUCUMBER curriculum QA system",
    version="1.0.0",
)

if STATIC_DIR.is_dir():
    app.mount(
        "/static",
        StaticFiles(directory=STATIC_DIR),
        name="static",
    )

if (DIST_DIR / "assets").is_dir():
    app.mount(
        "/assets",
        StaticFiles(directory=DIST_DIR / "assets"),
        name="assets",
    )


def _resolve_index_file() -> Path:
    """Prefer the built React bundle, fall back to the dev entry.

    Resolved from the current ``STATIC_DIR`` at call time so tests can
    patch ``main.STATIC_DIR`` to simulate a missing frontend.
    """
    dist_index = Path(STATIC_DIR) / "dist" / "index.html"
    if dist_index.is_file():
        return dist_index
    return Path(STATIC_DIR) / "index.html"


def _serve_spa() -> FileResponse:
    index_file = _resolve_index_file()
    if not index_file.is_file():
        raise HTTPException(status_code=404, detail="หน้าเว็บไม่พร้อมใช้งาน")
    return FileResponse(index_file)


class ProviderUnavailable(RuntimeError):
    """The optional model provider could not be created or called."""


_provider = None


def _lazy_provider(
    prompt: str,
    *,
    response_mime_type: str | None = None,
    response_schema: dict | None = None,
    response_json_schema: dict | None = None,
) -> str:
    global _provider
    if _provider is None:
        try:
            _provider = make_gemini_callable()
        except Exception as exc:
            raise ProviderUnavailable("optional model provider unavailable") from exc
    try:
        if response_mime_type is None and response_schema is None and response_json_schema is None:
            return _provider(prompt)
        generation_options = {}
        if response_mime_type is not None:
            generation_options["response_mime_type"] = response_mime_type
        if response_schema is not None:
            generation_options["response_schema"] = response_schema
        if response_json_schema is not None:
            generation_options["response_json_schema"] = response_json_schema
        return _provider(prompt, **generation_options)
    except Exception as exc:
        raise ProviderUnavailable("optional model provider unavailable") from exc


def _curriculum_db() -> Path:
    db_path = Path(DEFAULT_CURRICULUM_DB_PATH)
    if not db_path.is_file():
        raise HTTPException(status_code=503, detail="ไม่พบฐานข้อมูล CUCUMBER")
    return db_path


def _connect_ro(db_path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def _catalog_keys_for_academic_year(
    db_path: Path, program: str, academic_year: str
) -> tuple[str, ...]:
    """Return only edition catalogs for this program and exact academic year."""
    allowed = set(edition_catalog_keys_for_program(db_path, program))
    if not allowed:
        return ()
    con = _connect_ro(db_path)
    try:
        rows = con.execute(
            """
            SELECT DISTINCT c.catalog_key
            FROM curriculum_plans cp
            JOIN catalogs c ON c.catalog_id = cp.catalog_id
            JOIN programs p ON p.program_id = cp.program_id
            WHERE lower(trim(p.program_code)) = lower(trim(?))
              AND trim(c.academic_year) = ?
            ORDER BY c.catalog_key
            """,
            (program, academic_year),
        ).fetchall()
    finally:
        con.close()
    return tuple(
        row["catalog_key"]
        for row in rows
        if isinstance(row["catalog_key"], str) and row["catalog_key"] in allowed
    )


def _pending_catalog_context(query_spec, program: str, parsed_context=None) -> dict:
    """Keep only parsed curriculum scope while an edition is unresolved."""
    payload: dict = {"program": program, "pending_catalog_selection": True}
    plan = query_spec.plans[0] if len(query_spec.plans) == 1 else None
    plan = plan or (parsed_context.plan if parsed_context is not None else None)
    years = query_spec.years or (parsed_context.years if parsed_context is not None else ())
    semesters = query_spec.semesters or (
        parsed_context.semesters if parsed_context is not None else ()
    )
    operations = query_spec.operations or (
        parsed_context.operations if parsed_context is not None else ()
    )
    category = query_spec.category or (
        parsed_context.category if parsed_context is not None else None
    )
    course_code = (
        query_spec.course_codes[0]
        if len(query_spec.course_codes) == 1
        else parsed_context.course_code if parsed_context is not None else None
    )
    if plan is not None:
        payload["plan"] = plan
    if years:
        payload["years"] = list(years)
    if semesters:
        payload["semesters"] = list(semesters)
    if operations:
        payload["operations"] = list(operations)
    if category is not None:
        payload["category"] = category
    if course_code is not None:
        payload["course_code"] = course_code
    return payload


def _is_unsupported_retake_timing_question(question: str) -> bool:
    """Match only explicit failure/withdrawal + retake + timing requests."""
    return all(
        re.search(pattern, question, re.IGNORECASE) is not None
        for pattern in (
            r"ถอน|ตก|ไม่ผ่าน",
            r"(?:ลง\s*(?:เรียน\s*)?|เรียน\s*)(?:ใหม่|อีกที|อีกครั้ง|ซ้ำ)",
            r"ตอนไหน|เมื่อไหร่|เมื่อไร|เทอมไหน|ภาค(?:เรียน)?ไหน",
        )
    )


def _pending_catalog_query(context: dict) -> str | None:
    """Rebuild the supported list request from bounded pending fields."""
    if context.get("operations") != ["list"]:
        return None
    parts = [context["program"]]
    if isinstance(context.get("plan"), str):
        parts.append(context["plan"])
    for year in context.get("years", []):
        parts.append(f"ปี {year}")
    for semester in context.get("semesters", []):
        parts.append(f"เทอม {semester}")
    if isinstance(context.get("category"), str):
        parts.append(context["category"])
    if isinstance(context.get("course_code"), str):
        parts.append(context["course_code"])
    parts.append("มีวิชาอะไรบ้าง")
    return " ".join(parts)


def _validate_catalog_context(
    db_path: Path, catalog_key: str, program: str | None
) -> str:
    con = _connect_ro(db_path)
    try:
        rows = con.execute(
            """SELECT catalogs.catalog_id, catalogs.catalog_key,
                      programs.program_code_normalized
               FROM catalogs
               LEFT JOIN programs USING (catalog_id)
               WHERE lower(trim(catalogs.catalog_key)) = ?""",
            (catalog_key.strip().casefold(),),
        ).fetchall()
    finally:
        con.close()
    catalog_ids = {row["catalog_id"] for row in rows}
    if len(catalog_ids) != 1 or not rows or not rows[0]["catalog_key"]:
        raise ValueError("catalog_key does not identify exactly one available catalog")
    if program is not None and not any(
        row["program_code_normalized"] == program.strip().casefold()
        for row in rows
    ):
        raise ValueError("program is not available in the selected catalog")
    return str(rows[0]["catalog_key"]).strip()


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return _serve_spa()


@app.get("/chat", include_in_schema=False)
def spa_chat() -> FileResponse:
    return _serve_spa()


@app.get("/curriculum", include_in_schema=False)
def spa_curriculum() -> FileResponse:
    return _serve_spa()


@app.get("/api/health")
def health() -> dict:
    db_path = Path(DEFAULT_CURRICULUM_DB_PATH)

    return {
        "status": "ok" if db_path.is_file() else "degraded",
        "database": str(db_path),
        "database_ready": db_path.is_file(),
    }


@app.get("/api/programs", response_model=ProgramsResponse)
def list_programs() -> dict:
    db_path = _curriculum_db()
    con = _connect_ro(db_path)
    try:
        rows = con.execute(
            """
            SELECT p.program_code, c.catalog_key, c.academic_year,
                   cp.plan_key, cp.plan_name
            FROM curriculum_plans cp
            JOIN catalogs c ON c.catalog_id = cp.catalog_id
            JOIN programs p ON p.program_id = cp.program_id
            ORDER BY p.program_code, c.academic_year, c.catalog_key, cp.plan_key
            """
        ).fetchall()
    finally:
        con.close()

    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        program_code = row["program_code"]
        program = grouped.setdefault(program_code, {"plans": {}, "editions": {}})
        plan_info = {"plan_key": row["plan_key"], "plan_name": row["plan_name"]}
        program["plans"].setdefault(row["plan_key"], plan_info)
        edition_key = (row["catalog_key"], row["academic_year"])
        edition = program["editions"].setdefault(
            edition_key,
            {
                "catalog_key": row["catalog_key"],
                "academic_year": row["academic_year"],
                "plans": {},
            },
        )
        edition["plans"].setdefault(row["plan_key"], plan_info)
    return {
        "programs": [
            {
                "program_code": code,
                "plans": list(data["plans"].values()),
                "editions": [
                    {
                        **edition,
                        "plans": list(edition["plans"].values()),
                    }
                    for _, edition in sorted(
                        data["editions"].items(),
                        key=lambda item: (
                            str(item[0][1] or ""), str(item[0][0] or "")
                        ),
                    )
                ],
            }
            for code, data in sorted(grouped.items())
        ]
    }


@app.get("/api/curriculum", response_model=CurriculumResponse)
def list_curriculum(
    program: str | None = Query(default=None, max_length=20),
    catalog_key: str | None = Query(default=None, max_length=128),
    plan: str | None = Query(default=None, max_length=40),
    year: int | None = Query(default=None, ge=1, le=8),
    semester: int | None = Query(default=None, ge=1, le=3),
    search: str | None = Query(default=None, max_length=100),
    limit: int = Query(default=200, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> dict:
    db_path = _curriculum_db()
    conditions = ["(c.course_id IS NOT NULL OR mc.course_id IS NOT NULL)"]
    params: list = []

    if catalog_key is not None:
        try:
            catalog_key = _validate_catalog_context(db_path, catalog_key, program)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=f"invalid catalog_key: {exc}") from exc
        conditions.append("p.catalog_id = (SELECT catalog_id FROM catalogs WHERE catalog_key = ?)")
        params.append(catalog_key)

    if program:
        conditions.append("p.program_code = ?")
        params.append(program)
    if plan:
        conditions.append("p.plan_key = ?")
        params.append(plan)
    if year is not None:
        conditions.append("pl.year_number = ?")
        params.append(year)
    if semester is not None:
        conditions.append("pl.semester_number = ?")
        params.append(semester)
    if search:
        conditions.append(
            "(COALESCE(mc.course_code, c.course_code) LIKE ?"
            " OR COALESCE(mc.name_en, c.name_en) LIKE ?"
            " OR COALESCE(mc.name_th, c.name_th) LIKE ?)"
        )
        like = f"%{search}%"
        params.extend([like, like, like])

    where = " AND ".join(conditions)
    base_from = """
        FROM plan_placements pl
        JOIN curriculum_plans p ON p.plan_id = pl.plan_id
        JOIN catalogs cat ON cat.catalog_id = p.catalog_id
        LEFT JOIN courses c ON c.course_id = pl.course_id
        LEFT JOIN alternative_course_group_members m
            ON m.alternative_group_id = pl.alternative_group_id
        LEFT JOIN courses mc ON mc.course_id = m.course_id
        WHERE {where}
    """.format(where=where)

    con = _connect_ro(db_path)
    try:
        total = con.execute(f"SELECT COUNT(*) {base_from}", params).fetchone()[0]
        rows = con.execute(
            f"""
            SELECT
                cat.catalog_key AS catalog_key,
                cat.academic_year AS academic_year,
                p.program_code AS program,
                p.plan_key AS plan_key,
                pl.year_number AS year,
                pl.semester_number AS semester,
                pl.category AS placement_category,
                COALESCE(mc.course_code, c.course_code) AS course_code,
                COALESCE(mc.name_en, c.name_en) AS name_en,
                COALESCE(mc.name_th, c.name_th) AS name_th,
                COALESCE(mc.credits, c.credits) AS credits,
                COALESCE(mc.credit_units, c.credit_units) AS credit_units,
                COALESCE(mc.category, c.category) AS category,
                COALESCE(mc.course_type, c.course_type) AS course_type,
                COALESCE(mc.prerequisite_text, c.prerequisite_text) AS prerequisite_text
            {base_from}
            ORDER BY p.program_code, p.plan_key, pl.year_number,
                     pl.semester_number, pl.placement_order,
                     COALESCE(mc.course_code, c.course_code)
            LIMIT ? OFFSET ?
            """,
            [*params, limit, offset],
        ).fetchall()
    finally:
        con.close()

    return {
        "items": [dict(row) for row in rows],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@app.get("/api/courses/{course_code}", response_model=CourseDetailResponse)
def course_detail(
    course_code: str,
    program: str | None = Query(default=None, max_length=20),
    catalog_key: str | None = Query(default=None, max_length=128),
) -> dict:
    db_path = _curriculum_db()
    con = _connect_ro(db_path)
    try:
        if catalog_key is not None:
            try:
                catalog_key = _validate_catalog_context(db_path, catalog_key, program)
            except (TypeError, ValueError) as exc:
                raise HTTPException(status_code=422, detail=f"invalid catalog_key: {exc}") from exc
        conditions = ["c.course_code_normalized = ?"]
        params: list[Any] = [course_code.strip().lower()]
        if catalog_key is not None:
            conditions.append("c.catalog_id = (SELECT catalog_id FROM catalogs WHERE catalog_key = ?)")
            params.append(catalog_key)
        if program:
            conditions.append(
                "EXISTS (SELECT 1 FROM programs p WHERE p.catalog_id = c.catalog_id "
                "AND p.program_code_normalized = ?)"
            )
            params.append(program.strip().casefold())
        courses = con.execute(
            "SELECT c.*, cat.catalog_key, cat.academic_year FROM courses c "
            "JOIN catalogs cat ON cat.catalog_id = c.catalog_id WHERE "
            + " AND ".join(conditions)
            + " ORDER BY cat.academic_year, cat.catalog_key",
            params,
        ).fetchall()
        if not courses:
            raise HTTPException(status_code=404, detail="ไม่พบรายวิชา")
        if len(courses) > 1:
            raise HTTPException(
                status_code=409,
                detail="พบรหัสวิชาในหลายหลักสูตร กรุณาระบุ catalog_key",
            )
        course = courses[0]

        placements = con.execute(
            """
            SELECT cat.catalog_key, cat.academic_year,
                   p.program_code AS program, p.plan_key AS plan_key,
                   pl.year_number AS year, pl.semester_number AS semester,
                   pl.category AS placement_category
            FROM plan_placements pl
            JOIN curriculum_plans p ON p.plan_id = pl.plan_id
            JOIN catalogs cat ON cat.catalog_id = p.catalog_id
            LEFT JOIN alternative_course_group_members m
                ON m.alternative_group_id = pl.alternative_group_id
            WHERE pl.course_id = ?
               OR m.course_id = ?
            ORDER BY p.program_code, p.plan_key, pl.year_number, pl.semester_number
            """,
            (course["course_id"], course["course_id"]),
        ).fetchall()

        prereqs = con.execute(
            """
            SELECT COALESCE(pc.course_code, mg.course_code) AS course_code,
                   COALESCE(pc.name_en, mg.name_en) AS name_en,
                   pr.requirement_type, pr.raw_text
            FROM prerequisites pr
            LEFT JOIN courses pc ON pc.course_id = pr.prerequisite_course_id
            LEFT JOIN alternative_course_groups g
                ON g.alternative_group_id = pr.alternative_group_id
            LEFT JOIN alternative_course_group_members gm
                ON gm.alternative_group_id = pr.alternative_group_id
            LEFT JOIN courses mg ON mg.course_id = gm.course_id
            WHERE pr.course_id = ?
            ORDER BY pr.prerequisite_order
            """,
            (course["course_id"],),
        ).fetchall()
    finally:
        con.close()

    return {
        "course": dict(course),
        "placements": [dict(row) for row in placements],
        "prerequisites": [dict(row) for row in prereqs],
    }


@app.post("/api/ask", response_model=AskResponse, response_model_exclude_unset=True)
def ask(request: AskRequest) -> dict:
    db_path = _curriculum_db()
    query_spec = parse_query_spec(request.question)
    effective_question = request.question

    try:
        raw_context = request.conversation_context
        legacy_context = raw_context
        raw_focus = None
        raw_result_courses = None
        raw_result_scope = None
        raw_result_set_empty = False
        has_result_set_empty = False
        has_result_scope = False
        focus_catalog_key = None
        if isinstance(raw_context, dict):
            legacy_context = dict(raw_context)
            if "pending_catalog_selection" in legacy_context:
                pending_marker = legacy_context.pop("pending_catalog_selection")
                if pending_marker is not True or legacy_context.get("catalog_key") is not None:
                    raise ValueError("pending catalog context is invalid")
                pending_context = parse_conversation_context(legacy_context)
                if pending_context is None or pending_context.program is None:
                    raise ValueError("pending catalog context requires a program")
                year_reply = re.fullmatch(
                    r"\s*(?:(?:ของ\s*)?(?:ปี\s*)?)?(\d{4})\s*",
                    request.question,
                )
                selected_keys = (
                    _catalog_keys_for_academic_year(
                        db_path,
                        pending_context.program,
                        year_reply.group(1),
                    )
                    if year_reply is not None
                    else ()
                )
                if len(selected_keys) != 1:
                    available = list(
                        edition_catalog_keys_for_program(db_path, pending_context.program)
                    )
                    pending_payload = dict(legacy_context)
                    pending_payload["pending_catalog_selection"] = True
                    return {
                        "question": request.question,
                        "answer": (
                            f"โปรดเลือกฉบับหลักสูตรของ {pending_context.program}: "
                            + ", ".join(available)
                        ),
                        "status": "clarification_required",
                        "action": "catalog_required",
                        "route": "llm_sql",
                        "provenance": [],
                        "next_context": pending_payload,
                        "comparison": None,
                    }
                legacy_context["catalog_key"] = selected_keys[0]
                effective_question = _pending_catalog_query(legacy_context) or (
                    request.question + " อีกที"
                )
            raw_focus = legacy_context.pop("focus_course", None)
            focus_catalog_key = legacy_context.pop("focus_catalog_key", None)
            raw_result_courses = legacy_context.pop("result_courses", None)
            has_result_set_empty = "result_set_empty" in legacy_context
            raw_result_set_empty = legacy_context.pop("result_set_empty", False)
            has_result_scope = "result_scope_program" in legacy_context
            raw_result_scope = legacy_context.pop("result_scope_program", None)
            if focus_catalog_key is not None and (
                not isinstance(focus_catalog_key, str)
                or not focus_catalog_key.strip()
                or len(focus_catalog_key.strip()) > 128
            ):
                raise ValueError("focus_catalog_key must be a non-empty string or null")
            active_catalog_key = legacy_context.get("catalog_key")
            source_catalog_key = focus_catalog_key
            if source_catalog_key is None and isinstance(raw_focus, dict):
                source_catalog_key = raw_focus.get("catalog_key")
            if source_catalog_key is None and isinstance(raw_result_courses, list):
                result_catalog_keys = {
                    item.get("catalog_key")
                    for item in raw_result_courses
                    if isinstance(item, dict) and isinstance(item.get("catalog_key"), str)
                }
                if len(result_catalog_keys) == 1:
                    source_catalog_key = next(iter(result_catalog_keys))
            if (
                active_catalog_key is None
                and isinstance(source_catalog_key, str)
                and source_catalog_key.strip()
            ):
                # A single-edition previous result can carry its own canonical
                # scope when the caller omitted the top-level selector.
                legacy_context["catalog_key"] = source_catalog_key.strip()
                active_catalog_key = source_catalog_key.strip()
            if (
                isinstance(source_catalog_key, str)
                and isinstance(active_catalog_key, str)
                and source_catalog_key.strip().casefold()
                != active_catalog_key.strip().casefold()
            ):
                # The selected edition supersedes all prior focus and result context.
                raw_focus = None
                raw_result_courses = None
                raw_result_scope = None
                raw_result_set_empty = False
                has_result_scope = False
                has_result_set_empty = False
                for key in ("plan", "plans", "year", "years", "semester", "semesters", "operations"):
                    legacy_context.pop(key, None)
                focus_catalog_key = active_catalog_key.strip()
            if raw_result_courses is None and has_result_scope:
                raise ValueError("result_scope_program requires result_courses")
            if has_result_set_empty and raw_result_set_empty is not True:
                raise ValueError("result_set_empty must be true when provided")
        parsed_context = parse_conversation_context(legacy_context)
        if (
            query_spec.program
            and parsed_context is not None
            and parsed_context.program is not None
            and query_spec.program.casefold() != parsed_context.program.casefold()
        ):
            # An explicit program in the current turn replaces the prior turn's
            # structural context, including edition and term focus.
            parsed_context = parse_conversation_context({"program": query_spec.program})
            raw_focus = None
            raw_result_courses = None
            raw_result_scope = None
            raw_result_set_empty = False
            has_result_scope = False
            has_result_set_empty = False
            focus_catalog_key = None
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"invalid conversation_context: {exc}") from exc

    program = (
        query_spec.program
        or (parsed_context.program if parsed_context is not None else None)
    )
    explicit_course_credit = bool(
        (query_spec.course_codes or query_spec.course_name is not None)
        and "sum_credits" in query_spec.operations
    )
    catalog_key = parsed_context.catalog_key if parsed_context is not None else None
    if catalog_key is not None:
        try:
            catalog_key = _validate_catalog_context(db_path, catalog_key, program)
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=422,
                detail=f"invalid conversation_context: {exc}",
            ) from exc
    scope_program = program or query_spec.program
    explicit_edition_comparison = (
        (
            "2560" in request.question
            and "2565" in request.question
            and any(
                cue in request.question.casefold()
                for cue in ("เปรียบเทียบ", "เทียบ", "ต่างกัน", "compare", " vs ")
            )
        )
        or (
            "หลักสูตรเก่า" in request.question
            and "หลักสูตรใหม่" in request.question
        )
        or (
            "coop" in request.question.casefold()
            and "no_coop" in request.question.casefold()
            and any(
                cue in request.question.casefold()
                for cue in ("ต่างกัน", "เปรียบเทียบ", "compare")
            )
        )
    )
    policy_query = parse_policy_question(request.question)
    requires_catalog = bool(
        query_spec.operations
        or query_spec.plans
        or query_spec.years
        or query_spec.semesters
        or query_spec.course_codes
        or query_spec.course_name
        or query_spec.topic
        or (
            policy_query is not None
            and policy_query.kind == "program_total_credits"
        )
    )
    unanchored_term_followup = (
        scope_program is None
        and bool(query_spec.years or query_spec.semesters)
        and re.search(
            r"(?:แล้ว\s*(?:ปี|เทอม|ภาค)|(?:ปี|เทอม|ภาค)[^\n]{0,24}ล่ะ)",
            request.question,
        )
        is not None
    )
    if unanchored_term_followup:
        return {
            "question": request.question,
            "answer": "โปรดระบุหลักสูตรก่อนค้นหาข้อมูลภาคเรียนนี้",
            "status": "clarification_required",
            "action": "program_required",
            "route": "llm_sql",
            "provenance": [],
            "next_context": None,
            "comparison": None,
        }
    if (
        scope_program is not None
        and requires_catalog
        and catalog_key is None
        and not explicit_edition_comparison
    ):
        try:
            program_catalogs = catalog_keys_for_program(db_path, scope_program)
            edition_catalogs = edition_catalog_keys_for_program(db_path, scope_program)
        except (OSError, sqlite3.Error, TypeError, ValueError):
            program_catalogs = ()
            edition_catalogs = ()
        if edition_catalogs:
            keys = list(edition_catalogs)
            return {
                "question": request.question,
                "answer": f"โปรดเลือกฉบับหลักสูตรของ {scope_program}: " + ", ".join(keys),
                "status": "clarification_required",
                "action": "catalog_required",
                "route": "llm_sql",
                "provenance": [],
                "next_context": _pending_catalog_context(
                    query_spec, scope_program, parsed_context
                ),
                "comparison": None,
            }
    try:
        focus_course = parse_focus_course_context(
            raw_focus, program, catalog_key
        )
        parsed_results = parse_result_courses_context(
            raw_result_courses,
            raw_result_scope if has_result_scope else program,
            program,
            raw_result_set_empty,
            catalog_key,
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422, detail=f"invalid conversation_context: {exc}"
        ) from exc
    if _is_unsupported_retake_timing_question(request.question):
        next_context: dict = {}
        if program is not None:
            next_context["program"] = program
        if catalog_key is not None:
            next_context["catalog_key"] = catalog_key
        if (
            parsed_context is not None
            and parsed_context.plan is not None
            and parsed_context.program == program
        ):
            next_context["plan"] = parsed_context.plan
        return {
            "question": request.question,
            "answer": (
                "CUCUMBER มีข้อมูลโครงสร้างและตำแหน่งรายวิชาในหลักสูตร "
                "แต่ไม่มีข้อมูลการเปิดสอนจริงในแต่ละภาคเรียน จึงยืนยันไม่ได้ว่า "
                "หลังถอนหรือสอบไม่ผ่านจะลงเรียนวิชานี้ซ้ำได้เมื่อใด "
                "โปรดตรวจสอบระบบเปิดรายวิชาหรือระบบลงทะเบียนของมหาวิทยาลัย"
            ),
            "status": "insufficient_evidence",
            "action": "course_offering_data_required",
            "route": "llm_sql",
            "provenance": [],
            "next_context": next_context or None,
            "comparison": None,
        }
    service_context: dict | None = {}
    if program is not None:
        service_context["program"] = program
    if catalog_key is not None:
        service_context["catalog_key"] = catalog_key
    if parsed_context is not None:
        if parsed_context.plan is not None and not explicit_course_credit:
            service_context["plan"] = parsed_context.plan
        if parsed_context.years and not explicit_course_credit:
            service_context["years"] = list(parsed_context.years)
        if parsed_context.semesters and not explicit_course_credit:
            service_context["semesters"] = list(parsed_context.semesters)
        if parsed_context.operations and not explicit_course_credit:
            service_context["operations"] = list(parsed_context.operations)
        if parsed_context.category is not None and not explicit_course_credit:
            service_context["category"] = parsed_context.category
        if parsed_context.course_code is not None and not explicit_course_credit:
            service_context["course_code"] = parsed_context.course_code
        if focus_catalog_key is not None:
            service_context["focus_catalog_key"] = focus_catalog_key.strip()
    if focus_course is not None and not explicit_course_credit:
        service_context["focus_course"] = focus_course
    if parsed_results is not None and not explicit_course_credit:
        service_context["result_courses"] = parsed_results[0]
        service_context["result_scope_program"] = parsed_results[1]
        if not parsed_results[0]:
            service_context["result_set_empty"] = True
    if not service_context:
        service_context = None

    if (
        policy_query is not None
        and policy_query.kind == "program_total_credits"
        and policy_query.program
        and edition_catalog_keys_for_program(db_path, policy_query.program)
        and catalog_key is not None
    ):
        policy_result = route_policy_question(
            db_path, request.question, catalog_key=catalog_key
        )
        if policy_result is not None:
            return {
                "question": request.question,
                "answer": policy_result.final_answer
                or "ไม่พบหลักฐานที่ยืนยันหน่วยกิตรวมของฉบับหลักสูตรนี้",
                "status": policy_result.status,
                "action": None,
                "route": "llm_sql",
                "hard_task_type": None,
                "provenance": list(policy_result.provenance),
                "next_context": service_context,
                "comparison": None,
            }
    provider_unavailable = False

    def model_provider(
        prompt: str,
        *,
        response_mime_type: str | None = None,
        response_schema: dict | None = None,
        response_json_schema: dict | None = None,
    ) -> str:
        nonlocal provider_unavailable
        try:
            if response_mime_type is None and response_schema is None and response_json_schema is None:
                return _lazy_provider(prompt)
            generation_options = {}
            if response_mime_type is not None:
                generation_options["response_mime_type"] = response_mime_type
            if response_schema is not None:
                generation_options["response_schema"] = response_schema
            if response_json_schema is not None:
                generation_options["response_json_schema"] = response_json_schema
            return _lazy_provider(
                prompt,
                **generation_options,
            )
        except ProviderUnavailable:
            provider_unavailable = True
            raise

    def hard_interpreter_provider(prompt: str) -> str:
        return model_provider(
            prompt,
            response_mime_type="application/json",
            response_json_schema=HARD_INTERPRETATION_RESPONSE_JSON_SCHEMA,
        )

    try:
        hard_context = {}
        if program is not None:
            hard_context["program"] = program
        if catalog_key is not None:
            hard_context["catalog_key"] = catalog_key
        if (
            parsed_context is not None
            and parsed_context.plan is not None
            and parsed_context.program == program
            and not explicit_course_credit
        ):
            hard_context["plan"] = parsed_context.plan
        if (
            parsed_context is not None
            and parsed_context.course_code is not None
            and not explicit_course_credit
        ):
            hard_context["course_code"] = parsed_context.course_code
        result = None
        # Hard QA does not accept bounded result-course context; SQL QA does.
        if parsed_results is None:
            result = answer_hard_question(
                db_path,
                request.question,
                hard_context or None,
                hard_interpreter_provider,
            )
        if result is None:
            def ground_sql_answer(current_question: str) -> dict:
                grounding_context = {
                    key: value
                    for key, value in (service_context or {}).items()
                    if key
                    in {
                        "program", "catalog_key", "plan", "years", "semesters",
                        "operations", "category", "course_code",
                    }
                }
                if service_context and service_context.get("result_courses"):
                    # The canonical QA context cannot represent an arbitrary
                    # bounded multi-course result set; do not widen it.
                    return {"status": "insufficient_evidence", "provenance": []}
                focus_course = (service_context or {}).get("focus_course")
                if isinstance(focus_course, dict):
                    course_code = focus_course.get("course_code")
                    if isinstance(course_code, str) and course_code.strip():
                        grounding_context["course_code"] = course_code.strip()
                return answer_question_once(
                    db_path,
                    current_question,
                    intent_model_callable=model_provider,
                    conversation_context=grounding_context or None,
                )

            result = ask_sql(
                db_path,
                effective_question,
                program,
                model_provider,
                model_provider,
                conversation_context=service_context,
                grounding_callable=ground_sql_answer,
            )
    except ProviderUnavailable as exc:
        raise HTTPException(
            status_code=503,
            detail="ตัวให้บริการโมเดลไม่พร้อมใช้งาน",
        ) from exc
    except Exception:
        result = {"status": "error", "answer": "", "error": {"code": "service_failure"}}

    if provider_unavailable:
        raise HTTPException(
            status_code=503,
            detail="ตัวให้บริการโมเดลไม่พร้อมใช้งาน",
        )

    if not isinstance(result, dict):
        result = {"status": "error", "answer": "", "error": {"code": "service_failure"}}
    error = result.get("error") if isinstance(result.get("error"), dict) else {}
    status = result.get("status")
    if not isinstance(status, str):
        status = "error"

    fallback_context: dict = {}
    if program is not None:
        fallback_context["program"] = program
    if catalog_key is not None:
        fallback_context["catalog_key"] = catalog_key
    if parsed_context is not None:
        if parsed_context.plan is not None and not explicit_course_credit:
            fallback_context["plan"] = parsed_context.plan
        if parsed_context.years and not explicit_course_credit:
            fallback_context["years"] = list(parsed_context.years)
        if parsed_context.semesters and not explicit_course_credit:
            fallback_context["semesters"] = list(parsed_context.semesters)
        if parsed_context.operations and not explicit_course_credit:
            fallback_context["operations"] = list(parsed_context.operations)
        if parsed_context.category is not None and not explicit_course_credit:
            fallback_context["category"] = parsed_context.category
        if parsed_context.course_code is not None and not explicit_course_credit:
            fallback_context["course_code"] = parsed_context.course_code
        if focus_catalog_key is not None:
            fallback_context["focus_catalog_key"] = focus_catalog_key.strip()
    if focus_course is not None and not explicit_course_credit:
        fallback_context["focus_course"] = focus_course
    if parsed_results is not None and not explicit_course_credit:
        fallback_context["result_courses"] = parsed_results[0]
        fallback_context["result_scope_program"] = parsed_results[1]
        if not parsed_results[0]:
            fallback_context["result_set_empty"] = True
    next_context = result.get("next_context", fallback_context or None)

    response = {
        "question": request.question,
        "answer": result.get("answer") if isinstance(result.get("answer"), str) else "",
        "status": status,
        "action": (result.get("action") or error.get("code")) if status == "error" else result.get("action"),
        "route": result.get("route") if result.get("route") in {"hard", "llm_sql"} else "llm_sql",
        "hard_task_type": result.get("hard_task_type"),
        "provenance": result.get("provenance", []) if isinstance(result.get("provenance"), list) else [],
        "next_context": next_context,
        "comparison": result.get("comparison") if isinstance(result.get("comparison"), dict) else None,
    }
    if isinstance(result.get("plan_results"), list):
        response["plan_results"] = result["plan_results"]
    return response
