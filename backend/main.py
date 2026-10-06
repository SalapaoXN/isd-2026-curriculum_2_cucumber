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
from rag.policy.answer import answer_policy_query
from rag.prev_answer import classify_prev_followup, parse_last_answer
from backend.prev_followup import (
    answer_previous_followup,
    build_course_reference,
    build_policy_reference,
)
from rag.policy.routing import route_policy_question
from rag.providers.gemini import make_gemini_callable
from rag.query_spec import is_prerequisite_collection, parse_query_spec
from rag.resolution import resolve_ordinal_course_reference
from rag.semantic.modes import (
    QA_MODE_SEMANTIC,
    QA_MODE_SHADOW,
    active_qa_mode,
    run_shadow_comparison,
    semantic_ask_response,
)
from rag.structured.queries import (
    catalog_keys_for_program,
    edition_catalog_keys_for_program,
)
from .hard_qa import (
    HARD_INTERPRETATION_RESPONSE_JSON_SCHEMA,
    answer_hard_question,
    answer_seven_term_followup,
)
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


def _focus_course_from_context_course_code(
    parsed_context, program: str | None, catalog_key: str | None
) -> dict | None:
    """Translate a prior answer's structural course_code into service focus shape.

    The grounded path persists its target as ``QueryContext.course_code``,
    which is valid public conversation context but not valid ``ask_sql``
    service context. Translate it through the single focus validator so
    system-emitted context stays consumable on the next turn; stale
    cross-program/edition targets fail closed to None.
    """
    if parsed_context is None or parsed_context.course_code is None:
        return None
    candidate: dict[str, str | None] = {
        "course_code": parsed_context.course_code,
        "program": parsed_context.program,
    }
    if parsed_context.catalog_key is not None:
        candidate["catalog_key"] = parsed_context.catalog_key
    try:
        return parse_focus_course_context(candidate, program, catalog_key)
    except (TypeError, ValueError):
        return None


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
        "qa_mode": active_qa_mode(),
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


def _semantic_providers() -> dict:
    """Build semantic-mode provider callables; missing key fails closed."""
    try:
        generate = make_gemini_callable()
    except Exception:
        return {}
    return {
        "interpret_callable": generate,
        "answer_callable": generate,
        "sql_callable": generate,
    }


def _semantic_route_response(request: AskRequest) -> dict:
    """Serve POST /api/ask from the semantic pipeline (mode=semantic)."""
    db_path = _curriculum_db()
    providers = _semantic_providers()
    if not providers:
        return {
            "question": request.question,
            "answer": "",
            "status": "insufficient_evidence",
            "action": "insufficient_evidence",
            "route": "semantic",
            "provenance": [],
            "next_context": None,
            "comparison": None,
        }
    context = (
        dict(request.conversation_context)
        if isinstance(request.conversation_context, dict)
        else None
    )
    try:
        return semantic_ask_response(
            db_path,
            request.question,
            context,
            interpret_callable=providers["interpret_callable"],
            answer_callable=providers["answer_callable"],
            sql_callable=providers["sql_callable"],
        )
    except Exception:
        return {
            "question": request.question,
            "answer": "",
            "status": "insufficient_evidence",
            "action": "insufficient_evidence",
            "route": "semantic",
            "provenance": [],
            "next_context": None,
            "comparison": None,
        }


def _capture_shadow_semantic(request: AskRequest) -> None:
    """Best-effort shadow run: legacy output is never touched or read."""
    try:
        db_path = _curriculum_db()
        providers = _semantic_providers()
        if not providers:
            return
        context = (
            dict(request.conversation_context)
            if isinstance(request.conversation_context, dict)
            else None
        )
        run_shadow_comparison(
            db_path,
            request.question,
            context,
            None,
            interpret_callable=providers["interpret_callable"],
            answer_callable=providers["answer_callable"],
            sql_callable=providers["sql_callable"],
        )
    except Exception:
        return


@app.post("/api/ask", response_model=AskResponse, response_model_exclude_unset=True)
def ask(request: AskRequest) -> dict:
    qa_mode = active_qa_mode()
    if qa_mode == QA_MODE_SEMANTIC:
        return _semantic_route_response(request)
    if qa_mode == QA_MODE_SHADOW:
        # Shadow runs beside legacy output through internal logging only.
        _capture_shadow_semantic(request)
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
        raw_last_answer = None
        has_result_set_empty = False
        has_result_scope = False
        focus_catalog_key = None
        raw_semantic_topic = None
        raw_study_plan_context = None
        if isinstance(raw_context, dict):
            legacy_context = dict(raw_context)
            raw_semantic_topic = legacy_context.pop("semantic_topic", None)
            raw_study_plan_context = legacy_context.pop("study_plan_context", None)
            if raw_study_plan_context is not None and (
                not isinstance(raw_study_plan_context, dict)
                or set(raw_study_plan_context) != {"kind", "program", "catalog_key", "plan"}
                or raw_study_plan_context.get("kind") != "seven_term_plan"
                or any(
                    not isinstance(raw_study_plan_context.get(key), str)
                    or not raw_study_plan_context[key].strip()
                    for key in ("program", "catalog_key", "plan")
                )
            ):
                raw_study_plan_context = None
            if raw_semantic_topic is not None:
                if (
                    not isinstance(raw_semantic_topic, str)
                    or not raw_semantic_topic.strip()
                    or len(raw_semantic_topic.strip()) > 80
                ):
                    raise ValueError(
                        "semantic_topic must be a non-empty string of at most "
                        "80 characters"
                    )
                raw_semantic_topic = raw_semantic_topic.strip()
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
            raw_last_answer = legacy_context.pop("last_answer", None)
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
                raw_semantic_topic = None
                raw_study_plan_context = None
                raw_last_answer = None
                has_result_scope = False
                has_result_set_empty = False
                for key in ("plan", "plans", "year", "years", "semester", "semesters", "operations", "course_code", "category"):
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
            raw_semantic_topic = None
            raw_study_plan_context = None
            raw_last_answer = None
            has_result_scope = False
            has_result_set_empty = False
            focus_catalog_key = None
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"invalid conversation_context: {exc}") from exc

    if is_prerequisite_collection(query_spec):
        # A fresh relationship collection replaces narrow answer targets;
        # only stable curriculum identity/plan survives from previous turns.
        if parsed_context is not None:
            stable_context = {
                "program": parsed_context.program,
                "catalog_key": parsed_context.catalog_key,
                "plan": query_spec.plans[0] if len(query_spec.plans) == 1 else parsed_context.plan,
                # This operation is derived from the current collection intent,
                # never copied from the previous answer's operation.
                "operations": ["list"],
            }
            parsed_context = parse_conversation_context(
                {key: value for key, value in stable_context.items() if value is not None}
            )
        raw_focus = None
        raw_result_courses = None
        raw_result_scope = None
        raw_result_set_empty = False
        raw_semantic_topic = None
        raw_study_plan_context = None
        raw_last_answer = None
        has_result_scope = False
        has_result_set_empty = False
        focus_catalog_key = None

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
    selected_plan = (
        query_spec.plans[0]
        if len(query_spec.plans) == 1
        else parsed_context.plan if parsed_context is not None else None
    )
    study_plan = raw_study_plan_context
    if study_plan is not None and not (
        study_plan["program"].casefold() == str(program or "").casefold()
        and study_plan["catalog_key"].casefold() == str(catalog_key or "").casefold()
        and study_plan["plan"].casefold() == str(selected_plan or "").casefold()
    ):
        study_plan = None
    followup_operations = set(query_spec.operations)
    credit_cue = bool(
        "sum_credits" in followup_operations
        or re.search(r"หน่วย(?:กิต)?|เครดิต|\bcredits?\b", request.question, re.IGNORECASE)
    )
    explicit_h4_term = bool(
        re.search(r"ปี\s*(?:ที่\s*)?\d+", query_spec.normalized_question)
        and re.search(r"(?:เทอม|ภาคเรียน)\s*\d+", query_spec.normalized_question)
    )
    if study_plan is not None and explicit_h4_term and (
        query_spec.judgement == "unsupported"
        or len(query_spec.years) != 1
        or len(query_spec.semesters) != 1
    ):
        return {
            "question": request.question,
            "answer": "ไม่พบข้อมูลที่ยืนยันได้สำหรับภาคเรียนนี้",
            "status": "insufficient_evidence",
            "action": None,
            "route": "hard",
            "hard_task_type": "seven_term_plan",
            "provenance": [],
            "next_context": {
                "program": program,
                "catalog_key": catalog_key,
                "plan": selected_plan,
                "study_plan_context": study_plan,
            },
            "comparison": None,
        }
    h4_term_followup = bool(
        study_plan is not None
        and query_spec.judgement != "unsupported"
        and len(query_spec.years) == 1
        and len(query_spec.semesters) == 1
        and not query_spec.course_codes
        and query_spec.topic is None
        and query_spec.category is None
        and followup_operations <= {"list", "sum_credits"}
    )
    if h4_term_followup:
        include_courses = "list" in followup_operations
        summary = answer_seven_term_followup(
            db_path,
            program=program or study_plan["program"],
            catalog_key=catalog_key or study_plan["catalog_key"],
            plan=selected_plan or study_plan["plan"],
            year=query_spec.years[0],
            semester=query_spec.semesters[0],
            include_courses=include_courses,
            include_credits=credit_cue or not include_courses,
        )
        next_context = {
            "program": program,
            "catalog_key": catalog_key,
            "plan": selected_plan,
            "study_plan_context": study_plan,
        }
        return {
            "question": request.question,
            "answer": summary["answer"],
            "status": summary["status"],
            "action": None,
            "route": "hard",
            "hard_task_type": "seven_term_plan",
            "provenance": summary["provenance"],
            "next_context": next_context,
            "comparison": None,
        }
    try:
        last_answer = parse_last_answer(
            raw_last_answer, program=program, catalog_key=catalog_key
        )
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
    # An explicit in-text topic replaces the retained one; the retained topic
    # otherwise carries forward as retrieval input only, never as evidence.
    # Like other scope, it is dropped for self-contained exact-course questions.
    semantic_topic = query_spec.topic or raw_semantic_topic
    if semantic_topic is not None and explicit_course_credit:
        semantic_topic = None
    fresh_semantic_collection = bool(
        query_spec.topic is not None
        and not query_spec.course_codes
        and query_spec.course_name is None
        and query_spec.result_ordinal is None
        and not query_spec.references_previous_result_set
    )
    if fresh_semantic_collection:
        # A self-contained semantic collection starts a new query target while
        # retaining stable curriculum identity and any scope in this turn.
        focus_course = None
        parsed_results = None
        last_answer = None
    service_context: dict | None = {}
    if program is not None:
        service_context["program"] = program
    if catalog_key is not None:
        service_context["catalog_key"] = catalog_key
    if semantic_topic is not None:
        service_context["semantic_topic"] = semantic_topic
    if parsed_context is not None:
        retained_plan = (
            query_spec.plans[0]
            if fresh_semantic_collection and len(query_spec.plans) == 1
            else parsed_context.plan
        )
        retained_years = query_spec.years if fresh_semantic_collection else parsed_context.years
        retained_semesters = (
            query_spec.semesters if fresh_semantic_collection else parsed_context.semesters
        )
        retained_operations = (
            query_spec.operations if fresh_semantic_collection else parsed_context.operations
        )
        if retained_plan is not None and not explicit_course_credit:
            service_context["plan"] = retained_plan
        if retained_years and not explicit_course_credit:
            service_context["years"] = list(retained_years)
        if retained_semesters and not explicit_course_credit:
            service_context["semesters"] = list(retained_semesters)
        if retained_operations and not explicit_course_credit:
            service_context["operations"] = list(retained_operations)
        # NOTE: parsed_context.category / parsed_context.course_code are valid
        # public context but are not valid ask_sql service keys; forwarding them
        # raw used to poison every follow-up with invalid_context. The course
        # target is translated into the validated focus_course shape instead;
        # category has no service slot and is intentionally not forwarded.
        if focus_catalog_key is not None:
            service_context["focus_catalog_key"] = focus_catalog_key.strip()
    if focus_course is None and not explicit_course_credit and not fresh_semantic_collection:
        focus_course = _focus_course_from_context_course_code(
            parsed_context, program, catalog_key
        )
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
        and (
            (policy_query.kind == "probation_entry" and policy_query.observed_gpa is not None)
            or policy_query.kind == "probation_value_invalid"
        )
    ):
        policy_result = route_policy_question(
            db_path,
            request.question,
            catalog_key=catalog_key,
            program_context=program,
        )
        if policy_result is not None:
            # Only carry stable scope forward; the deterministic policy
            # comparison does not depend on prior course/result targets.
            stable_context = {
                key: value
                for key, value in (service_context or {}).items()
                if key in {"program", "catalog_key", "plan"}
            }
            return {
                "question": request.question,
                "answer": policy_result.final_answer,
                "status": policy_result.status,
                "action": None,
                "route": "llm_sql",
                "hard_task_type": None,
                "provenance": list(policy_result.provenance),
                "next_context": stable_context or None,
                "comparison": None,
            }

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
            # The requirement total is catalog-global, but an explicit plan
            # remains the user's conversational scope and overrides old scope.
            if policy_query.plan is not None:
                service_context = {**(service_context or {}), "plan": policy_query.plan}
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

    followup_kind = classify_prev_followup(request.question, query_spec)
    if followup_kind is not None and policy_query is None:
        # A substance-free explain/source/rationale turn refers to the
        # immediately previous grounded answer only; it never searches anew.
        followed = answer_previous_followup(
            db_path=db_path,
            question=request.question,
            followup_kind=followup_kind,
            last_answer=last_answer,
            program=program,
            catalog_key=catalog_key,
            model_callable=model_provider,
        )
        followed_status = followed.get("status")
        if not isinstance(followed_status, str):
            followed_status = "error"
        return {
            "question": request.question,
            "answer": followed.get("answer")
            if isinstance(followed.get("answer"), str)
            else "",
            "status": followed_status,
            "action": followed.get("action"),
            "route": "llm_sql",
            "hard_task_type": None,
            "provenance": followed.get("provenance")
            if isinstance(followed.get("provenance"), list)
            else [],
            "next_context": followed.get("next_context"),
            "comparison": None,
        }

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
            and not fresh_semantic_collection
        ):
            hard_context["course_code"] = parsed_context.course_code
        result = None
        # Hard QA receives only this turn's question and validated stable
        # scope. Prior bounded result sets belong to SQL ordinal follow-ups
        # and must not suppress an independently recognized current Hard task.
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
                # Ordinal references (ตัวแรก/ตัวที่ N) resolve against the retained
                # set first; explicit codes/names in the question always win
                # (handled by the question parser downstream); a stored single
                # course target applies only when no ordinal resolves.
                ordinal_target = resolve_ordinal_course_reference(
                    query_spec,
                    parsed_results[0] if parsed_results is not None else None,
                )
                ordinal_code = (
                    ordinal_target.get("course_code").strip()
                    if isinstance(ordinal_target, dict)
                    and isinstance(ordinal_target.get("course_code"), str)
                    and ordinal_target.get("course_code").strip()
                    else None
                )
                has_own_substance = bool(
                    query_spec.course_codes
                    or query_spec.course_name is not None
                    or query_spec.topic is not None
                    or query_spec.plans
                    or query_spec.years
                    or query_spec.semesters
                    or query_spec.program is not None
                    or query_spec.operations
                    or query_spec.category is not None
                    or query_spec.references_previous_result_set
                    or ordinal_code is not None
                )
                if (
                    service_context
                    and service_context.get("result_courses")
                    and not has_own_substance
                ):
                    # The canonical QA context cannot represent an arbitrary
                    # bounded multi-course result set; do not widen it. Bare
                    # follow-ups fail closed here, while ordinal references,
                    # explicit entities, and fresh scoped searches take their
                    # normal paths below instead.
                    return {"status": "insufficient_evidence", "provenance": []}
                focus_course = (service_context or {}).get("focus_course")
                focus_code = (
                    focus_course.get("course_code").strip()
                    if isinstance(focus_course, dict)
                    and isinstance(focus_course.get("course_code"), str)
                    and focus_course.get("course_code").strip()
                    else None
                )
                if (
                    query_spec.result_ordinal is not None
                    and not query_spec.course_codes
                    and query_spec.course_name is None
                    and ordinal_code is None
                ):
                    # An explicit ordinal that resolves to nothing is
                    # terminal: it must never degrade into the focus course.
                    return {"status": "insufficient_evidence", "provenance": []}
                if ordinal_code is not None:
                    grounding_context["course_code"] = ordinal_code
                elif focus_code is not None:
                    grounding_context["course_code"] = focus_code
                # An exact course target (explicit, ordinal-resolved, or focus)
                # takes precedence over a retained topic for this turn; the
                # topic stays retained in context for later refinements.
                has_grounding_course_target = (
                    ordinal_code is not None
                    or focus_code is not None
                    or bool(query_spec.course_codes)
                    or query_spec.course_name is not None
                )
                return answer_question_once(
                    db_path,
                    current_question,
                    intent_model_callable=model_provider,
                    conversation_context=grounding_context or None,
                    semantic_topic=(
                        None
                        if has_grounding_course_target
                        else (service_context or {}).get("semantic_topic")
                    ),
                )

            result = ask_sql(
                db_path,
                effective_question,
                program,
                model_provider,
                model_provider,
                conversation_context=service_context,
                grounding_callable=ground_sql_answer,
                allow_grounded_row_rescue=True,
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
        retained_plan = (
            query_spec.plans[0]
            if fresh_semantic_collection and len(query_spec.plans) == 1
            else parsed_context.plan
        )
        retained_years = query_spec.years if fresh_semantic_collection else parsed_context.years
        retained_semesters = (
            query_spec.semesters if fresh_semantic_collection else parsed_context.semesters
        )
        retained_operations = (
            query_spec.operations if fresh_semantic_collection else parsed_context.operations
        )
        retained_category = (
            query_spec.category if fresh_semantic_collection else parsed_context.category
        )
        if retained_plan is not None and not explicit_course_credit:
            fallback_context["plan"] = retained_plan
        if retained_years and not explicit_course_credit:
            fallback_context["years"] = list(retained_years)
        if retained_semesters and not explicit_course_credit:
            fallback_context["semesters"] = list(retained_semesters)
        if retained_operations and not explicit_course_credit:
            fallback_context["operations"] = list(retained_operations)
        if retained_category is not None and not explicit_course_credit:
            fallback_context["category"] = retained_category
        # NOTE: the prior course target is carried as focus_course below
        # (translated through the focus validator); emitting raw course_code
        # here would re-introduce the invalid_context round-trip poison.
        if focus_catalog_key is not None and not fresh_semantic_collection:
            fallback_context["focus_catalog_key"] = focus_catalog_key.strip()
    if focus_course is not None and not explicit_course_credit and not fresh_semantic_collection:
        fallback_context["focus_course"] = focus_course
    if semantic_topic is not None:
        fallback_context["semantic_topic"] = semantic_topic
    if last_answer is not None and not fresh_semantic_collection:
        # A safe failure preserves the most recent valid grounded-answer
        # referent; the next answered factual turn replaces it.
        fallback_context["last_answer"] = last_answer
    if parsed_results is not None and not explicit_course_credit:
        fallback_context["result_courses"] = parsed_results[0]
        fallback_context["result_scope_program"] = parsed_results[1]
        if not parsed_results[0]:
            fallback_context["result_set_empty"] = True
    reported_context = result.get("next_context", fallback_context or None)
    # Safe failures report next_context=None; re-offer the validated fallback
    # instead of wiping previously validated scope. Explicit payloads (answers,
    # pending catalog selection, hard scope) pass through untouched, and an
    # empty fallback still yields None. The fallback is built only from
    # validated state, never from raw user-supplied context.
    next_context = reported_context or (fallback_context or None)
    if fresh_semantic_collection and isinstance(next_context, dict):
        # The SQL row-derived next context owns the new result set; add back
        # only the sanitized current/stable scope assembled above.
        for key in ("plan", "years", "semesters", "operations", "category"):
            if key in fallback_context:
                next_context[key] = fallback_context[key]

    if status == "answer":
        # A freshly answered policy turn becomes the previous-answer
        # referent only when the policy route actually produced the answer;
        # curriculum answers to dual shapes never inherit a policy referent.
        routed_policy = route_policy_question(
            db_path, request.question, catalog_key=catalog_key
        )
        if (
            routed_policy is not None
            and routed_policy.status == "answer"
            and routed_policy.provenance
        ):
            question_policy = parse_policy_question(request.question)
            policy_reference = (
                build_policy_reference(
                    question_policy,
                    answer_policy_query(
                        db_path, question_policy, catalog_key=catalog_key
                    ),
                    catalog_key=catalog_key,
                )
                if question_policy is not None
                else None
            )
            if policy_reference is not None:
                next_context = {
                    **(next_context if isinstance(next_context, dict) else {}),
                    "last_answer": policy_reference,
                }
        else:
            reported_course = result.get("next_context")
            if isinstance(reported_course, dict) and isinstance(
                reported_course.get("course_code"), str
            ):
                # A freshly answered single-course turn replaces any older
                # referent; follow-up facts still re-ground from authority.
                course_reference = build_course_reference(
                    reported_course["course_code"],
                    reported_course.get("operations"),
                    program=program,
                    catalog_key=catalog_key,
                )
                if course_reference is not None:
                    next_context = {
                        **(next_context if isinstance(next_context, dict) else {}),
                        "last_answer": course_reference,
                    }
    if result.get("route") == "hard" and isinstance(next_context, dict):
        # A hard comparison answer is a new factual turn outside the
        # previous-answer reference contract.
        next_context = {
            key: value for key, value in next_context.items() if key != "last_answer"
        } or None

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
