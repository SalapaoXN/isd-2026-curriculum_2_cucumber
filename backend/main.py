"""FastAPI bridge for the CUCUMBER curriculum QA system."""

import sqlite3
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from rag.grounded_answer import GroundedAnswerResult
from rag.hybrid_demo import (
    DEFAULT_CURRICULUM_DB_PATH,
    answer_question_once,
    parse_conversation_context,
)
from rag.providers.gemini import make_gemini_callable
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


def _lazy_provider(prompt: str) -> str:
    global _provider
    if _provider is None:
        try:
            _provider = make_gemini_callable()
        except Exception as exc:
            raise ProviderUnavailable("optional model provider unavailable") from exc
    try:
        return _provider(prompt)
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
            SELECT program_code, plan_key, plan_name
            FROM curriculum_plans
            ORDER BY program_code, plan_key
            """
        ).fetchall()
    finally:
        con.close()

    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["program_code"], []).append(
            {"plan_key": row["plan_key"], "plan_name": row["plan_name"]}
        )
    return {
        "programs": [
            {"program_code": code, "plans": plans}
            for code, plans in sorted(grouped.items())
        ]
    }


@app.get("/api/curriculum", response_model=CurriculumResponse)
def list_curriculum(
    program: str | None = Query(default=None, max_length=20),
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
def course_detail(course_code: str, program: str | None = None) -> dict:
    db_path = _curriculum_db()
    con = _connect_ro(db_path)
    try:
        if program:
            course = con.execute(
                """
                SELECT c.* FROM courses c
                JOIN curriculum_plans p ON p.catalog_id = c.catalog_id
                WHERE c.course_code_normalized = ?
                  AND p.program_code = ?
                LIMIT 1
                """,
                (course_code.strip().lower(), program),
            ).fetchone()
        else:
            course = con.execute(
                "SELECT * FROM courses WHERE course_code_normalized = ? LIMIT 1",
                (course_code.strip().lower(),),
            ).fetchone()
        if course is None:
            raise HTTPException(status_code=404, detail="ไม่พบรายวิชา")

        placements = con.execute(
            """
            SELECT p.program_code AS program, p.plan_key AS plan_key,
                   pl.year_number AS year, pl.semester_number AS semester,
                   pl.category AS placement_category
            FROM plan_placements pl
            JOIN curriculum_plans p ON p.plan_id = pl.plan_id
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


@app.post("/api/ask", response_model=AskResponse)
def ask(request: AskRequest) -> dict:
    db_path = Path(DEFAULT_CURRICULUM_DB_PATH)

    if not db_path.is_file():
        raise HTTPException(
            status_code=503,
            detail="ไม่พบฐานข้อมูล CUCUMBER",
        )

    try:
        parsed_context = parse_conversation_context(request.conversation_context)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"invalid conversation_context: {exc}") from exc

    try:
        response = answer_question_once(
            db_path,
            request.question,
            structured_model_callable=_lazy_provider,
            top_k=10,
            answer_model_callable=_lazy_provider,
            intent_model_callable=_lazy_provider,
            conversation_context=parsed_context,
        )
    except ProviderUnavailable as exc:
        raise HTTPException(
            status_code=503,
            detail="ตัวให้บริการโมเดลไม่พร้อมใช้งาน",
        ) from exc

    result = response.get("result")

    if isinstance(result, GroundedAnswerResult):
        answer = result.final_answer
        status = result.status
        action = None
        provenance = list(result.provenance)
    else:
        answer = response.get("final_answer", "")
        status = result.get("status", "unknown") if isinstance(result, dict) else "unknown"
        action = result.get("action") if isinstance(result, dict) else None
        provenance = []

    return {
        "question": request.question,
        "answer": answer,
        "status": status,
        "action": action,
        "route": response.get("route"),
        "provenance": provenance,
        "next_context": response.get("next_context"),
    }
