"""FastAPI application; serves the React chatbot at the site root."""

from pathlib import Path

from fastapi import FastAPI, Query, Response
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

from .rag_service import answer, health
from .schemas import ChatRequest
from .curriculum import filters as curriculum_filters, list_courses

FRONTEND_DIR = ROOT / "frontend"
DIST_DIR = FRONTEND_DIR / "dist"

app = FastAPI(title="Cucumber Curriculum Assistant", version="1.0.0")

if (DIST_DIR / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=DIST_DIR / "assets"), name="assets")


@app.get("/api/health")
def api_health():
    return health()


@app.get("/api/curriculum/filters")
def api_curriculum_filters(
    search: str = Query(default="", max_length=200),
    program: str = Query(default="", max_length=40),
    academic_year: str = Query(default="", max_length=20),
    plan: str = Query(default="", max_length=80),
    category: str = Query(default="", max_length=200),
    course_type: str = Query(default="", max_length=100),
    study_year: str = Query(default="", max_length=10),
    semester: str = Query(default="", max_length=10),
):
    return curriculum_filters(search=search.strip(), program=program, academic_year=academic_year,
                              plan=plan, category=category, course_type=course_type,
                              study_year=study_year, semester=semester)


@app.get("/api/curriculum")
def api_curriculum(
    search: str = Query(default="", max_length=200),
    program: str = Query(default="", max_length=40),
    academic_year: str = Query(default="", max_length=20),
    plan: str = Query(default="", max_length=80),
    category: str = Query(default="", max_length=200),
    course_type: str = Query(default="", max_length=100),
    study_year: str = Query(default="", max_length=10),
    semester: str = Query(default="", max_length=10),
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0, le=1000000),
):
    return list_courses(search=search.strip(), program=program, academic_year=academic_year,
                        plan=plan, category=category, course_type=course_type,
                        study_year=study_year, semester=semester, limit=limit, offset=offset)


@app.post("/api/chat")
def api_chat(body: ChatRequest, response: Response):
    result = answer(body.question.strip(), body.history)
    response.headers["X-Request-ID"] = result["request_id"]
    return result


@app.get("/", include_in_schema=False)
def home():
    built = DIST_DIR / "index.html"
    if built.is_file():
        return FileResponse(built)
    dev_entry = FRONTEND_DIR / "index.html"
    if dev_entry.is_file():
        return HTMLResponse(
            "<!doctype html><html><body style='font:16px sans-serif;padding:3rem'>"
            "<h1>Cucumber frontend is not built</h1>"
            "<p>Run <code>cd frontend &amp;&amp; npm install &amp;&amp; npm run build</code>, "
            "or start <code>npm run dev</code> while developing.</p></body></html>", status_code=503,
        )
    return HTMLResponse("Frontend files are missing.", status_code=404)


@app.get("/{path:path}", include_in_schema=False)
def spa_fallback(path: str):
    if path.startswith("api/"):
        return HTMLResponse("Not found", status_code=404)
    built = DIST_DIR / "index.html"
    if built.is_file():
        return FileResponse(built)
    return home()
