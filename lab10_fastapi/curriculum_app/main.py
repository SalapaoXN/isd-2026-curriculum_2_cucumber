"""FastAPI bridge for the CUCUMBER curriculum QA system."""

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from rag.grounded_answer import GroundedAnswerResult
from rag.hybrid_demo import DEFAULT_CURRICULUM_DB_PATH, answer_question_once
from rag.providers.gemini import make_gemini_callable
from rag.structured.queries import capture_sql_queries

from .schemas import AskRequest, AskResponse


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = Path(__file__).resolve().parent / "static"

load_dotenv(PROJECT_ROOT / ".env")

app = FastAPI(
    title="CUCUMBER Curriculum API",
    description="FastAPI frontend bridge for the CUCUMBER curriculum QA system",
    version="1.0.0",
)

app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static",
)

provider = make_gemini_callable()


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")