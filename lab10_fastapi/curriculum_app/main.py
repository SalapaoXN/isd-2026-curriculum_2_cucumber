"""FastAPI bridge for the CUCUMBER curriculum QA system."""

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from rag.grounded_answer import GroundedAnswerResult
from rag.hybrid_demo import DEFAULT_CURRICULUM_DB_PATH, answer_question_once
from rag.providers.gemini import make_gemini_callable
from .schemas import AskRequest, AskResponse


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = Path(__file__).resolve().parent / "static"

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


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    if not (STATIC_DIR / "index.html").is_file():
        raise HTTPException(status_code=404, detail="หน้าเว็บไม่พร้อมใช้งาน")
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    db_path = Path(DEFAULT_CURRICULUM_DB_PATH)

    return {
        "status": "ok" if db_path.is_file() else "degraded",
        "database": str(db_path),
        "database_ready": db_path.is_file(),
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
        response = answer_question_once(
            db_path,
            request.question,
            structured_model_callable=_lazy_provider,
            top_k=10,
            answer_model_callable=_lazy_provider,
            intent_model_callable=_lazy_provider,
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
    }
