# CUCUMBER FastAPI Backend

FastAPI bridge for the existing CUCUMBER curriculum QA system.

## Architecture

```text
Browser (React SPA in ../frontend/)
  -> FastAPI (backend/main.py)
  -> CUCUMBER RAG
  -> cucumber_outputs/runtime/curriculum.db
  -> Gemini
  -> JSON response
```

The React app lives in `frontend/` (`npm run build` → `frontend/dist/`).
`backend/main.py` serves `frontend/dist/index.html` when built, otherwise the
Vite dev entry, plus SPA routes `/chat` and `/curriculum`.

## Requirements

Install the main CUCUMBER dependencies from the repository root:

```powershell
python -m pip install -r requirements.txt
```

Then install the Lab 10 web dependencies:

```powershell
python -m pip install -r backend/requirements.txt
```

## Environment

Create `.env` at the repository root (see `.env.example`):

```dotenv
GEMINI_API_KEY=your_key_here
```

Do not commit `.env`.

## Run

Run from the repository root:

```powershell
python -m uvicorn backend.main:app --reload --port 8000
```

## URLs

Frontend pages (left navigation: Chat, Curriculum document):

```text
http://127.0.0.1:8000/chat
http://127.0.0.1:8000/curriculum
```

Swagger API docs:

```text
http://127.0.0.1:8000/docs
```

Health check:

```text
http://127.0.0.1:8000/api/health
```

## Current API

```text
GET  / | /chat | /curriculum
GET  /api/health
GET  /api/programs
GET  /api/curriculum?program=&plan=&year=&semester=&search=&limit=&offset=
GET  /api/courses/{course_code}?program=
POST /api/ask
```

Example request:

```json
{
  "question": "IT วิชา 06016414 กี่หน่วยกิต"
}
```

The API uses the existing CUCUMBER runtime database:

```text
cucumber_outputs/runtime/curriculum.db
```
