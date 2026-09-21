# CUCUMBER FastAPI Web App

FastAPI frontend bridge for the existing CUCUMBER curriculum QA system.

## Architecture

```text
Browser
  -> static/index.html
  -> FastAPI
  -> CUCUMBER RAG
  -> cucumber_outputs/runtime/curriculum.db
  -> Gemini
  -> JSON response
```

## Requirements

Install the main CUCUMBER dependencies from the repository root:

```powershell
python -m pip install -r requirements.txt
```

Then install the Lab 10 web dependencies:

```powershell
python -m pip install -r lab10_fastapi/curriculum_app/requirements.txt
```

## Environment

Create `.env` at the repository root:

```dotenv
GEMINI_API_KEY=your_key_here
```

Do not commit `.env`.

## Run

Run from the repository root:

```powershell
python -m uvicorn lab10_fastapi.curriculum_app.main:app --reload --port 8000
```

## URLs

Web page:

```text
http://127.0.0.1:8000/
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
GET  /
GET  /api/health
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
