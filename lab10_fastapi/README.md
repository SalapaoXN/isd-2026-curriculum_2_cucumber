# Lab 10 — CUCUMBER Curriculum API

The curriculum application is a small FastAPI front end for the current
CUCUMBER Natural QA path. It is not a standalone text-to-SQL application.

## What the application uses

Questions are handled by the deterministic QuerySpec, resolution, canonical
SQLite evidence, and grounded-answer pipeline. Answers retain canonical
provenance. Academic Rules and policy evidence are available through the same
grounded backend. Bounded placement and count interpretation may use the
optional Gemini provider only for validated intent proposals; the model never
supplies curriculum facts.

Deterministic questions that the backend can answer do not require
`GEMINI_API_KEY`. The provider is created lazily only if a request needs model
assistance. Missing provider configuration therefore does not prevent import,
startup, health checks, or supported deterministic answers.

## Run

From the repository root, install the curriculum application's dependencies:

```powershell
python -m pip install -r lab10_fastapi/curriculum_app/requirements.txt
python -m uvicorn lab10_fastapi.curriculum_app.main:app --reload --host 127.0.0.1 --port 8000
```

The application uses the canonical runtime database selected by the current
CUCUMBER backend. Check `/api/health` before asking a question.

- UI: <http://127.0.0.1:8000/>
- Health: <http://127.0.0.1:8000/api/health>
- OpenAPI: <http://127.0.0.1:8000/docs>

To enable bounded model interpretation, set `GEMINI_API_KEY` in the
repository-root `.env`. It is optional for deterministic paths and must never
be committed.

## API

`POST /api/ask` accepts:

```json
{"question": "IT ปี 2 เทอม 1 มีวิชาอะไรบ้าง"}
```

The response exposes only the current QA contract:

```json
{
  "question": "...",
  "answer": "...",
  "status": "answer",
  "action": null,
  "route": null,
  "provenance": [{"program": "IT", "source_page": 12}]
}
```

It does not fabricate SQL, database rows, model traces, or other fields that
the grounded QA result does not provide. Unsupported or unresolved questions
retain their fail-closed status. If a required optional provider is
unavailable, the endpoint returns a controlled `503` response.

## Tests

Run the focused API tests from the repository root:

```powershell
python -m unittest tests.test_lab10_curriculum_app -v
```

The core QA tests remain the authority for curriculum semantics:

```powershell
python -m unittest tests.rag.test_rag_qa
```
