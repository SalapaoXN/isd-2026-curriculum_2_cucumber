# CUCUMBER Curriculum Assistant

CUCUMBER contains the curriculum extraction/index pipeline, a React chatbot, a
curriculum document browser, and a FastAPI service. FastAPI serves the built React
pages at `/` and `/curriculum`. The chat API routes course
questions through the Gemini model, retrieves relevant chunks from the SQLite
curriculum index, and returns source-linked responses.

## Data flow

```text
curriculum sources -> OCR / extraction / merge / correction
  -> reviewed finals -> evaluation -> canonical curriculum JSON
  -> python -m rag.build_index
  -> cucumber_outputs/runtime/curriculum.db
  -> FastAPI retrieval + Gemini answer
  -> React chatbot at http://127.0.0.1:8000/
```

The index defaults to the post-evaluation files in
`data/output/canonical/*_final.json`. It stores structured course information,
source provenance, text chunks, and 384-dimensional vectors. Query-time retrieval
uses the same `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` embedder
and sqlite-vec database.

## Run locally

Python 3.10+ and Node.js 20+ are required. Install dependencies and set `GEMINI_API_KEY`
in a root `.env` file. Never commit `.env` or API keys. A separate lightweight
classifier defaults to `gemini-3.1-flash-lite`; override it with
`CHAT_CLASSIFIER_MODEL` if needed. The answering/rewrite model uses the provider's
default model configuration.

Build the curriculum index if it does not exist yet:

```powershell
python -m rag.build_index
```

Build React and start FastAPI from the project root:

```powershell
cd frontend
npm install
npm run build
cd ..
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/`. The index health endpoint is `/api/health`.

Chat diagnostics are written to the Uvicorn log stream. Each turn logs a short
`request_id`, the classifier's `use_rag` decision, whether database retrieval ran,
result/source counts, final status, and elapsed time. Question text, answer text,
and prompt contents are not logged. Successful responses include the same
`X-Request-ID` header for correlation; errors include the header as well. If Uvicorn
is run in the background with redirected output, inspect
`cucumber_outputs/runtime/uvicorn.log` and `uvicorn-error.log`.

For frontend development, leave FastAPI running and run Vite in another terminal:

```powershell
cd frontend
npm install
npm run dev
```

Vite proxies `/api` requests to the FastAPI server.

## Curriculum extraction and index rebuild

Reviewed finals can be evaluated/canonicalized without repeating OCR or correction:

```powershell
python -m src.pipeline.tools.evaluation.evaluate --reports-dir reports/evaluation_precanonical
python -m src.pipeline.tools.canonicalize_runtime
python -m src.pipeline.tools.preflight_runtime
python -m rag.build_index
```

The index builder currently loads canonical curriculum sources only; policy and total
credit supplemental JSON files are not loaded into this build path. See
`src/pipeline/README.md` for extraction runbooks.

## Project areas

- `frontend/` — React JSX chatbot, curriculum browser, conversation list, citations, filters, and responsive UI.
- `backend/` — FastAPI home/static routes and curriculum RAG query service.
- `src/pipeline/` — OCR, extraction, correction, evaluation, canonicalization, and
  preflight.
- `rag/build_index.py` — index CLI and canonical source selection.
- `rag/structured/` and `rag/retrieval/` — curriculum loading, chunks, embeddings,
  and SQLite vector construction.

Course-specific answers need a built index and Gemini credentials. When evidence is
missing or insufficient, the assistant reports that limitation rather than relying
on model memory.
