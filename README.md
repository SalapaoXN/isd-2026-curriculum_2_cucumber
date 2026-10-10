# CUCUMBER Curriculum Assistant

CUCUMBER turns curriculum and academic-rule scans into a searchable assistant. The
repository includes OCR/extraction tools, curriculum correction and evaluation,
SQLite-backed structured lookups, semantic retrieval, a React chat interface, and a
FastAPI server. FastAPI serves the chatbot at `/` and the curriculum browser at
`/curriculum`.

## End-to-end flow

```text
scanned source pages in data/input/
  -> EasyOCR
  -> structured extraction and merge
  -> Gemini course-name correction and review artifacts
  -> evaluation, canonicalization, and preflight
  -> data/output/canonical/*_final.json
  -> python -m rag.build_index
  -> cucumber_outputs/runtime/curriculum.db
  -> FastAPI + React
```

Academic rules are processed separately. `python -m src.pipeline.run_rules` creates
`data/output/rules_extracted.json`, which the chat service searches for regulation
questions. The course vector index is built from canonical curriculum JSON; course
lists, comparisons, placements, and prerequisites use structured SQLite queries.

## Requirements

- Windows PowerShell is used in the commands below.
- Python 3.10 or newer; Python 3.11 is a good default for EasyOCR/PyTorch compatibility.
- Node.js 20 or newer and npm.
- A Gemini API key for correction and chat generation.
- Source page images in `data/input/<dataset>/`; institution-rule and program-requirement
  images are in `data/input/rule/`.

The supported curriculum dataset keys are `ait2566`, `bit2565`, `bit2560`,
`dsba2565`, `dsba2560`, `gened2564`, `gened2557`, `it2565`, and `it2560`.

## 1. Set up the environment

Run these commands from the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
npm ci --prefix frontend
```

Create a root `.env` file and set your key:

```text
GEMINI_API_KEY=your-gemini-api-key
```

Do not commit `.env` or copy it into a public image/repository. The chat classifier
defaults to `gemini-3.1-flash-lite` and can be changed with `CHAT_CLASSIFIER_MODEL`.
The resolver/answer model defaults to `gemini-3.5-flash-lite`.

### GPU note for OCR

The pipeline uses EasyOCR. To use an NVIDIA GPU, install a CUDA-enabled PyTorch build
compatible with the machine and verify it inside `.venv`:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Use `--no-gpu` on pipeline commands to force CPU mode. CPU OCR can take considerably
longer on a full set of scans.

## 2. Run OCR through reviewed curriculum finals

First preview a dataset run:

```powershell
python -m src.pipeline.run --dataset it2565 --dry-run
```

Then run one dataset end to end. This runs OCR, extraction/merge, Gemini course-name
correction, and evaluation where accepted ground truth exists:

```powershell
python -m src.pipeline.run --dataset it2565
```

Run the other dataset keys as needed. For a complete refresh, the following PowerShell
loop processes all supported curriculum datasets sequentially:

```powershell
$datasets = @(
    "ait2566", "bit2565", "bit2560", "dsba2565", "dsba2560",
    "gened2564", "gened2557", "it2565", "it2560"
)

foreach ($dataset in $datasets) {
    Write-Host "===== Processing $dataset ====="
    python -m src.pipeline.run --dataset $dataset --skip-eval
    if ($LASTEXITCODE -ne 0) { throw "Pipeline failed for $dataset" }
}
```

This performs Gemini correction, so it requires `GEMINI_API_KEY` and may take time or
be subject to API quotas. For an existing OCR run, resume from an appropriate stage,
for example `--from consolidated`. See [`src/pipeline/README.md`](src/pipeline/README.md)
for stage-by-stage/resume commands and artifact details. The pipeline protects existing
edition outputs from accidental overwrite; archive the affected generated artifacts
before intentionally regenerating a dataset.

## 3. Build academic-rule data

Run the separate rules pipeline when refreshing the institutional rules or
program-requirement sources:

```powershell
python -m src.pipeline.run_rules --dry-run
python -m src.pipeline.run_rules
```

It writes `data/output/rules_extracted.json`,
`data/output/final/institution_policy.json`, and
`data/output/final/program_requirements.json`. The chat service uses
`rules_extracted.json` for rule retrieval; this source does not need to be added to the
course vector index. If rule OCR already exists and the extracted JSON is available,
`python -m src.pipeline.run_rules --skip-ocr` rebuilds the mapped rule outputs without
repeating OCR.

## 4. Evaluate, canonicalize, and build the course index

After reviewed `*_final.json` files are ready, create the runtime artifacts in this
order:

```powershell
python -m src.pipeline.tools.evaluation.evaluate --reports-dir reports/evaluation_precanonical
python -m src.pipeline.tools.canonicalize_runtime
python -m src.pipeline.tools.preflight_runtime
python -m rag.build_index
```

The index builder reads `data/output/canonical/*_final.json` and creates
`cucumber_outputs/runtime/curriculum.db`. Preflight must complete without unresolved
shared-course conflicts before building the index. For an ordinary deployment update
where the canonical files have not changed, you do not need to rerun OCR or Gemini
correction; rebuild the index only if curriculum artifacts changed.

## 5. Build and run locally

Build the production React files, then start FastAPI from the repository root:

```powershell
npm run build --prefix frontend
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open:

- Chat: `http://127.0.0.1:8000/`
- Curriculum browser: `http://127.0.0.1:8000/curriculum`
- Health/index status: `http://127.0.0.1:8000/api/health`

Keep the server terminal open while using the application. For frontend development,
leave FastAPI running and start Vite in another terminal:

```powershell
npm run dev --prefix frontend
```

Vite proxies `/api` requests to FastAPI on `127.0.0.1:8000`.

## 6. Deploy the backend server

On the deployment machine:

1. Check out the repository and install the Python and Node dependencies from the
   repository root.
2. Provide `.env` with `GEMINI_API_KEY` through the host's secret/environment
   configuration.
3. Copy or generate the reviewed runtime data: `data/output/canonical/*_final.json`,
   `data/output/rules_extracted.json`, and `data/output/final/program_requirements.json`.
   Build the course database if it is not already included:

   ```powershell
   python -m rag.build_index
   ```

4. Build the frontend and start Uvicorn:

```powershell
npm ci --prefix frontend
npm run build --prefix frontend
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 1
```

`0.0.0.0` allows connections to the host from other machines. For an internet-facing
deployment, place the app behind your HTTPS reverse proxy and apply your normal access
controls/firewall policy. Run Uvicorn under a process manager/service so it restarts
after a machine reboot; keep the working directory at the repository root so `.env`,
the index, and `frontend/dist` resolve correctly.

Verify the deployed service:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/curriculum
```

After changing frontend code, run `npm run build --prefix frontend` and restart
Uvicorn. After changing curriculum data, rerun evaluation → canonicalization → preflight
→ index build before restarting the service.

## Diagnostics

`GET /api/health` reports whether the curriculum index is available. Chat responses
include response time and whether structured/vector retrieval ran. Uvicorn logs include
a request ID, route/status, result counts, and elapsed time, but omit question and answer
text. If the server is started with redirected output, inspect
`cucumber_outputs/runtime/uvicorn.log` and `cucumber_outputs/runtime/uvicorn-error.log`.

Common fixes:

- **Course retrieval unavailable:** build `cucumber_outputs/runtime/curriculum.db` using
  the evaluation/canonicalization/preflight/index sequence above.
- **Rule answer unavailable:** run `python -m src.pipeline.run_rules` and confirm
  `data/output/rules_extracted.json` exists.
- **Gemini unavailable:** verify `.env` contains a valid `GEMINI_API_KEY` and restart
  the server after changing it.
- **OCR fails to use GPU:** verify `torch.cuda.is_available()` in `.venv`, or run the
  OCR command with `--no-gpu`.

## Main project areas

- `data/input/` — curriculum and academic-rule source scans.
- `src/pipeline/` — OCR, extraction/merge, correction, evaluation, canonicalization,
  and preflight.
- `rag/` — structured curriculum loading, text chunks, embeddings, and SQLite vector
  index construction.
- `backend/` — FastAPI chat, course/rule retrieval, comparison, and static frontend
  routes.
- `frontend/` — React chat and curriculum browser.
