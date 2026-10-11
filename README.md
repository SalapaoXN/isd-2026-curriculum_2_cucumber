# CUCUMBER

**P2: ระบบถาม-ตอบข้อมูลหลักสูตรด้วย LLM**

CUCUMBER เป็นระบบถาม-ตอบข้อมูลหลักสูตรภาษาไทย โดยให้ LLM ช่วยตีความคำถาม แต่ใช้ข้อมูลหลักสูตรและข้อบังคับจากแหล่งที่ตรวจสอบแล้วในการตอบคำถามข้อเท็จจริง หากขอบเขตหลักสูตรหรือหลักฐานไม่ชัดเจน ระบบจะถามเพิ่มหรือแจ้งว่าไม่มีข้อมูลเพียงพอ แทนการคาดเดา

สมาชิก:

1. 67070049 Nattachai Kaewchum — Discord: GoodDee
2. 67070063 Thanachin Chukiatchai — Discord: วันลพ มีงบมาก
3. 67070103 Pongsakorn Panyacom — Discord: เบบี๋คือดวงใจ

## End-to-end flow

```text
source scans in data/input/
  -> EasyOCR
  -> extraction and merge
  -> Gemini course-name correction and review artifacts
  -> evaluation -> canonicalization -> preflight
  -> data/output/canonical/*_final.json
  -> python -m rag.build_index
  -> cucumber_outputs/runtime/curriculum.db
  -> FastAPI + React at / and /curriculum
```

Academic rules are processed separately. `src.pipeline.run_rules` produces
`data/output/rules_extracted.json`, `data/output/final/institution_policy.json`, and
`data/output/final/program_requirements.json`. `rag.build_index` loads the policy and
program-requirement supplements into SQLite tables (`regulation_rules`, `policy_facts`,
and `program_requirements`). Runtime rule retrieval reads those tables; it does not read
the extracted JSON directly.

## Requirements

- Windows PowerShell is used in the commands below.
- Python 3.11 recommended (3.10+ supported) and Node.js 20+.
- Gemini API key for curriculum-name correction and chat generation.
- Source page images under `data/input/<dataset>/`; academic-rule and program-requirement
  images are under `data/input/rule/`.

Supported curriculum datasets: `ait2566`, `bit2565`, `bit2560`, `dsba2565`, `dsba2560`,
`gened2564`, `gened2557`, `it2565`, and `it2560`.

## 1. Set up the environment

Run from the repository root:

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
npm ci --prefix frontend
```

Create a root `.env` file:

```text
GEMINI_API_KEY=your-gemini-api-key
```

Do not commit `.env` or expose the API key. The chat classifier defaults to
`gemini-3.1-flash-lite`; resolver and answer generation use `gemini-3.5-flash-lite`.

### OCR GPU setup

`requirements.txt` installs EasyOCR, PyTorch, and torchvision. For NVIDIA GPU OCR,
verify PyTorch can access CUDA inside `.venv`:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Use `--no-gpu` on a pipeline command to force CPU OCR. CPU processing can take longer.

## 2. Run OCR and curriculum processing

Preview a dataset before processing it:

```powershell
python -m src.pipeline.run --dataset it2565 --dry-run
```

For a full refresh, run GENED before the programs in the same era. Program merges can
backfill empty course descriptions from the corresponding GENED final (`gened2564` for
the 2565/2566 era and `gened2557` for the 2560 era):

```powershell
$datasets = @(
    "gened2564", "ait2566", "bit2565", "dsba2565", "it2565",
    "gened2557", "bit2560", "dsba2560", "it2560"
)

foreach ($dataset in $datasets) {
    Write-Host "===== Processing $dataset ====="
    python -m src.pipeline.run --dataset $dataset --skip-eval
    if ($LASTEXITCODE -ne 0) { throw "Pipeline failed for $dataset" }
}
```

This runs OCR → extraction/merge → Gemini course-name correction and creates reviewed
final artifacts. It requires `GEMINI_API_KEY` and may take time or encounter API quotas.
The pipeline protects existing edition outputs from overwrite; archive the relevant
generated artifacts before intentionally regenerating a dataset. To resume existing
work, use the appropriate `--from extracted` or `--from consolidated` stage.

## 3. Build academic-rule data

When refreshing academic rules or program requirements, run the separate rules pipeline:

```powershell
python -m src.pipeline.run_rules --dry-run
python -m src.pipeline.run_rules
```

It writes `data/output/rules_extracted.json`,
`data/output/final/institution_policy.json`, and
`data/output/final/program_requirements.json`. If rule OCR and extraction already exist,
`python -m src.pipeline.run_rules --skip-ocr` remaps the existing extracted rules without
repeating OCR.

## 4. Evaluate, canonicalize, preflight, and build SQLite

After reviewed curriculum finals and rule artifacts are ready, run these commands in order:

```powershell
python -m src.pipeline.tools.evaluation.evaluate --reports-dir reports/evaluation
python -m src.pipeline.tools.canonicalize_runtime
python -m src.pipeline.tools.preflight_runtime
python -m rag.build_index
```

Evaluation reports go under `reports/evaluation/`. Preflight must pass without unresolved
shared-course conflicts. The index creates `cucumber_outputs/runtime/curriculum.db` with
structured courses, plans, placements, provenance, regulation rules, policy facts,
program requirements, and semantic course vectors. It refreshes the rule/policy tables
from the final supplement JSON files whenever the default index build runs. Rebuild after
curriculum or rule data changes; do not rerun OCR/Gemini if the reviewed source artifacts
have not changed.

## 5. Build and run locally

Build React and start FastAPI from the repository root:

```powershell
npm run build --prefix frontend
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open:

- Chat: `http://127.0.0.1:8000/`
- Curriculum browser: `http://127.0.0.1:8000/curriculum`
- Index status: `http://127.0.0.1:8000/api/health`

Keep that terminal open. For frontend development, leave FastAPI running and start Vite
in another terminal with `npm run dev --prefix frontend`; Vite proxies `/api` to FastAPI.

## 6. Deploy the backend

On the deployment machine:

1. Check out the repository and install the Python and Node dependencies.
2. Supply `.env` securely with `GEMINI_API_KEY`.
3. Copy or generate the canonical curriculum JSON and the final policy supplements.
4. Build the runtime SQLite database and production frontend; start Uvicorn:

```powershell
python -m rag.build_index
npm ci --prefix frontend
npm run build --prefix frontend
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 1
```

`0.0.0.0` permits access from other machines. For an internet-facing deployment, put
Uvicorn behind HTTPS and apply the host's firewall/access controls. Use a service or
process manager to restart the server after reboots; run it from the repository root so
`.env`, `frontend/dist`, and the SQLite database resolve correctly.

Verify deployment:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/curriculum
```

## Diagnostics

Chat responses include response time, retrieval status, and source citations. Uvicorn logs
include request IDs, route/status, result counts, and elapsed time; they omit question and
answer text. For redirected server output, see `cucumber_outputs/runtime/uvicorn.log`
and `cucumber_outputs/runtime/uvicorn-error.log`.

- **Course retrieval unavailable:** rebuild `cucumber_outputs/runtime/curriculum.db`
  after evaluation, canonicalization, and preflight.
- **Academic rule answer unavailable:** confirm SQLite contains `regulation_rules` and
  `policy_facts`; rerun `src.pipeline.run_rules` and `python -m rag.build_index` after
  updating policy sources.
- **Gemini unavailable:** verify `.env` and restart FastAPI after changing it.
- **OCR GPU unavailable:** check `torch.cuda.is_available()` or use `--no-gpu`.

Chat history is stored in the browser's `localStorage`. Recent turns are sent to FastAPI
for context; the backend does not persist conversations.
