# CUCUMBER

CUCUMBER converts curriculum documents into structured data and provides
grounded Thai curriculum question answering with source provenance.

## Architecture

Each stage writes a persistent, replayable artifact boundary. Downstream
stages can restart from an existing boundary; OCR is never rerun implicitly.

```text
Part 1 — OCR
inputs/
  -> python -m src.run_pipeline
  -> outputs/ocr/

Part 2 — Data preparation
outputs/ocr/
  -> python prepare_data.py
  -> outputs/extracted/
  -> outputs/consolidated/
  -> python llm_spell_corrector.py
  -> outputs/llm/
  -> python evaluate.py
  -> reports/evaluation/

Part 3 — RAG / QA
outputs/llm/*_corrected.json
  -> python -m rag.build_index
  -> cucumber_outputs/runtime/curriculum.db
  -> python ask.py
```

## Installation

Run commands from the repository root. Create an environment and install the
project dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt -r requirements-rag.txt
```

On macOS/Linux, activate the environment with `source .venv/bin/activate`.

Create a local `.env` file when using Gemini-backed stages:

```dotenv
GEMINI_API_KEY=...
HF_TOKEN=...
```

`GEMINI_API_KEY` is required by LLM correction and `ask.py` answer generation.
`HF_TOKEN` is optional. Do not commit `.env` or expose either value. OCR and
RAG model dependencies may download their models on first use.

## Part 1: OCR

OCR is intentionally standalone because it is the expensive stage. It reads
images from `inputs/<program>/` and writes only persistent OCR artifacts under
`outputs/ocr/<program>/`.

Example:

```powershell
python -m src.run_pipeline -i inputs/it -p 32-38 --program IT --plan no_coop
```

Use `--no-gpu` for CPU execution. The command also supports the existing
`--pages`, `--input-dir`, `--output-dir`, `--program`, and `--plan` options.

## Part 2: Data Preparation

After OCR artifacts exist, the normal commands are:

```powershell
python prepare_data.py
python llm_spell_corrector.py
python evaluate.py
```

`prepare_data.py` discovers supported, existing, non-empty OCR program
directories and runs the existing extraction and merge stages with explicit
program, plan, and page configuration. Missing programs are skipped; unknown
directories are reported; incomplete scopes are omitted. It does not run OCR,
LLM correction, evaluation, or RAG.

`llm_spell_corrector.py` discovers available full consolidated files matching
`outputs/consolidated/**/full/merged_*_full.json`, in deterministic order. It
ignores page-range and correction-log files, supports partial corpora, and
writes both `*_corrected.json` and `*_corrections.json` to `outputs/llm/`.
LLM correction is the post-extraction text-correction stage.

`evaluate.py` discovers available `outputs/llm/*_corrected.json` files, matches
them to the accepted ground truth for their program/plan, supports partial
corpora, and writes reports under `reports/evaluation/`. Its reports include
CER, WER, and course-record coverage metrics. Coverage Precision/Recall/F1
measures record coverage, not spelling accuracy.

## Part 3: RAG / QA

Build or rebuild the runtime database from the corrected corpus:

```powershell
python -m rag.build_index
```

The source of truth is `outputs/llm/*_corrected.json`. The generated runtime
database is `cucumber_outputs/runtime/curriculum.db`; `outputs/consolidated/`
is not the direct RAG input.

Ask one question:

```powershell
python ask.py "IT ปี 2 เทอม 1 เรียนวิชาอะไรบ้าง"
```

Start interactive mode:

```powershell
python ask.py
```

Interactive mode repeats until `exit`, `quit`, or EOF. Routing between
structured, semantic, and hybrid QA is automatic; users do not select a route.
The runtime database must already exist. If it is missing, run
`python -m rag.build_index`; `ask.py` does not rebuild it silently.

Normal output is concise:

```text
ถาม: <question>
ตอบ: <final answer>
แหล่งข้อมูล: <existing provenance/evidence>
```

When the curriculum does not contain the requested information, the exact
fallback is:

```text
ไม่พบข้อมูลนี้ในเล่มหลักสูตร
```

## Artifact Boundaries / Repository Structure

Important directories are:

```text
inputs/                         source images
outputs/ocr/                    persistent OCR artifacts
outputs/extracted/              extraction artifacts
outputs/consolidated/           merged curriculum artifacts
outputs/llm/                    corrected downstream corpus and logs
reports/evaluation/             evaluation reports
ground_truth/                   accepted evaluation references
cucumber_outputs/runtime/       generated RAG database
src/                            OCR implementation
rag/                            indexing, routing, retrieval, and QA
tests/                          focused and regression tests
submission/                     separate frozen submission package
```

The outputs under `outputs/` and `reports/`, plus the runtime database, are
reproducible pipeline artifacts. `outputs/llm/` is the final downstream corpus
for RAG and may be retained so collaborators can build/query without rerunning
OCR or Gemini correction. The runtime database is reproducible from it, but a
current snapshot may also be retained for immediate demonstration. Exact
tracked/untracked status is repository-specific and is not assumed here.

`submission/` contains a separate frozen submission package; it is not normal
runtime input for preparation or RAG.

## Optional Debugging / Replay

The stage boundaries can be replayed independently when debugging:

```powershell
python extract.py outputs/ocr/it --output-dir outputs/extracted --program IT --plan no_coop
python merge_consecutive.py --prefix it --plan no_coop -p 32-38,328-371 -d 328-371
```

These commands preserve the existing extraction and merge behavior and are
not required for the normal zero-argument workflow. The compatibility
`rag.hybrid_demo` module is also available for development/demo use; `ask.py`
is the normal user-facing interface.

## Evaluation

The evaluator reports text quality with CER/WER and extraction coverage.
Coverage Precision/Recall/F1 measures course-record coverage, not spelling or
text accuracy. Current metric values are produced in `reports/evaluation/` and
are not hard-coded in this document.

## Testing

Run focused suites from the repository root, for example:

```powershell
python -m unittest tests.test_ask
python -m unittest tests.test_prepare_data
python -m unittest tests.test_llm_spell_corrector
python -m unittest tests.test_rag_qa tests.test_rag_hybrid_demo
python -m unittest tests.test_evaluate tests.test_evaluate_gold_questions
```

The full unittest command is also available when a complete regression run is
intended:

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

## Supported Programs / Plans

Current preparation scopes are:

- AIT
- BIT: `coop`, `no_coop`
- DSBA: `coop`, `no_coop`
- GENED
- IT: `coop`, `no_coop`

## Team Members

Member:
1. 67070049 Nattachai Kaewchum >> Discord: GoodDee
2. 67070063 Thanachin Chukiatchai >> Discord: วันลพ มีงบมาก
3. 67070103 Pongsakorn Panyacom >> Discord: เบบี๋คือดวงใจ
