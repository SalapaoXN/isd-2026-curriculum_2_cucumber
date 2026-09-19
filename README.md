# Curriculum CUCUMBER — Clean Rebuild

Behavior-preserving rebuild of `isd-2026-curriculum_2_cucumber` with a clean,
modular architecture. The original project is the source of truth for
**behavior**; this project is the source of truth for **architecture**.

## Purpose

OCR curriculum page images → structured curriculum JSON → LLM spelling
correction → evaluation → RAG database → Thai QA (`scripts/ask.py`).

## Architecture

```text
data/input/<program>/*.png
  ↓  src/pipeline/tools/ocr  (EasyOCR th/en + pre-clean + page metadata)
in-memory OCR lines (+ persisted txt/json when run standalone)
  ↓  src/pipeline/tools/extraction  (engine.py / tool.py / rules.py / rule_cli.py)
in-memory extracted docs
  ↓  src/pipeline/tools/merge  (consolidator.py + policy.py)
merged curriculum docs
  ↓  src/pipeline/tools/correction  (Gemini name-only fix, fail-closed)
data/output/final/*_corrected.json  ← FINAL RAG-ready output
  ↓  rag/  (structured + retrieval + grounded answer; reads final/ only)
cucumber_outputs/runtime/curriculum.db
  ↓  scripts/ask.py
```

| Area | Responsibility |
|---|---|
| `src/pipeline/run.py` | ONE entry point; orchestrates tools; no stage logic itself |
| `src/pipeline/tools/ocr` | `input → OCR result` |
| `src/pipeline/tools/extraction` | `OCR result → extracted info` (doc/field/rule/helpers kept separate) |
| `src/pipeline/tools/correction` | `extracted → corrected` (only `name_th`/`name_en`) |
| `src/pipeline/tools/merge` | `corrected → merged` (+ deterministic rules→policy mapping) |
| `src/pipeline/tools/preparation` | scope orchestration per program (`config/programs.yaml` mirror) |
| `src/pipeline/tools/evaluation` | corrected vs ground truth → `reports/` |
| `src/pipeline/tools/indexing` | final JSON → `curriculum.db` (delegates to `rag`) |
| `src/pipeline/config.py` | env (`GEMINI_API_KEY`) + program/plan resolution |
| `src/pipeline/models.py` | output contracts (`CurriculumDocument`, `Course`, `Provenance`) |
| `src/pipeline/utils` | file handling, page metadata, pre-cleaning |
| `rag/providers, retrieval, structured` | RAG subsystem, unchanged behavior, separated from pipeline |
| `data/input`, `data/output/final` | canonical input / final output locations |
| `tests/pipeline, tests/tools, tests/rag` | preserved coverage, reorganized by layer |
| `scripts/` | `ask.py`, `verify_curriculum.py`, `evaluate_gold_questions.py`, `build_conversion_report.py`, `generate_semantic_threshold_dev.py` |
| `reports/` | generated evaluation output (reference snapshot in `reports/evaluation_reference/`) |
| `submission/` | frozen submission package copy |
| `config/` | `programs.yaml` + `pipeline.yaml` (mirror; code is source of truth) |

## Directory structure

```text
Project-recreate/
├── src/pipeline/run.py  (+ config.py, models.py, utils/, tools/ocr|extraction|correction|merge|preparation|evaluation|indexing/)
├── rag/  (providers/, retrieval/, structured/)
├── data/input/  data/output/final/
├── tests/pipeline/  tests/tools/  tests/rag/  (+ fixtures/, conftest.py legacy shims)
├── ground_truth/  reports/  submission/  scripts/  config/
├── cucumber_outputs/runtime/curriculum.db (prebuilt runtime DB copy)
├── .env  .env.example  .gitignore  README.md  AGENTS.md  pyproject.toml  requirements.txt
```

Root contains no implementation scripts — only project-level files.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Copy source images in (not vendored, large binaries):

```powershell
Copy-Item -Recurse ..\isd-2026-curriculum_2_cucumber\inputs\* data\input\
```

Set secrets in `.env` (never commit):

```dotenv
GEMINI_API_KEY=...
HF_TOKEN=...   # optional, Hugging Face models
```

## Run — ONE command

```powershell
python -m src.pipeline.run --program it
```

- Final output: `data/output/final/*_corrected.json` (+ `*_corrections.json`)
- Evaluation included by default (`--skip-eval` to skip → `reports/`)
- `--with-index` continues to `cucumber_outputs/runtime/curriculum.db`
- `--only-index` builds only the DB from existing final files
- `--from {ocr,extracted,consolidated,corrected}` resume points
- `--keep-intermediates` persists `extracted/` + `consolidated/` for debug
- `--dry-run` prints stages without executing
- In-memory flow: intermediates live in temp dirs and are cleaned up

QA (needs DB + `GEMINI_API_KEY` for synthesis polish only):

```powershell
python -m rag.build_index                      # rebuild DB from data/output/final
python scripts/ask.py "IT ปี 2 เทอม 1 เรียนวิชาอะไรบ้าง"
python scripts/ask.py                            # REPL (exit/quit/EOF)
```

## Tests

```powershell
python -m pytest tests/ -q
# or stdlib:
python -m unittest discover -s tests -p "test_*.py"
```

`tests/conftest.py` aliases legacy top-level imports
(`prepare_data`, `extract`, `merge_consecutive`, `llm_spell_corrector`,
`evaluate`, `config`, `src.*`) to the new locations, so preserved tests run
unchanged and verify identical behavior.

## RAG consumption

RAG reads ONLY `data/output/final/*_corrected.json`
(via `rag/retrieval/index.py::llm_source_paths`, with fallback to the original
`outputs/llm/`). `consolidated/` intermediates are never read directly.

## Web-ready (no web app built)

Core logic is plain callable Python (`run.py::main(argv)`, `tool.run_*_stage`,
`rag.hybrid_demo.answer_question_once`) with no web-framework dependency, so a
future `Future Web API → Pipeline Service → Pipeline Tools → Final Output`
layer can call it without rewriting the pipeline. No FastAPI/Flask/frontend
was added.

## Original → New mapping

| Original | New |
|---|---|
| `pipeline.py` | `src/pipeline/run.py` |
| `config.py` + `src/pipeline_config.py` | `src/pipeline/config.py` |
| (implicit JSON schemas) | `src/pipeline/models.py` |
| `src/file_handler.py`, `src/page_metadata.py`, `src/pre_clean.py` | `src/pipeline/utils/` |
| `src/ocr_engine.py`, `src/run_pipeline.py`, `tools/ocr_tool.py`, `ocr.py` | `src/pipeline/tools/ocr/` |
| `src/extractor.py`, `extract.py`, `tools/extract_tool.py` | `src/pipeline/tools/extraction/` (`engine.py` / `tool.py`) |
| `src/rule_extractor.py`, `extract_rules.py` | `src/pipeline/tools/extraction/rules.py`, `rule_cli.py` |
| `llm_spell_corrector.py`, `tools/correct_tool.py` | `src/pipeline/tools/correction/` |
| `merge_consecutive.py`, `tools/merge_tool.py` | `src/pipeline/tools/merge/consolidator.py`, `tool.py` |
| `src/rules_policy_mapper.py`, `map_rules_policy.py` | `src/pipeline/tools/merge/policy.py`, `policy_cli.py` |
| `prepare_data.py` | `src/pipeline/tools/preparation/tool.py` |
| `evaluate.py`, `evaluate_curriculum_layers.py`, `tools/evaluate_tool.py` | `src/pipeline/tools/evaluation/` |
| `tools/build_index_tool.py` | `src/pipeline/tools/indexing/tool.py` |
| `rag/` (all 30 files) | `rag/` (verbatim, only `build_index` help text updated) |
| `ask.py` | `scripts/ask.py` |
| `scripts/` (4 files) | `scripts/` (verbatim) |
| `tests/test_*.py` (53) | `tests/pipeline|tools|rag/` (same files, categorized) |
| `inputs/` | `data/input/` (populate by copy; `.gitkeep` placeholder) |
| `outputs/llm/` | `data/output/final/` (pre-seeded with reference copies) |
| `outputs/ocr|extracted|consolidated` | temp/in-memory; `--keep-intermediates` under `data/output/` |
| `ground_truth/`, `submission/` | verbatim copies |
| `reports/evaluation/` | `reports/` (live) + `reports/evaluation_reference/` snapshot |
| `requirements.txt` + `requirements-rag.txt` | merged `requirements.txt` |
| `cucumber_outputs/runtime/curriculum.db` | preserved copy (prebuilt) |

Removed/consolidated: `src/pipeline/utils/ocr_engine.py` duplicate (canonical
is `tools/ocr/engine.py`); root-level CLIs folded into `run.py`/`scripts/`
(no behavior lost — same flags); `pyproject.toml` added (packaging metadata
only). Nothing else omitted: every production component has a new home.
