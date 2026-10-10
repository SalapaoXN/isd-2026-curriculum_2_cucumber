# CUCUMBER curriculum extraction and index builder

This repository is currently reduced to curriculum-source processing and building
the SQLite curriculum index. The user-facing application and question-answering
runtime orchestration have been removed so they can be recreated from this point.

## Current data flow

```text
approved source curriculum documents
  -> OCR / extraction / merge / correction
  -> data/output/final/*_final.json (reviewed finals)
  -> evaluation and canonicalization
  -> data/output/canonical/*_final.json
  -> python -m rag.build_index
  -> cucumber_outputs/runtime/curriculum.db
```

`rag.build_index` reads canonical, post-evaluation curriculum artifacts by default.
The canonical source selection and shared-course conflict checks remain in
`src/pipeline/tools/runtime_artifacts.py`. Structured loading, chunking, embedding,
and SQLite index construction remain under `rag/structured` and `rag/retrieval`.

Build or reuse the curriculum index:

```powershell
python -m rag.build_index
```

Optional explicit JSON paths may be passed to the command. The default sources are
the canonical `data/output/canonical/*_final.json` artifacts. Only curriculum
artifacts are loaded into this index; institutional-policy and program-requirement
supplemental JSON files are not loaded by the current build path.

## Curriculum extraction pipeline

Python 3.10+ is required. Install `requirements.txt`; source processing may require
the OCR and Gemini credentials documented by the pipeline configuration. Never
commit `.env` or API credentials.

Reviewed finals can be evaluated and canonicalized without repeating OCR/correction:

```powershell
python -m src.pipeline.tools.evaluation.evaluate --reports-dir reports/evaluation_precanonical
python -m src.pipeline.tools.canonicalize_runtime
python -m src.pipeline.tools.preflight_runtime
python -m rag.build_index
```

The canonical directory contains the current reviewed/evaluated curriculum records.
`data/output/final/` holds pre-canonical extraction/correction artifacts and
supplemental source JSON. Ground-truth artifacts are used for evaluation and
canonicalization, not as a replacement for extracted descriptions/provenance.

## Remaining directories

- `src/pipeline/` — source extraction, correction, evaluation, canonicalization,
  preflight, and the index build stage.
- `rag/build_index.py` — index CLI and default source selection.
- `rag/structured/` — curriculum JSON to relational SQLite loading.
- `rag/retrieval/` — document chunking, embedding, vector persistence, and index
  construction primitives.
- `data/`, `ground_truth/`, and `reports/` — source, reviewed, and evaluation data.
- `tests/pipeline/`, `tests/tools/`, and index-specific tests — remaining data and
  build-pipeline validation.

The former course Q&A application, answer/retrieval runtime, and application-specific
acceptance suite were intentionally removed for recreation.
