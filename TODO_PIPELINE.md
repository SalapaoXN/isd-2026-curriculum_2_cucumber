# Prioritized Pipeline Roadmap

## Phase 1 — Diagnose Missing GT Courses

### Objective

Explain why 8 ground-truth courses are missing from the **DSBA coop benchmark only** by tracing each record through the full pipeline:

`source image -> OCR -> extracted JSON -> merged JSON -> evaluator matching`

The current DSBA coop benchmark is **GT 90 / Predicted 82 / Matched 82 / Missing 8**.

### Why now

Current CER/WER evaluates matched records only, so the 8 missing DSBA coop records must be traced before making OCR or parser changes. The cause may be unavailable source evidence, OCR output, extraction, merge behavior, duplicate or alternative-code handling, or evaluator matching.

### Minimal work

- Build a per-record trace for all 8 missing DSBA coop GT records.
- Record the GT code/row, expected source page, OCR evidence, extracted record, merged record, and evaluator match outcome.
- Check whether each record disappears at the source-image, OCR, extracted-JSON, merged-JSON, or evaluator-matching stage.
- Check duplicate, placeholder, and alternative-code behavior without modifying ground truth.

### Verification

- Account for all 90 DSBA coop GT records and all 82 DSBA coop predictions.
- Explain the outcome of every missing DSBA coop record.
- Reproduce the DSBA coop result of 90 GT, 82 predicted, 82 matched, and 8 missing.
- Produce evidence identifying the first stage at which each record is lost or fails to match.

## Phase 2 — Fix Confirmed Code/OCR/Extractor Causes

### Objective

Correct course-code, OCR, or extractor defects only when Phase 1 proves they cause missing DSBA coop records.

### Why now

Targeted evidence avoids speculative parser changes and prevents unrelated regressions.

### Minimal work

Apply only the smallest confirmed fixes to normalization, OCR cleanup, or extraction rules.

### Verification

Rerun the traced DSBA coop pages and show that the missing-record count improves without changing ground truth.

## Phase 3 — Improve Thai/English OCR Text

### Objective

Improve Thai and English course-name text quality.

### Why now

After missing-record causes are separated from text-quality errors, OCR improvements can target measured failures.

### Minimal work

Use sampled Thai/English OCR errors to add narrowly scoped preprocessing or extraction corrections.

### Verification

Compare Thai and English name CER/WER on the same benchmark before and after the changes.

### Current Progress

- The English-only second-pass prototype at `experiments/dsba_english_second_pass.py` remains experiment-only.
- The placeholder-code boundary fix is complete. Normalized placeholders matching `^\d{4,8}X+$` terminate title regions, but are not exact-code anchors or associations.
- Development holdout after the fix: canonical CER `43/1162 = 0.037005`; candidate CER `1/1162 = 0.000861`; association coverage `47/52`; improved `28`; worsened `0`; unchanged `19`; exact regressions `0`; exact improvements `28`; candidate exact `46/47`.
- The remaining OCR-only error is `90644007`: `FOUNDATION ENGLISH` versus GT `FOUNDATION ENGLISH 1`.
- Production integration review: PASS. The opt-in production path is available through `python -m src.run_pipeline --english-second-pass`.
- Independent unseen production validation: unavailable. No unused course-bearing DSBA pages remain; the remaining unused page is non-course content.
- The prototype previously passed an independent unseen holdout. That result is prototype evidence only and must not be reported as production regression evidence.
- The production path is now undergoing regression/equivalence validation on previously validated course-bearing DSBA sets. These regression results are not unseen validation.
- This development holdout is no longer considered unseen because it was used to debug the heuristic.

## Phase 4 — Add Evaluation Coverage Metrics

### Objective

Report missing, extra, precision, recall, and F1 alongside the existing CER/WER metrics.

### Why now

Matched-only quality scores can overstate pipeline quality when records are absent or extra.

### Minimal work

Expose unmatched GT and prediction records plus match diagnostics while preserving the existing CER/WER output.

### Verification

- Added GT count, prediction count, matched, missing, extra, precision, recall, and F1 coverage metrics.
- Existing exact-first, fuzzy fallback, and one-to-one matching remains unchanged.
- Existing CER/WER behavior remains unchanged.
- Six focused tests passed.
- DSBA validation: GT `90`; predictions `89`; matched `89`; missing `1`; extra `0`; precision `1.0`; recall `0.9889`; F1 `0.9944`; CER `0.0188`; WER `0.1433`.
- These DSBA benchmark numbers must not be generalized to other datasets without their own evaluation.
- Phase 4 is complete.
- Next phase: Phase 5 — Preserve Source/Page Provenance.

## Phase 5 — Preserve Source/Page Provenance

### Objective

Retain source-file and page provenance for each extracted and merged course.

### Why now

Provenance is required for later Page Level evaluation and source citations.

### Minimal work

Carry input filename, page number, and relevant OCR references through extraction and merge.

### Verification

- Added per-record `source_provenance` with `program`, `source_filename`, `source_page`, and `document_category`.
- Provenance is source-derived only; no GT or `code_page_mapping.csv` data is used.
- OCR metadata now preserves the original filename, page, and program.
- Legacy and explicit TXT/JSON inputs remain supported.
- Plan + description merges union provenance in source order.
- Co-op alternatives preserve provenance from all contributing records.
- Repeated same-code records remain independent.
- Duplicate provenance entries are removed.
- English second-pass provenance remains separate.
- `evaluate.py` remains unchanged.
- Tests: provenance `9` passed; English enrichment `8` passed; evaluator compatibility `6` passed.
- Phase 5 is complete.
- Next phase: Phase 6 — rubric-aligned Field/Page/Category evaluation.

## Phase 6 — Align Field/Page/Category Evaluation with the Rubric

### Objective

Define rubric-compliant Field, Page, and Category evaluation and aggregation.

### Why now

Coverage semantics and provenance must be established before evaluation levels can be aligned reliably.

### Minimal work

Document official evaluation inputs and aggregation rules. Treat `code_page_mapping.csv` as project-generated helper data, not official ground truth. Do not modify ground truth.

### Verification

Rubric scenarios produce the expected Field, Page, and Category results using clearly defined inputs and matching rules.

## Phase 7 — Extract Academic Rules Separately

### Objective

Build a dedicated academic-rule extraction path separate from course extraction.

### Why now

Academic rules have different structures and evaluation needs from course records.

### Minimal work

Define rule fields, extraction boundaries, provenance, and evaluation inputs without coupling them to course extraction.

### Verification

Extract and evaluate a representative rule-page sample against the applicable rule ground truth.

## Phase 8 — Prepare Retrieval/RAG

### Objective

Prepare provenance-preserving records for retrieval and RAG.

### Why now

Reliable structured content, evaluation, and page metadata are prerequisites for useful retrieval.

### Minimal work

Define chunking, metadata, source/page identifiers, and a retrieval benchmark.

### Verification

Known course and rule queries retrieve the relevant passages with source and page metadata.

## Phase 9 — Build LLM Q&A with Source/Page Citations

### Objective

Answer curriculum questions with grounded source and page citations.

### Why now

LLM Q&A should consume validated retrieval artifacts rather than unverified OCR output.

### Minimal work

Implement a narrow Q&A flow that returns answers tied to retrieved evidence and citations.

### Verification

Benchmark questions cite the correct source/page and reject or qualify unsupported answers.

## Phase 10 — Consider Structured Course-Description Extraction if Needed

### Objective

Extract deeper structured course-description fields only if later requirements justify them.

### Why now

This is lower priority than record coverage, evaluation, rules, retrieval, and cited Q&A.

### Minimal work

Define the required fields and run a limited pilot only when a concrete downstream use exists.

### Verification

Demonstrate a measurable retrieval or Q&A benefit before expanding the extraction scope.

## Current Task

Run a new frozen unseen holdout validation before production integration.

## Why Next

The development holdout validated the boundary fix but is no longer unseen. A new frozen holdout is required before production integration.

## Blockers

Production integration remains blocked until the new unseen holdout is validated.
