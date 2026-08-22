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

- Added additive `rubric` evaluation namespace: `overall_text`, `field_level`, `page_level`, and `category_level`.
- Field Level: existing CER/WER is preserved; field-presence coverage and `matched_prediction_field_missing_count` were added.
- Page Level: true per-page evaluation is implemented and requires authoritative GT `source_provenance`.
- Current DSBA GT lacks authoritative per-record page provenance, so DSBA page-level rubric evaluation reports unavailable.
- `code_page_mapping.csv` is not used as authoritative GT.
- Category Level: grouped by GT curriculum category; plan/description is not used as the new rubric category definition.
- Legacy `category_level` remains preserved for backward compatibility.
- Existing CER/WER, coverage, `field_level`, legacy `page_level`, and legacy `category_level` are unchanged.
- Exact-first, fuzzy fallback, and one-to-one matching are unchanged.
- Evaluator suite: 14 tests passed.
- Authoritative GT page annotation is still required for real DSBA Page Level scoring.
- Instructor clarification may still be useful for the exact intended meaning of rubric Category Level.
- Phase 6 is complete for everything supported by current authoritative data.
- Next phase: Phase 7 — Academic Rules OCR / extraction.

## Current Progress

- `src.run_pipeline --pages` is now optional.
- When omitted, valid `<group>_page_<NNN>` images are discovered and sorted numerically.
- Explicit page selection remains supported.
- The program can be derived safely from supported input directories.
- An explicit `--program` still overrides derivation.
- `coop` / `no_coop` is never inferred automatically.
- AIT uses `plan = null` internally; `no_plan` is naming-only.
- GenEd uses `program=GENED`, `plan=gened`.
- IT `coop` / `no_coop` page allocation remains user-supplied; it is not inferred.
- Rules remain outside the course pipeline.
- CLI cleanup verification: 37 unit tests passed.
- No OCR was run for this cleanup.
- The repeated-code description association defect was fixed.
- Automatic description enrichment now occurs only for unique 1-plan/1-description matches.
- Ambiguous repeated groups preserve plan occurrences independently; no association is guessed by occurrence order.
- Unmatched/ambiguous description information is preserved additively in `unresolved_descriptions`; top-level course-count semantics remain unchanged.
- Page-group prerequisite enrichment has the same multiplicity protection.
- IT course counts and coverage remain unchanged after temporary replay.
- Temporary replay prerequisite CER: `coop` `0.0352 -> 0.0295`; `no_coop` `0.0346 -> 0.0290`.
- Repeated-course focused tests: 13 passed; full suite: 45 passed.
- No OCR was run for the repeated-course guard.
- Fully numeric course-code candidates now require exactly eight digits in standalone and embedded extraction; placeholder semantics remain unchanged.
- AIT page-24 truncated code `0604640` is rejected, following valid code `06046408` remains extracted, and description record `06046407` remains unchanged.
- Temporary AIT replay: total courses `57 -> 56`; matched `56 -> 56`; missing `2 -> 2`; extra `1 -> 0`; precision `0.9825 -> 1.0`; recall `0.9655 -> 0.9655`; F1 `0.9739 -> 0.9825`.
- Numeric course-code focused tests: 10 passed; full suite: 51 passed.
- No OCR was run for numeric course-code validation.
- Added standalone `RuleExtractor` for Academic Rules.
- Added `extract_rules.py` for consolidated Rule OCR TXT/JSON extraction.
- Thai and Arabic rule IDs are supported.
- Chapter and sub-rule hierarchy is supported with deterministic parent IDs and section paths.
- Cross-page rule continuation preserves source-page provenance.
- Only explicit rule references are extracted.
- Phase 7A focused tests: 14 passed; full suite: 65 passed.
- Existing Rules OCR replay: top-level rule recovery `41/53 -> 53/53`.
- Existing Rules OCR replay: category-less records `20 -> 8`.
- Existing Rules OCR replay: chapter-9 over-assignment `49 -> 0`.
- Rule anchors tolerate `ข้อ`/`ขอ` OCR variants.
- Wrapped rule anchors recover adjacent-line `ข้อ` plus identifier pairs.
- Explicit anchors only apply contextual `O`/`D -> 0` identifier normalization.
- Split and numberless chapter headings are recovered sequentially.
- Remaining nested misses: `19.3`, `27.1.5`, `31.3`, `45.8`, `51.1`.
- Table false IDs and signature contamination remain unresolved.
- No OCR was rerun; verification replayed existing Rules OCR artifacts.

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
