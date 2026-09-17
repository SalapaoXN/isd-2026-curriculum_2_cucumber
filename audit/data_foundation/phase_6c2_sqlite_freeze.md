# Phase 6C.2 Corrected → SQLite Freeze (Runtime) and Submission Version Boundary

Status: runtime PASS / frozen. Submission intentionally unchanged (see boundary below).

## 1. Runtime Corrected → SQLite freeze (PASS)

- Build path: `python -m rag.build_index` (defaults) → `llm_source_paths()`
  (`outputs/llm/*_corrected.json`, 8 documents) → `load_jsons_to_sqlite` →
  `cucumber_outputs/runtime/curriculum.db` (loader + chunks + local embeddings).
- Recorded source fingerprints match current corrected files (prior 9/13 fingerprints
  were stale, which justified the rebuild).
- Population: 841 placements = 841 corrected records.
  Per plan: AIT 56, BIT coop 63, BIT no_coop 63, DSBA coop 89, DSBA no_coop 90,
  GENED 266, IT coop 106, IT no_coop 108. Placement order dense 1..N per plan.
- Provenance: 841/841 placements and 816/816 courses covered, 0 missing pages
  (182 description / 57 plan entries).
- Prerequisites: 57 total — 52 resolved to courses, 3 group-based, 2 raw-only
  fail-closed (`BASIC CREATIE STEM DESIGV`, `FOUNDATION ENGLISH` — OCR fragments
  matching no course; raw text preserved, nothing inferred).
- Determinism: two independent temp builds byte-identical; production loader
  introduces no GT/evaluation references and no SQL UPDATE patching.
- D1 resolved through SQLite: DSBA coop and no_coop `06026260` (standalone
  placements and the `06026259 หรือ 06026260` group member) all store the approved
  `OVERSEA COOPERATIVE EDUCATION IN DATA SCIENCE AND BUSINESS ANALYTICS`.
  All other Phase 6C.2 pins/preserves verified in SQLite.

## 2. Submission version boundary (intentionally NOT synchronized)

- `submission/` is a separate historical frozen IT-only package
  (212 placements: IT coop 105 / no_coop 107), not a runtime input.
  Its DB metadata records sources under `outputs/consolidated/it/...`.
- It is NOT synchronized to the current corrected/runtime layer, by decision.

## 3. Why submission is intentionally not refreshed

- No supported propagation path exists: no script builds `submission/curriculum.json`
  from corrected IT sources (only readers/validators exist). Prior propagation
  (`75092db`) was manual.
- Independent re-alignment of corrected IT (106/108) vs `submission/curriculum.json`
  (105/107) does not reproduce the previously reported "198 changed aligned records"
  figure: code-group full-field alignment yields 180/212 differing (18 on
  names+credits alone; 179 on first-record comparison; 200 on naive order pairing).
  The 198 figure is therefore recorded as reported-but-unreproduced, not as verified.
  In every alignment the delta is broad, so automatic propagation would risk importing
  unrelated historical changes, and no authoritative manifest isolates Phase 6C.2-only
  submission changes.

## 4. Deferred issues (not blockers for this runtime freeze)

- D2 placeholder-code identity limitation: 16 wildcard (`xxx`/`xxxxxxxx`) groups with
  genuinely divergent elective titles collapse to first-record titles in `courses`
  (placements preserved). Pre-existing loader architecture; needs a separate design
  decision, not a freeze-blocker — corrected JSON layer itself is faithful.
- Stale DSBA corrections audit-log entry: `merged_dsba_coop_full_corrections.json`
  still logs the `06026259 หรือ 06026260` entry as `…OVERSEAS…` while the corrected
  file holds the approved `…OVERSEA…`. Log-only staleness; SQLite reads the file,
  so production values are correct. Do not clean up here.

## 5. Scope guardrails applied

- No edits to correction policy, tests, corrected JSON, databases beyond the runtime
  rebuild, loader/schema, D2 behavior, corrections logs, OCR/geometry files, or
  `project_restart`. Dirty OCR work and unrelated untracked files preserved.
