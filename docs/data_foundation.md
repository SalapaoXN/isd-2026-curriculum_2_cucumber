# Data-Foundation Validation — Summary

This note summarizes the validation work that used to live under
`audit/data_foundation/` (since removed from the repository).
It is a record of what was checked and what the frozen state is,
not a debugging log.

## Why this validation existed

Answers in this system are only as trustworthy as the structured
curriculum data behind them. Before freezing the runtime database,
every pipeline stage from OCR text to SQLite was checked for
identity preservation: no record may silently change program, plan,
course, placement, credit, or provenance on its way through.

Pipeline stages checked:

```text
OCR -> Extracted -> Consolidated -> Corrected -> SQLite
```

The checked corpus at manifest time: 239 OCR files, 247 extracted
files, 2065 course records (DSBA 276, BIT 226, GENED 1064, AIT 154,
IT 345).

## Invariants enforced

- **Program/plan isolation.** Records never merge across programs or
  across coop/no_coop plans. Per-plan placement populations are
  counted explicitly (e.g. runtime: AIT 56, BIT 63+63, DSBA 89+90,
  GENED 266, IT 106+108).
- **Course/placement identity.** Course codes are validated as
  8-digit strings; one catalog course may have several plan
  placements and each placement keeps its own year/semester identity.
  Placement order is dense per plan.
- **Credit integrity.** Credit strings are only ever filled in from
  authoritative source records for the exact same program, plan,
  course code, source file, and source page. Fragmentary OCR credit
  tuples are left unresolved rather than guessed.
- **Provenance/page traceability.** Every placement and course carries
  source file, source page, and document page. A deterministic
  regeneration pass brought all 209 canonical physical pages to full
  document-page agreement (209/209, zero wrong pages) without
  re-running OCR and without changing any extracted semantics.
- **Deterministic regeneration.** Two independent runtime builds from
  the corrected corpus are byte-identical; the loader performs no
  SQL UPDATE patching and references no ground-truth data.
- **Fail-closed behavior.** Unresolvable references keep raw text and
  resolve to nothing instead of being inferred (e.g. two OCR-fragment
  prerequisites matching no course). Weak or absent evidence returns
  `insufficient_evidence` / `valid_empty`, never a fabricated answer.

## Defects actually found and resolved

- **Missing document-page propagation.** 642 of 2065 extracted
  records lacked `document_page`. Resolved by deterministic
  regeneration from a canonical page map: 209/209 pages agree, zero
  wrong pages, zero semantic changes. Thirty legacy OCR files without
  source-page identity were deliberately left untouched, as were raw
  OCR artifacts (478 checked, 0 changed).
- **Two genuinely missing credits fixed from source.** IT 06016454
  and BIT 06036135 had empty credits caused by split/incomplete OCR
  tuples; both were corrected to `3(3-0-6)` using agreeing
  authoritative academic-plan records for the exact program/plan/page.
- **Stale build fingerprints.** 9 of 13 source fingerprints did not
  match the corrected files, which justified the runtime rebuild.
- **GENED name suffix artifacts.** Four GENED English titles carried
  a stray trailing `3` (e.g. `SCIENCE OF BURGER 3`); corrected with
  zero correct-to-wrong transitions anywhere in the corpus.
- **One DSBA placement wording pin.** DSBA coop/no_coop `06026260`
  (including the `06026259 หรือ 06026260` group member) stores the
  approved `OVERSEA COOPERATIVE EDUCATION …` title through SQLite.

Known and explicitly deferred (not blockers): 16 wildcard-code
elective groups collapse to first-record titles in `courses`
(placements preserved); one stale DSBA corrections log line;
a dead global `retrieve()` kept test-only; possibly duplicated
identical similarity evidence rows.

## Final frozen / validated state

- **Runtime DB** (`cucumber_outputs/runtime/curriculum.db`, built by
  `python -m rag.build_index` from `data/output/final/*_corrected.json`,
  8 documents): 841 placements = 841 corrected records,
  provenance complete (841/841 placements, 816/816 courses),
  57 prerequisites (52 resolved, 3 group-based, 2 raw-only).
- **Structured QA:** focused suites pass (including 35/35 placement
  integration); unsupported queries stay fail-closed with zero model
  calls. Five query-spec frozen-fixture mismatches
  (nq_016/020/025/028/029) predate the freeze and are recorded as
  separate fixture work.
- **Semantic/vector:** 1667 chunks (816 description + 851 metadata),
  each carrying program/plan/course/placement identity and full
  provenance; identity scoping happens before vector work; no
  cross-program/plan leakage observed.
- **Hybrid QA:** end-to-end probes pass; structured facts stay
  SQLite-first and are never replaced by semantic guesses.
- **Gold evaluation:** final full 30-question run passes every case
  (provenance 30/30, evidence 30/30, semantic coverage 85/85),
  including a non-vacuous cross-program similarity case.

## Relationship to the submission and runtime inputs

- `data/output/final/*_corrected.json` is the corrected downstream corpus
  and the RAG source of truth; `cucumber_outputs/runtime/curriculum.db`
  is generated from it deterministically. (`outputs/llm/` retains only
  legacy/historical copies and is not production authority.)
- `submission/` is a separate historical IT-only package
  (212 placements) and is intentionally **not** synchronized to the
  current corrected/runtime layer — no supported propagation path
  exists, and automatic propagation would risk importing unrelated
  historical changes.
