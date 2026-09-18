# Gold Evaluation Checkpoint Freeze (First Non-Vacuous Baseline)

Status: RECORDED / frozen. First full 30-question Gold run in which the
cross-program similarity case is graded on real content instead of passing
by vacuity. Zero evaluator crashes at freeze time.

## Baseline

- Full Gold run: 30/30 executed, errors 0.
- PASS 20 / REVIEW 10 / FAIL 0.
- provenance 30/30, evidence 30/30, not-found 4/4.
- route_match 24/25 (diagnostic only; route labels do not determine
  correctness).
- semantic evidence coverage 85/85 (includes the 10 newly graded nested
  course descriptions).
- Output recorded outside the repo under Temp
  (`gold_reeval_20260917_postfix.json`).

## Target case

`semantic_cross_program_database_topics_06016402_06026207`
(IT 06016402 × DSBA 06026207, no_coop):

- PASS with non-vacuous grading; evaluator reason: all expected facts
  present in evidence and final answer.
- Complete grounded similarity pair plus per-side description evidence
  for both courses.
- description coverage 10/10, metadata checks 12/12, evidence 22/22,
  answer 18/18, no missing provenance.
- Non-empty user-facing answer comparing both courses' emphases.

## Frozen evaluation behaviors

1. Nested `expected.courses[]` semantic facts are graded per course
   (identity, description evidence, provenance).
2. Empty nested expectations fail closed (no vacuous PASS).
3. Flat semantic grading remains backward-compatible and unchanged.
4. `_semantic_checks()` returns `description_expected` on every path,
   matching the strings actually graded; coverage accounting no longer
   crashes (`KeyError: 'description_expected'` resolved).
5. GT stale OCR references for the target case were corrected to
   authoritative corrected-corpus wording.
6. Production was not changed to match stale GT references.
7. Similarity pair-level partitions carry shared structural scope only
   (`plan`/`plans`); per-side program identity stays on left/right
   evidence, which is what rendering, provenance, and grading consume.

## Deferred (not a blocker, do not act here)

- 10 REVIEW cases remain unchanged and were not investigated in this
  checkpoint.
- route_match 24/25 is diagnostic only.
- Any remaining GT/evaluator quality work must be separate from
  production correctness.

## Scope guardrails applied

- Documentation only. No production, test, GT, evaluator, DB, data,
  OCR, `submission/`, or `project_restart` changes. Unrelated
  working-tree changes and untracked files preserved. No git mutation
  commands used.
