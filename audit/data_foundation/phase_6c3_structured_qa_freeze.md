# Structured QA Checkpoint Freeze (Post-Phase-6C.2)

Status: PASS / frozen. No new regressions.

## Baseline

- 184 focused tests run across:
  - `tests/test_rag_qa.py`
  - `tests/test_rag_exact_course_candidates.py`
  - `tests/test_rag_course_placement_integration.py`
  - `tests/test_rag_evidence_executor.py`
  - `tests/test_rag_evidence_planner.py`
  - `tests/test_query_spec.py`
- Result: 179 pass + 5 known pre-existing fixture mismatches (see below).
- Integration: `test_rag_course_placement_integration.py` 35/35 pass.

## Frozen behaviors

- F1 mixed complete/valid_empty exact-course credit scope
  (DSBA 06026259: 3/1, 3/2, 4/1 valid_empty; 4/2 complete value 6).
- F2 source-faithful suffix variants
  (DSBA 06026200: `ไม่ระบุ 1` coop, `แคลคูลัส 1` no_coop).
- M2 flexible-only exact-course credit (IT 06016465 grounded `3`, no model call).
- Exact-course name+credit composition (identity claims plus grounded sums).
- Prerequisite/successor wording contract
  (`ต้องเรียนก่อนวิชาอะไร` = successor semantics with empty relation for 06016420;
  `ต้องผ่านวิชาอะไรมาก่อน` = prerequisites of the course).
- `เตรียมผ่านวิชา` prerequisite trigger.
- Typed `GroundedAnswerResult` integration contract (top-level route `None`;
  no legacy dict results; no obsolete `rag.qa.retrieve` patching).
- Unsupported NL-to-SQL top-level fallback remains fail-closed
  (`insufficient_evidence`, zero model calls).

## Deferred non-blockers (not fixed here)

`tests/test_query_spec.py::test_all_frozen_questions_match_query_spec_contract`
has exactly 5 pre-existing fixture mismatches:

- nq_016
- nq_020
- nq_025
- nq_028
- nq_029

These predate this checkpoint, are unrelated to the Structured QA freeze,
and must be addressed (if at all) as separate fixture work.

## Scope guardrails applied

- Documentation only. No production, test, fixture, DB, or data changes.
- Dirty OCR work, unrelated working-tree changes, and untracked files preserved.
- `submission/` and `project_restart` untouched.
