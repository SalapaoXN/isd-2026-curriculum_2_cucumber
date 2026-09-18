# Hybrid QA / End-to-End Grounding Freeze

Status: PASS / frozen, on top of the frozen Corrected → SQLite →
Structured QA → Semantic/Vector chain. Zero correctness blockers
at freeze time.

## Baseline

- Focused suites: 75 passed (`hybrid_demo`, `router`, `grounded_answer`,
  `answer`, `answer_rendering`).
- Six live probes on the frozen runtime DB passed with zero model calls
  where applicable:
  1. Structured-only fact query → scoped SQLite sum (`30`, IT no_coop 2/2).
  2. Exact-course content query → direct scoped description chunk
     (IT 06016418 no_coop, no embedding search).
  3. Mixed name+credit query → identity facts plus per-plan grounded
     credit facts, composed without short-circuiting.
  4. Prerequisite query → grounded deterministic prerequisite evidence.
  5. Similarity query → complete partition-local pair plus supporting
     grounded descriptions.
  6. Ambiguous/unsupported query → `insufficient_evidence`, empty answer.

## Frozen end-to-end behaviors

1. Structured-only facts remain SQLite-first.
2. Exact-course semantic content uses direct scoped chunk retrieval.
3. Mixed name+credit questions compose identity + grounded credit facts
   without short-circuiting.
4. Prerequisite questions return grounded deterministic evidence.
5. Similarity remains partition-local and includes supporting grounded
   descriptions.
6. Ambiguous/unsupported queries fail closed with `insufficient_evidence`
   and empty answer.
7. Structured facts are never replaced by semantic guesses.
8. Semantic retrieval runs only when required.
9. Program/plan/course identity is preserved through planning, execution,
   grounding, and rendering.
10. Every successful claim retains file/page/course/program/plan provenance.
11. Route remains `None` / diagnostic-only and does not determine correctness.

## Deferred hygiene (not a blocker, do not act here)

- Similarity answers may contain duplicate identical describe evidence rows.
  This does not change facts or answer correctness. Deduplication or
  redesign is explicitly out of this checkpoint.

## Scope guardrails applied

- Documentation only. No production, test, DB, data, schema, evaluator,
  or network changes. Dirty OCR work, unrelated working-tree changes, and
  untracked files preserved. D2, query-spec fixtures, global-`retrieve()`
  hygiene, `submission/`, and `project_restart` untouched.
