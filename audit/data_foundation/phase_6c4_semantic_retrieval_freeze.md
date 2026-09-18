# Semantic/Vector Retrieval Baseline Freeze

Status: PASS / frozen, on top of the frozen SQLite (Phase 6C.2) and
Structured QA foundations. No known semantic/vector correctness blockers
at freeze time.

## Baseline

- Focused suites: 68 passed (`retrieve`/`search`, `similarity`/`chunks`,
  `vector_store`/`embedder`).
- Runtime DB chunks: 1667 total — 816 description + 851 metadata.
- Each chunk carries program/plan/course/placement identity plus full
  provenance (source file, source page, document page, category).

## Frozen production retrieval paths (scoped only)

1. Exact-course content: direct description-chunk lookup by course
   identity/`course_id`. No embedding search required. Provenance retained
   (e.g. IT 06016418 no_coop resolves to its description chunk with
   plan + description source pages).
2. Topic/content queries: structural candidates selected first; vector
   scoring restricted to candidate chunks; fixed distance threshold;
   conservative lexical backstop (whole-token ASCII phrase match, plus the
   single AI/artificial-intelligence alias pair).
3. Course similarity: partition-local exact-chunk cosine comparison with
   program/plan/partition mismatch guards; per-course provenance retained;
   summaries must equal pair distances (enforced by constructor).

## Invariants

- Identity scoping occurs before vector work where identity is known.
- No observed cross-program or cross-plan leakage.
- Semantic evidence retains source file/page and course/program/plan identity
  into the final answer.
- Structured curriculum facts remain deterministic/SQLite-first.
- Weak or absent semantic evidence returns `insufficient_evidence` or
  `valid_empty`; no guessing or fabrication (ambiguous no-identity probe
  fails closed with an empty answer).
- Representative exact-content, similarity, and ambiguous probes passed
  with zero model calls where applicable.

## Explicit non-behavior

- Production performs no global ANN followed by post-filtering.
  `retrieve()` (global search + post-filter) has no production callers.

## Deferred hygiene (not a blocker, do not act here)

- Bare/global `retrieve()` remains as dead production surface; it is
  test-only today. Removal or redesign is optional future cleanup,
  explicitly out of this checkpoint.

## Scope guardrails applied

- Documentation only. No production, test, vector/chunk, DB, data, schema,
  evaluator, or network changes. Dirty OCR work, unrelated working-tree
  changes, and untracked files preserved. `submission/` and
  `project_restart` untouched. D2 and query-spec fixtures out of scope.
