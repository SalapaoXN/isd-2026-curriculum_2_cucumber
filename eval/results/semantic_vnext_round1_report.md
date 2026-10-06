# Semantic vNext Round 1 — baseline report (PARTIAL: quota-blocked)

## Verdict

**Round 1 INCOMPLETE — provider quota exhausted after 9 of 50 rows.**
Free-tier `generate_content` limit (15) hit during E09–E10; rows E10–H10
carry provider-failure signatures and are marked unevaluated, not failed.
No second run was performed in this task. No accuracy tuning was performed.

## Coverage (9 evaluated)

- Overall correct: **4/9** (E02, E03, E04, E06)
- Wrong factual answers: **0** — every non-answer was a safe failure; every
  answer carried correct core facts with provenance.
- False failures: **2** (E01, E09) — safe failures where an answer was due.
- Answer errors: **3** (E05, E07, E08) — correct evidence, degraded presentation.
- Latency ms (evaluated): min 1500 / median 2375 / p95 2563 / max 2578.
- LLM calls: total 16, mean 1.78, max 2. Paths: deterministic 7, none 2, SQL 0.

## Per-row corrected verdicts (recorded evidence, no new calls)

| ID | Status | Verdict | Category | Note |
|----|--------|---------|----------|------|
| E01 | insufficient | incorrect | INTERPRETATION_ERROR | Leading "AIT" mention dropped by model → ambiguous, safe failure OK |
| E02 | answer | **correct** | NONE | Prereq 06046400, program via validated context (gold scope corrected: turn mentions only) |
| E03 | answer | **correct** | NONE | Credit 3, canonical |
| E04 | answer | **correct** | NONE | Identity 06046401, canonical |
| E05 | answer | incorrect | ANSWER_ERROR | Code present, English name dropped by answerer |
| E06 | answer | **correct** | NONE | Description, code present |
| E07 | answer | incorrect | ANSWER_ERROR | Value 3 correct, course code dropped by answerer |
| E08 | answer | incorrect | ANSWER_ERROR | Code present, term detail dropped; Latin op word "placement" leaked into Thai answer |
| E09 | insufficient | incorrect | INTERPRETATION_ERROR | Leading "IT" mention dropped by model → unresolved, safe failure OK |

A grading-shape mismatch in the first analysis (flat trace summary vs
nested dataset gold) initially mislabeled E02–E08; the harness trace shape
was fixed (nested intent + raw payload recording) and verdicts above use
byte-level answer checks. Dataset gold needed no changes (turn-scope fields
were already correct).

## Failure clusters (for the NEXT task — no fixes applied)

- **Cluster A — leading-position program mentions dropped (E01, E09).**
  Mid-sentence mentions (E04/E06/E07) survive; sentence-initial ones do
  not. Layer: interpretation. Smallest generic fix: strengthen prompt scope
  extraction (leading-position examples) — prompt change, re-eval required.
  Risk: medium.
- **Cluster B — answerer drops identifiers / leaks jargon (E05, E07, E08).**
  Verified evidence intact; presentation omits codes/names/terms and emitted
  "placement" verbatim. Layer: answer. Smallest generic fix: answerer
  identifier-citation hardening + output validation (identifiers present,
  Thai-only). Risk: low–medium.
- **Cluster C — edition-ambiguous totals answer cross-edition sums.**
  Pre-round stubbed finding (not live data): multi-edition program total
  without catalog answers instead of failing closed. Smallest generic fix:
  edition-clarify guard mirroring legacy `clarify_catalog`. Risk: low.
- **Cluster D — program totals answered from placement fragments.**
  Pre-round stubbed finding: lookup/program/credits routes to placement
  sums, never program-requirement evidence. Smallest generic fix: planner
  program-total → policy requirement route. Risk: low–medium.
- **Cluster E — comparison operator gap (STRUCTURAL_GAP).**
  Schema has left/right/measure only: greater/less/equal/difference,
  set-difference/overlap, and course operands unrepresentable. No redesign
  performed. Risk of fix: medium (schema + planner + executor).
- **Cluster F — plan aliases (M04-shape, pending live).**
  Resolver lacks plan-alias mapping (สหกิจ→coop). Risk: low–medium.

Ranked next-task order: D, C, B, F, A, E.

## Teacher questions

H01–H03 recorded with verbatim wording, unevaluated (quota). Easy/Medium
teacher verbatim unavailable locally — all 47 non-teacher rows are custom
compositions (never used in prompts).

## Quota accounting

Pipeline-counted live calls: 16 (9 evaluated rows) + ~4 prior-task accidental
+ provider-level transient retries. No further live calls in this task.
Resume requires quota reset or upgraded key; then run remaining 41 rows once
with the fixed harness (nested trace shape already in tree, uncommitted).
