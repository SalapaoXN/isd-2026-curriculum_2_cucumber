# Semantic targeted-fix live validation

## Run summary

- Targeted cases: **25/25 evaluated**; infrastructure-excluded: **0**.
- Gemini requests: **42**, retries: **0**.
- Strict runner gold: **11/25 overall correct**.
- Analytical review: **16/25 correct** after separating five proven gold/lexical false negatives (E08, M07, E23, M05, M04).
- Wrong factual answers: **1 (H10)**.
- False failures: **6** (M03, M14a, M14b, H04, H05, H07).
- Presentation-only errors: **2** (E01, E07).
- Correct safe failures: **2** (E12, H06).
- Structural-gap failures: **3** (H04, H05, H07; subset of false failures). H06 additionally fails closed on masked course identities as designed.

## Per-cluster result

- **K1 — relation compatibility / filtered searches:** E14, E15, E22 pass. E23 is semantically correct (GENED resolved through canonical gened plan) but strict gold wrongly required `scope.program`. M03 and M14a still fail because the interpreter emits the topic word (`network` / `cyber security`) both as a literal course target and as a topic filter; deterministic resolution then rejects the nonexistent exact course. M14b consequently has no retained result set and safely cannot resolve its ordinal.
- **K2 — identifiers and wording:** E05 and T-E2i now include code + canonical title; E08 and M07 include course identifiers and year/semester in natural Thai without English `placement` leakage (strict lexical gold matching was over-literal). E01 and E07 still answer `3 หน่วยกิต` without naming the course code/title. No wrong numeric value observed in these two rows.
- **K3 — program as target:** E11 now interprets AIT as the program target and answers from authoritative program-requirement evidence. E12 (ambiguous IT edition without catalog) fails closed.
- **K4 — aggregates:** M05 computes the semester aggregate (18) on the deterministic aggregate path; model added compatible `relation=credits`, which K1 explicitly accepts (strict gold had relation null). H10 remains a **wrong factual answer**: “AIT year 1 total credits” was interpreted as the whole-program total and returned 120 rather than the year aggregate.
- **K5 — comparisons:** H04 fails because comparison-side scope includes unsupported `plan_hint`; H05 fails on missing comparison measure / old-new operands; H07 returns fenced JSON that the strict parser rejects. H06 safely fails closed because the compared course sets contain masked placeholder identities that cannot be treated as canonical course identities. Numeric course/plan comparison execution passed stubbed checks, but did not receive a successful live row here.
- **K6 — plan aliases:** M04’s “แผนสหกิจ” was normalized to `coop`, canonically validated, and returned the scoped BIT list. The old expected safe-failure gold is stale. `no_coop` was not present in this targeted subset.
- **K7 — trace:** latest focused checks show positive total latency on success, safe failure, and validation failure. The targeted runner rounds sub-tick 0.001 ms trace values to 0.0 ms; the run rows therefore cannot independently demonstrate the minimum positive floor.

## Corrected gold cases

| ID | Result |
|---|---|
| E19 | Pass — DSBA course identity/code resolved. |
| M03 | Fail — topic term duplicated as an unresolved literal target. |
| M08 | Pass — retained ordinal resolves deterministically. |
| E25 | Pass — graduation policy is answered from canonical policy evidence. |

## Remaining failed rows and root causes

| ID | Classification | Root cause |
|---|---|---|
| E01 | Presentation-only | Course credit is correct, but answer omits which course it belongs to. |
| E07 | Presentation-only | Course credit is correct, but answer omits the course identifier/title. |
| M03 | False failure / interpretation-resolution | Topic phrase is also emitted as a literal exact-course target. |
| M14a | False failure / interpretation-resolution | Same topic-as-target duplication for “cyber security”. |
| M14b | False failure / context | Turn 1 yielded no verified result set, so ordinal 2 correctly fails closed. |
| H04 | Structural gap | Plan-alias field is unsupported on comparison operands. |
| H05 | Structural gap | Comparison measure/old-new operand representation is incomplete. |
| H07 | Structural gap | Fenced JSON is rejected by the closed parser. |
| H10 | **Wrong factual answer** | Program-total policy route wins over the explicitly scoped year aggregate. |

## Metrics

- Strict gold score: 11/25; analytical score: 16/25 (64%).
- Requests: 42 total, average 1.68/case, max 2/case; 0 retries.
- Paths: deterministic 15, policy 3, no plan / early failure 7, SQL 0.
- System latency: median 2250 ms, p95 2968 ms, max 21407 ms.
- Files/artifacts: `eval/semantic_targeted_fix.jsonl`, this report, and `eval/results/semantic_targeted_fix_summary.json`.

No edits occurred during the targeted live run. No full 50-question run or full unittest suite was run. No git mutation was performed.
