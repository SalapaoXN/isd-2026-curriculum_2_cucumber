# Semantic vNext — clean Round 1 report (50/50 evaluated, 0 infra-excluded)

## Verdict

Clean run **COMPLETE**: 50/50 rows evaluated with live Gemini, zero 429s
(pacing: 3 s gaps + 14/60 s rolling guard; 75 semantic calls, 0 retries).
No code/prompt/data edits during the run. No fixes applied afterwards.

## Scores

- Computed (dataset gold verbatim): **29/50** — Easy 13/25, Medium 9/15, Hard 7/10.
- Analytical (honest recount, §Wrong-gold below): **33/50 (66%)** —
  Easy 15/25, Medium 11/15, Hard 7/10.
- Teacher rows: **10/11** (only T-E2i missed: English name dropped).
- Wrong factual answers: **0**. False failures: **17**. Correct safe: **15**.

## Top lists

- **WRONG FACTUAL ANSWERS**: none. Every answer had correct core facts with
  provenance; every non-answer failed safely.
- **FALSE FAILURES** (safe failure where answer due): E01, E05, E07, E08,
  M05, M07, M14a, M14b, H08, H09, H10, T-E2i, E11, E14, E15, E22, E23.
- **CORRECT SAFE FAILURES**: E12, E21, M04, M09, M11, M12, M13, H01, H02,
  H03, T-E1v, T-E2v, T-M1, T-M2, T-M3v.
- **PRESENTATION-ONLY**: E01 (code missing from "3 หน่วยกิต"), E05/E07
  (identifier dropped), E08/M07 (term detail dropped + Latin "placement"
  leaked — traced to our own `_claim_line` placement template + fallback,
  not only the LLM), T-E2i (English name dropped).
- **STRUCTURAL GAPS**: M04 plan aliases; H05/H06 comparison operators,
  set-difference, course operands, catalog-year mentions; M09 catalog-year;
  M14b ordinal+previous_result_set schema tension.

## Wrong-gold corrections (dataset, not product)

Computed `overall_correct` is unfair in 4 rows where gold — written today,
uncommitted — was provably wrong while behavior was right: E19/M03/M08
(scope/taxonomy over-precision) and E25 (policy/requirement taxonomy +
in-text program omitted from gold scope). Analytical score counts them
correct; the computed 29 stands as recorded.

## Failure clusters (next task — ranked)

- **K1 — spurious relation on non-lookup + closed combo table (7 rows:
  E14, E15, E22, E23, M14a, H08, H09).** Model emits `relation` where none
  belongs (raw payloads captured); validator rejects. Fix options:
  (a) prompt: relation null unless lookup; (b) planner ignores non-lookup
  relations (it already does — validator tolerance). Risk low–medium.
- **K2 — answerer identifier drops + jargon (6 rows).** Fix B's gate
  exists, but our own placement template leaks "placement" and drops
  year/semester detail. Fix: template rewording with real term extraction
  + identifier requirements (already added). Risk low.
- **K3 — program-as-target confusion (E11).** "AIT" read as course literal
  instead of scope. Fix: prompt teaching + validator rescue (literal
  matching a known program code → scope). Risk medium.
- **K4 — aggregate degrades to listing (M05, H10).** Synthesis emits list
  utterances; totals never materialize. Fix: aggregate-capable synthesis
  or deterministic aggregate path. Risk medium.
- **K5 — comparison execution (H04–H07 pending behavior).** Schema now
  represents operators/sides (Fix E verified structurally); execution still
  unsupported → recorded safe failures. Risk medium.
- **K6 — edition guard + program-total route VERIFIED LIVE** (E12 safe,
  E13 = 129). Closed.
- **K7 — leading-scope extraction VERIFIED LIVE** (E01 intent now correct;
  only presentation lagged). Closed pending K2.

Ranked next fixes: K2-template, K1, K4, K3, K5-execution, plan aliases.

## Verified live fixes (A–E)

A: leading program extracted (E01). B: identifier/jargon gate active
(zero jargon from LLM answers; remaining leak is own template — K2).
C: multi-edition-no-catalog fails closed, answerer uncalled. D: program
total from requirement evidence (E13 = 129, policy path). E: comparison
schema parses all 6 ops; sides resolve independently (coop≠no_coop kept
separate in smoke; live compare rows route to safe unsupported).

## Latency / requests

System ms (trace total): min 0 / median 0 / p95 9766 / max 24343 —
fail-closed paths record 0 (trace gap: record wall total on early
returns; fix next). Wall ms: 1062 / 1680 / 9766 / 24343. Hard system:
0 / 0 / 9766 / 9766 (Hard <5 s target: NOT met at p95 — driven by one
24 s outlier + SQL multi-call rows; needs work). Calls: 75 total,
mean 1.5, max 3/question; 0 infra retries; ~77 s paced waiting.
Paths: deterministic 14, policy 6, SQL 7, unsupported 1, pre-plan safe 22.

## Teacher table

All 11 teacher rows recorded with verbatim/template metadata; 10 correct,
1 presentation miss (T-E2i). H01–H03 correctly safe on pending expectations.
Easy/Medium teacher verbatim unavailable locally — custom rows used, no
fabricated attribution.

## Artifacts

`eval/semantic_vnext_50.json`, `eval/results/semantic_vnext_clean_round1.jsonl`,
`..._summary.json`, `..._report.md`. Prior broken/429 round preserved
untouched (`semantic_vnext_round1.jsonl`).
