# Reports — how to read this directory

Nothing in `reports/` is edited by hand. Each subdirectory holds one
frozen evaluation stage: a small summary plus raw supporting evidence.
For background on what was validated, see `docs/data_foundation.md`.

## The four stages

- `eval_consolidated/` — rule-based merged records vs ground truth,
  **before** LLM correction. Baseline accuracy of the deterministic
  pipeline.
- `eval_llm/` — the same comparison **after** LLM correction
  (the `outputs/llm/*_corrected.json` corpus that feeds the runtime
  database). Shows what correction fixed.
- `eval_comparison/` — the delta between the two layers above:
  per-field improvements with zero regressions.
- `evaluation/` — final population check (every ground-truth record
  present exactly once per program/plan) plus CER/WER field accuracy
  for the evaluated corpus.

## Summaries vs raw evidence

Summaries (start here):

- `evaluation/evaluation_summary.csv`
- `eval_comparison/field_delta.csv` (+ `transitions.csv`)
- `eval_llm/summary.json`
- `eval_consolidated/summary.json`

Raw supporting evidence (detail, not narrative):

- `errors.csv` / `evaluation_errors.csv` — per-record mismatch rows
- `field_metrics.csv` — per-field accuracy tables
- `evaluation/evaluation.json` — full machine-readable run record
- `eval_comparison/comparison.json` — full machine-readable delta

## If you read only four files

1. `evaluation/evaluation_summary.csv` — population integrity per
   program/plan.
2. `eval_comparison/field_delta.csv` — what the correction step
   changed (course names fixed, everything else untouched).
3. `eval_llm/summary.json` — field accuracy of the final corpus.
4. `eval_consolidated/summary.json` — the pre-correction baseline
   for comparison.

Do not copy metric tables out of these files; they are the
authoritative record. CSV/JSON detail files exist so any number in a
summary can be traced back to individual records.
