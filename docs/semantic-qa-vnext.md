# Semantic QA vNext — architecture and operation

Build-phase implementation. Evaluation/debug of the 50-question benchmark is a
separate next phase.

## Old architecture (legacy, frozen at 0764d3e)

student language → deterministic regex parser (`parse_query_spec`) →
bounded LLM fallback → resolver → DB.

The deterministic parser owned language understanding, which forced repeated
phrase-grammar additions for every new student phrasing.

## New architecture (semantic)

```
Student Question
       │  LLM: language / semantic interpretation
       ▼
Structured SemanticIntent (closed schema, zero DB facts)
       │  deterministic: schema + question-grounding validation
       ▼
Conversation Context Merge (explicit turn > validated context > unknown)
       │  deterministic: canonical program/catalog/plan/course resolution
       ▼
ResolvedIntent (canonical IDs only, still fact-free)
       │  deterministic: plan deterministic vs guarded-SQL vs unsupported
       ▼
Query Planner ──┬── deterministic executor (frozen evidence machinery)
                └── guarded SQL executor (guard/scope/verify unchanged)
       │  canonical SQLite: FACTUAL AUTHORITY
       ▼
VerifiedResult (claims + provenance + missing info)
       │  LLM: natural-language presentation of verified results only
       ▼
Grounded Answer
```

Invariant: LANGUAGE → LLM. IDENTITY/SCOPE → DETERMINISTIC.
FACTS → DATABASE. TRUST → EVIDENCE. PRESENTATION → LLM.

## Authority boundaries

- The interpreter proposes language only (task/subject/relation, literal
  spans, scope mentions, filters, aggregation, ranking, comparison). It can
  never emit course codes-as-identity, credits, thresholds, SQL, or answers.
- Normalized hints (e.g. nickname spellings) only generate lookup
  candidates; v1 never accepts them as identity (`allow_hint_candidates`
  defaults False).
- Program/catalog/plan/course identity is created exclusively by
  deterministic resolvers against canonical tables.
- Aggregates are computed by deterministic aggregation or verified SQL
  results, never by the answerer.
- The answerer receives verified facts + missing-information only, with a
  deterministic fallback renderer when the provider fails.

## Code layout (`rag/semantic/`)

`schema.py` (closed intent/resolved/verified types + failure taxonomy),
`prompts.py` (versioned interpreter/answerer prompts),
`interpreter.py` (one call, strict parse, no retries),
`validation.py` (combo + question-grounding checks),
`context.py` (precedence merge + stale invalidation),
`resolver.py` (canonical bridges reusing `exact_course_candidates`),
`compiler.py` (ResolvedIntent → constructed QuerySpec; canonical utterance
synthesizer for the SQL adapter),
`planner.py` (deterministic / SQL / policy / unsupported routing),
`executor.py` (frozen evidence + policy + guarded `ask_sql` bridges),
`answerer.py` (evidence-bounded render + deterministic fallback),
`pipeline.py` (`semantic_answer` entry point, latency + call counts),
`modes.py` (mode switch, shadow logging, API adaptation),
`trace.py` (JSON-safe observability), `eval.py` (benchmark harness).

The semantic path never calls `parse_query_spec(question)` for language
understanding. QuerySpec is constructed from resolved structure only.

## Mode switch

`CUCUMBER_QA_MODE`: `legacy` (default) | `shadow` | `semantic`.
Unknown values fall back to legacy with a warning.

- legacy: frozen behavior; semantic code never runs.
- shadow: legacy response unchanged (same answer/claims/provenance/context);
  the semantic pipeline additionally runs and records a comparison entry
  through internal logging only; shadow failures never affect the user.
- semantic: `semantic_answer` serves `POST /api/ask` with the same
  `AskResponse` fields. Missing provider key fails closed, never 500s.

`GET /api/health` exposes non-sensitive `qa_mode`.

## Running modes

```powershell
$env:CUCUMBER_QA_MODE="legacy"    # default
$env:CUCUMBER_QA_MODE="shadow"    # compare via logs (logger cucumber.semantic_shadow)
$env:CUCUMBER_QA_MODE="semantic"  # semantic production path (needs GEMINI_API_KEY)
```

## Running evaluation (next phase)

```powershell
python -m rag.semantic.eval <db> <dataset.json> <out.jsonl>
```

Dataset format: `{version, cases[]}` with `id/difficulty/source/question`,
optional `conversation_context` + `stubs`, and `expected` stage constraints
(`intent/resolved/plan/status/facts_contain/safe_failure`). Stage grading:
`intent_correct, resolution_correct, query_correct, fact_correct,
safe_failure_correct`; one failure category per case
(`INTERPRETATION/RESOLUTION/QUERY/DATA/GROUNDING/ANSWER_ERROR`,
`EXPECTED_SAFE_FAILURE`, `NONE`).

Seed: `eval/semantic_vnext_seed.json` (custom phrasings only).
Teacher-slide verbatim rows (25/15/10 across Easy/Medium/Hard) are pending
source input; `ground_truth/rag/gold_questions.json` (30 rows, no slide
markers) is the identified candidate source — import via
`import_gold_candidates()` with source `gold_candidate_pending`, never as
`teacher_slide`, and never into interpreter few-shot prompts.

## Provider-call expectations

Simple question: interpreter (1) + answerer (1) ≈ 2 calls.
SQL question: + SQL generation/summary inside the guarded bridge.
Trace records `llm_request_count` and per-stage milliseconds; the benchmark
compares accuracy/latency/count between legacy and semantic.

## What remains factual authority

Canonical SQLite + deterministic evidence + provenance (HSQL rescue,
aggregate verification, COUNT semantics, plan isolation all reused
unchanged above the new pipeline). LLM output is never factual authority.
