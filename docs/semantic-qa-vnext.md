# Semantic QA vNext — current architecture

This document describes the **current semantic production path**, not an unfinished build phase.

## Core architecture

```text
Student Question
       │
       │  LLM: language interpretation only
       ▼
SemanticIntent (closed schema, zero factual authority)
       │
       │  deterministic validation + current-turn grounding
       ▼
Context Merge
       │
       │  deterministic canonical resolution
       ▼
ResolvedIntent
       │
       │  deterministic planner
       ▼
Execution
  ├─ deterministic evidence
  ├─ verified comparison / mixed-scope / course-set execution
  ├─ canonical policy path
  └─ guarded SQL only for explicitly supported compositional shapes
       │
       │  canonical SQLite = factual authority
       ▼
VerifiedResult + provenance
       │
       ├─ deterministic renderer for typed complex results
       └─ bounded LLM presentation for supported simple results
       ▼
User-facing grounded answer
```

Invariant:

```text
LANGUAGE       → LLM
IDENTITY/SCOPE → deterministic
FACTS          → canonical SQLite
TRUST          → evidence + provenance
PRESENTATION   → verified-facts-only
```

## Current contracts

- semantic intent schema: `semantic-intent/v4`
- interpreter prompt: `semantic-interpreter/v15`
- answerer prompt: `semantic-answerer/v1`
- requested fields include `alternative_selection`, `prerequisite_placement`, and `placement_sequence`
- `literal_set` supports bounded explicit multi-course references
- placement comparison supports explicit plan operands and `available_plans`

The interpreter never supplies canonical course identity or factual values. It supplies linguistic structure and exact current-turn spans; the resolver must verify identity against canonical data.

## Pipeline responsibilities

### `schema.py`
Closed typed intent / resolved / verified structures. Unknown transport fields fail closed.

### `prompts.py`
Versioned semantic-interpreter and answerer contracts. Prompt examples are synthetic/category examples, never benchmark answers.

### `interpreter.py`
One semantic interpretation call with strict JSON parsing/transport behavior. No factual authority.

### `validation.py`
Checks task/subject/relation compatibility, grounding, literal-set/sequence contracts, comparison contracts, and unsupported combinations.

### `context.py`
Merges bounded client-held conversation context. Explicit current-turn scope wins; stale references are invalidated when scope changes.

### `resolver.py`
Canonical program/catalog/plan/course resolution. Also resolves explicit course sets and available-plan inventories deterministically.

### `compiler.py`
Constructs deterministic execution specs from resolved structure. It does not re-parse the student's language.

### `planner.py`
Routes to deterministic, policy, guarded SQL, or unsupported execution. Complex supported shapes have explicit typed contracts rather than sentence-specific routing.

### `executor.py`
Runs canonical evidence and builds `VerifiedResult`. Important current capabilities include:

- multi-field course facts
- explicit course sets
- alternative-group choice verification
- mixed course + enclosing-term composition
- direct prerequisite placement
- placement matrices across plans
- available-plan earliest-placement comparison
- chronological placement sequence with member-local prerequisite facts

### `answerer.py`
Renders typed complex results deterministically where completeness matters, and validates bounded LLM presentation for simpler verified facts.

### `pipeline.py`
End-to-end semantic entry point and trace assembly.

### `modes.py`
`legacy | shadow | semantic` mode switch and API adaptation.

## Supported higher-order semantics

### Explicit course sets

Multiple explicitly named courses use `target.kind="literal_set"`. Every member must resolve independently in the same canonical program/catalog scope. One unresolved member blocks the whole set rather than silently dropping it.

### Alternative selection

Alternative groups are verified from canonical group membership and min/max choice bounds. The LLM never invents group IDs or selection counts.

### Mixed course + term scope

A single question may request facts about one exact course while separately requesting the enclosing semester total. The two scopes are executed independently and then combined atomically.

### Placement comparison across plans

Placement comparison is non-numeric. The executor preserves complete placement sets for every course-plan cell. `difference` means descriptive placement-set contrast, not arithmetic subtraction.

When the user asks which plan to choose without naming plans, the interpreter may request `plan_selector="available_plans"`; the resolver enumerates the canonical plan inventory.

### Placement sequence

For an explicit course set asking “เรียนอะไรก่อนหลัง / เรียงตามปีเทอม”:

1. every course placement is verified independently;
2. direct prerequisite facts remain owned by their target course;
3. ordering is derived only from canonical `(year, semester)` values;
4. chronological order never creates a dependency edge;
5. overlapping flexible placement ranges or same-term ties fail closed when a unique sequence cannot be proved;
6. retained `result_courses` are reordered to match the displayed deterministic sequence so ordinal follow-ups remain stable.

## QA modes

`CUCUMBER_QA_MODE`:

- `legacy` — default safe fallback mode
- `shadow` — legacy answer remains user-visible while semantic trace runs beside it
- `semantic` — current semantic production path

For demo/submission:

```powershell
$env:CUCUMBER_QA_MODE="semantic"
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

`GET /api/health` should report:

```text
status=ok
database_ready=true
qa_mode=semantic
```

## Fail-closed boundaries

Semantic QA intentionally refuses or clarifies when:

- program/catalog/plan/course identity is ambiguous;
- an explicit course-set member cannot be resolved;
- evidence/provenance is incomplete;
- a requested combination has no typed execution contract;
- a placement sequence has no uniquely provable order;
- policy authority lacks a required threshold;
- the user asks for future offering or personal eligibility without canonical authority.

The system must never answer only the easier subset of an accepted multi-field/multi-clause request.

## Testing status

Latest closeout snapshots before the documentation refresh:

- provider-isolated full discovery: **2,804 tests, 0 failures/errors, 3 skipped**
- G5-C sequence suite: **35/35 pass**
- focused semantic regression after retained-result ordering fix: **140/140 pass**
- latest instrumented Gold #30 production-route replay: **end-to-end success**

Historical semantic evaluation reports under `eval/results/` describe earlier checkpoints and should not be used as the current implementation contract.

## Authority reminder

Canonical SQLite + deterministic evidence + provenance are the only factual authority. LLM interpretation and presentation remain untrusted until bounded by deterministic validation and verified facts.
