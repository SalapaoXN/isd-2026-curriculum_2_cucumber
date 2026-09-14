# CUCUMBER RAG — Natural QA TODO

## Goal

Improve CUCUMBER RAG to understand natural Thai curriculum questions
without fixing failures with one-off keyword/regex patches.

Development branch: `heart`  
Submission branch: `project_restart`

Follow `AGENTS.md` for working style and Git policy.

Do not modify `project_restart`.  
Do not add/commit/push.

---

## Scope v1

Natural QA v1 = exactly **40 standalone questions**.

Not included:
- conversation history
- previous-turn context
- multi-turn QA
- UI
- LLM-as-a-judge

Design flow:

```text
question
→ normalization
→ entities + operations
→ ambiguity handling
→ retrieval/query
→ composition/aggregation
→ grounded answer
```

`structured / semantic / hybrid` are diagnostic route hints,
not the primary intent model.

---

## Test Spec

Each case:

```json
{
  "id": "nq_001",
  "question": "...",
  "expected": {
    "entities": {
      "program": null,
      "plans": [],
      "years": [],
      "semesters": [],
      "course_codes": [],
      "course_name": null,
      "category": null,
      "topic": null
    },
    "operations": [],
    "judgement": "none",
    "blocking_ambiguity": [],
    "action": "answer",
    "capabilities": [],
    "route_hint": null
  }
}
```

### Operations

`list, describe, count, sum_credits, existence, compare, earliest, placement, prerequisite, similarity`

### Judgement

`none, quantity, workload, preference, unsupported`

### Actions

`answer, clarify_program, no_data, unsupported`

### Capabilities

- `normalize_thai`
- `exact_course_resolution`
- `exact_course_name_resolution`
- `structured_filter`
- `semantic_retrieval`
- `aggregation`
- `comparison`
- `plan_aware`
- `prerequisite_lookup`
- `composition`
- `ambiguity_guard`
- `no_data_guard`
- `unsupported_guard`

### Route hint

`structured | semantic | hybrid | null`

Route hint is diagnostic only.

---

## Locked Policies

### Program

General/topic question without program:

```text
action = clarify_program
```

Do not silently retrieve across multiple programs.

Exact course code/name may work without explicit program only if it can
be uniquely resolved.

If an exact course name resolves to multiple course codes or programs in
the canonical data, require `clarify_program` rather than silently
selecting one. For example, `NOSQL DATABASE SYSTEMS` resolves to IT
`06016414` and DSBA `06026207`.

### Plan

Missing plan is not automatically blocking.

If a known program has multiple plans:
- do not silently choose one
- separate plan results when they differ

### Judgement

`quantity`
→ ground with measurable course count/proportion.

`workload`
→ use curriculum-supported proxies such as course count/credits and say
that these are proxies.

`preference`
→ recommend only from grounded course content.

`unsupported`
→ do not infer difficulty, salary, or unsupported outcomes.

### Entity normalization

- ปีสอง → year 2
- ปีสาม → year 3
- ตอนปีสาม → year 3
- เทอมสอง → semester 2
- เทอมปลาย → semester 2

Do not invent program from a course code.

---

## Canonical Natural QA v1

1. `IT ปี 2 เทอม 1 เรียนอะไรบ้าง`
2. `ปีสองเทอมสองของ IT มีวิชาอะไร`
3. `IT ปี 3 ต้องเรียนกี่วิชา`
4. `ปี 2 ของ IT รวมกี่หน่วยกิต`
5. `IT เทอมไหนหน่วยกิตเยอะสุด`
6. `06016414 เรียนปีไหน`
7. `06016414 เปิดให้ลงช่วงไหนได้บ้าง`
8. `06016414 ต้องเรียนวิชาอะไรมาก่อน`
9. `IT สหกิจกับไม่สหกิจต่างกันยังไง`
10. `06016465 แผนไหนได้เรียนเร็วกว่า`
11. `06016414 เรียนเกี่ยวกับอะไร`
12. `วิชา NOSQL เรียนเรื่องอะไรบ้าง`
13. `IT มีวิชาเกี่ยวกับ AI อะไรบ้าง`
14. `มีวิชาเกี่ยวกับ database อะไรบ้างใน IT`
15. `ถ้าชอบเขียนโปรแกรม IT มีวิชาอะไรน่าสนใจ`
16. `มีวิชาเกี่ยวกับเว็บไหม`
17. `IT ปี 2 มีวิชาเกี่ยวกับ database อะไรบ้าง`
18. `ปี 3 ของ IT มีวิชาเกี่ยวกับ AI กี่วิชา`
19. `IT ปี 2 เทอม 1 มีวิชาเกี่ยวกับ programming ไหม`
20. `ปี 2 มีวิชาเกี่ยวกับคอมพิวเตอร์เยอะมั้ย`
21. `IT ปี 2 มีวิชาเกี่ยวกับคอมพิวเตอร์เยอะมั้ย`
22. `ปีสองของ IT เรียนคอมหนักมั้ย`
23. `IT ปี 2 กับปี 3 ปีไหนมีวิชา programming เยอะกว่า`
24. `เทอมไหนของ IT มีวิชาเกี่ยวกับ database เยอะสุด`
25. `แผนสหกิจมีวิชาเกี่ยวกับ data มากกว่าแผนปกติไหม`
26. `วิชาเลือกของ IT ที่เกี่ยวกับ AI มีอะไรบ้าง`
27. `ปี 3 มีวิชาเลือกเกี่ยวกับเว็บกี่ตัว`
28. `ถ้าอยากเรียน AI เริ่มมีวิชาแนวนี้ตั้งแต่ปีไหน`
29. `ถ้าอยากเรียน database ต้องผ่านวิชาอะไรมาก่อนบ้าง`
30. `06016414 เรียนเรื่องอะไร แล้วเรียนปีไหน`
31. `06016414 กับ 06016419 เนื้อหาคล้ายกันไหม`
32. `06016414 กับ 06016419 ตัวไหนเรียนก่อน`
33. `ปี 4 ต้องเรียนอะไรบ้าง`
34. `IT ปี 4 ต้องเรียนอะไรบ้าง`
35. `เทอมสองปีสามมีตัว database เยอะปะ`
36. `IT ปีสามเทอมปลายเรียนหนักไหม`
37. `วิชาไหนยากที่สุดใน IT`
38. `เรียนวิชาไหนแล้วเงินเดือนสูงสุด`
39. `06019999 เรียนอะไร`
40. `ตอนปีสามของ IT มีวิชาเกี่ยวกับ network อะไรบ้าง`

Do not rewrite these questions to make implementation easier.

---

## Expected Core Behavior by Case

| ID | Operations | Judgement | Action |
|---|---|---|---|
| nq_001 | list | none | answer |
| nq_002 | list | none | answer |
| nq_003 | count | none | answer |
| nq_004 | sum_credits | none | answer |
| nq_005 | sum_credits, compare | none | answer |
| nq_006 | placement | none | answer |
| nq_007 | placement | none | answer |
| nq_008 | prerequisite | none | answer |
| nq_009 | compare | none | answer |
| nq_010 | placement, earliest, compare | none | answer |
| nq_011 | describe | none | answer |
| nq_012 | describe | none | clarify_program |
| nq_013 | list | none | answer |
| nq_014 | list | none | answer |
| nq_015 | list | preference | answer |
| nq_016 | none | none | clarify_program |
| nq_017 | list | none | answer |
| nq_018 | count | none | answer |
| nq_019 | existence | none | answer |
| nq_020 | none | quantity | clarify_program |
| nq_021 | count | quantity | answer |
| nq_022 | count, sum_credits | workload | answer |
| nq_023 | count, compare | none | answer |
| nq_024 | count, compare | none | answer |
| nq_025 | none | none | clarify_program |
| nq_026 | list | none | answer |
| nq_027 | none | none | clarify_program |
| nq_028 | none | none | clarify_program |
| nq_029 | none | none | clarify_program |
| nq_030 | describe, placement | none | answer |
| nq_031 | similarity | none | answer |
| nq_032 | placement, compare | none | answer |
| nq_033 | none | none | clarify_program |
| nq_034 | list | none | answer |
| nq_035 | none | quantity | clarify_program |
| nq_036 | count, sum_credits | workload | answer |
| nq_037 | none | unsupported | unsupported |
| nq_038 | none | unsupported | unsupported |
| nq_039 | describe | none | no_data |
| nq_040 | list | none | answer |

---

## Important Entity Expectations

Normalization that must eventually be represented:

- `ปีสอง` → year 2
- `ปีสาม` → year 3
- `ตอนปีสาม` → year 3
- `เทอมสอง` → semester 2
- `เทอมปลาย` → semester 2

Explicit `IT`:

```text
program = IT
```

Questions without an explicit program must not invent one.

Topic text should preserve the natural concept where practical:

- AI
- database
- programming
- คอมพิวเตอร์
- คอม
- เว็บ
- data
- network

Case `nq_026` must represent the elective/วิชาเลือก category.

Exact 8-digit course codes belong in `course_codes`.

---

# Work Plan

## Phase 1 — Freeze Regression Contract

### 1A Fixture

Create:

`tests/fixtures/natural_qa_v1.json`

Requirements:
- version = `natural_qa_v1`
- exactly 40 cases
- IDs `nq_001`–`nq_040`
- use canonical questions above exactly
- annotate entities / operations / judgement / ambiguity / action / capabilities / route_hint
- no production edits

Validate:
- JSON parses
- 40 unique IDs
- `git diff --check`

STOP for review.

- [x] 1A fixture created
- [x] fixture reviewed

### 1B Spec Validation

After 1A review, create:

`tests/test_natural_qa_spec.py`

Validate:
- version
- 40 exact IDs/questions
- required fields/types
- allowed enums
- clarify_program → program null + ambiguity `["program"]`
- no conversation/multi-turn fields

Run focused unittest only.

- [x] 1B validation test
- [x] contract frozen

---

## Phase 2 — Baseline Audit

Audit current `heart` implementation against all 40 cases.

No production changes.

Classify earliest failure as:
- normalization
- understanding/entity extraction
- ambiguity
- routing
- structured query/filter
- semantic retrieval
- composition
- aggregation/comparison
- judgement
- no-data
- unsupported guard
- answer synthesis
- supported

Group failures by capability.

- [x] baseline audit
- [x] review priorities before coding

---

## Phase 3 — Architecture Design

Goal:
Design the smallest Natural QA v1 architecture supported by the Phase 2 baseline.
No production implementation until the architecture is reviewed and approved.

- [x] 3A QuerySpec contract + Thai normalization semantics
- [x] 3A.1 Comparison/grouping dimension (`group_by`)
- [x] 3B Resolution / ambiguity / no-data / unsupported guards
- [x] 3C Evidence Planner matrix
- [x] 3D Constrained semantic retrieval strategy
  - [x] exact-course direct description retrieval
  - [x] candidate-only topic scoring
  - [x] explicit cosine-distance convention
  - [x] full-candidate scoring; no top-k membership truncation
  - [x] fixed-threshold calibration policy
  - [x] semantic provenance + empty/missing-evidence semantics
- [x] 3E Aggregation / comparison / composition / plan-aware semantics
- [x] 3F Map architecture to Phase 4 microtasks and regression tests
- [x] Architecture reviewed
- [x] Architecture approved before production changes

### Phase 3 constraints

- Do not treat `structured` / `semantic` / `hybrid` as the primary intent taxonomy.
- Do not rewrite the router unless Phase 3 evidence shows it is necessary.
- Reuse the existing SQLite curriculum database and vector index where practical.
- Prefer deterministic operations for filtering, aggregation, comparison, guards, and no-data behavior.
- Use semantic retrieval for course-description/topic matching after structural constraints are known.
- No conversation/session state in Natural QA v1.
- Do not add one-off regex patches for individual questions.

## Phase 4 — Implementation Micro-tasks

Do separately:

- [x] 4A Thai normalization
- [x] 4B entity/operation representation
- [x] 4C missing-program ambiguity guard
- [x] 4D plan-aware behavior
- [x] 4E constrained semantic retrieval
- [x] 4F constrained semantic retrieval + threshold calibration — PASS/FROZEN
  - [x] conservative lexical match OR cosine distance <= 0.4428954516935646
  - [x] calibration: precision 1.0, recall 0.8666666667, F1 0.9285714286
  - [x] accepted INFORMATION RETRIEVAL -> AI/AI-ML false negatives; do not tune specifically for them
  - [x] `no_threshold_matches` state supported
  - [x] one fixed global threshold and deterministic calibration policy
  - [x] no per-topic or dynamic thresholding
- [x] 4G aggregation/comparison/composition — PASS/FROZEN
  - [x] partition-local logical course deduplication
  - [x] list/count/existence
  - [x] `option_count` vs `required_load`
  - [x] authoritative counted-credit aggregation
  - [x] earliest `(year, semester)`
  - [x] deterministic compare
  - [x] complete / valid_empty / insufficient_evidence semantics
  - [x] component/group/member provenance preserved
- [x] 4H grounded judgement and similarity — PASS/FROZEN
  - [x] 4H.1 PASS: judgement/similarity audit
  - [x] 4H.2 PASS/FROZEN: quantity uses deterministic aggregate facts only; no arbitrary many/few threshold; workload uses only course_count / required_load / credits proxies and never infers difficulty; preference preserves grounded course identity, partition, description evidence, and provenance without ranking/best claims; distance is metadata only
  - [x] 4H.3 PASS/FROZEN: exact two-course persisted descriptions/vectors; matching partitions only; selected-plan filtering; separate per-partition results; descriptive mean/min/max distances; no binary similarity threshold/judgement
  - [x] malformed public inputs fail closed as `insufficient_evidence`
  - [x] aggregate semantic types, comparison operands/relations, provenance, supplied similarity evidence, chunk IDs, and selected_plan are validated
  - [x] 4H final regression: 165 focused tests passed

### Phase 4I — Pending Integration / Design

- [x] 4I.1 PASS/FROZEN: integration design
  - runtime orchestration entry point: `rag.qa.ask()`
  - route labels remain diagnostic only; `EvidencePlan` drives execution
  - `ask.py` remains the CLI/UI adapter
  - `hybrid_demo.py` remains a compatibility wrapper; normal runtime must not rebuild the DB/index
  - execution flow: QuerySpec + QueryContext -> resolution/guards -> EvidencePlan -> evidence executor -> aggregation/judgement/similarity -> grounded answer + provenance
  - typed evidence/provenance remains separate from final answer text; an LLM may phrase grounded evidence but may not recalculate or invent deterministic facts
- [x] 4I.2 PASS/FROZEN: QueryContext + Identity
  - immutable `{program: optional, plan: optional}`
  - QuerySpec remains immutable and represents question-derived entities
  - context fills missing scope; matching explicit scope is accepted
  - explicit program/plan conflict => action `context_conflict`
  - add explicit `context_conflicts` metadata; do not reuse `blocking_ambiguity`
  - no context preserves existing 4C ambiguity behavior
  - blocked conflicts stop before routing/retrieval/model
  - program-only context permits multi-plan questions
  - approved controlled QuerySpec extension: operation `identity`
  - deterministic exact resolution only: course name -> course code; exact course code -> canonical course identity/name
  - plan/catalog copies collapse by `(program, course_code)`
  - conflicting name values are preserved as variants and never silently selected
  - no vector search, fuzzy matching, LLM guessing, or program inference from course-code prefix
  - existing name normalization supports `Calculus 1`; no `Calculus I` alias behavior
  - CALCULUS: AIT + `Calculus 1` -> `06046400`; DSBA + `Calculus 1` -> `06026200`; no program context -> `clarify_program`; `Calculus I` -> unresolved/`no_data`
  - final regression: QueryContext PASS; Identity PASS; intent/parser regression PASS; numeric-token safety PASS; 63 focused tests passed
- [x] 4I.3a PASS/FROZEN: EvidencePlan executor design
  - EvidencePlanner may emit symbolic scopes; the executor materializes `APPLICABLE` and `UNCONSTRAINED` axes
  - planner does not build a Cartesian product; explicit grouped plan partitions already emitted separately remain separate
  - different runtime partitions must never be implicitly merged
  - `EvidenceExecutionResult`: `request_id`, `kind`, `planned_request`, `effective_scope`, `status` (`complete` | `valid_empty` | `insufficient_evidence`), typed primitive-specific `payload`, optional `primitive_state`
  - `EvidenceBundle`: original `EvidencePlan` plus ordered tuple of `EvidenceExecutionResult`
  - one symbolic planned request may produce multiple execution results, each with its own concrete `effective_scope`
  - preserve typed evidence/provenance; never flatten to text; result ordering is deterministic
  - `topic_matches` consumes its constrained `course_set` dependency; never global ANN then filter
  - missing glue: generic deterministic scoped `course_set` helper for symbolic/applicable plans, unconstrained/grouped axes, and category constraints
- [x] 4I.3b PASS/FROZEN: relational/direct primitive adapter
  - immutable execution results preserve the original EvidencePlan, planned requests, typed payloads, provenance, and concrete effective scopes
  - plan is always a partition axis; APPLICABLE / UNCONSTRAINED plans materialize as separate DB-backed plan results
  - grouped year/semester axes materialize separately; ungrouped unconstrained year/semester become concrete DB-backed tuples within one relation result
  - explicit multi-values remain together unless grouped; course is normally a relation axis; category is a filter only; no unintended Cartesian/course explosion
  - deterministic parameterized SQLite `scoped_course_set` honors structural scope and exact targets while preserving placement, alternative-group, and provenance evidence
  - direct primitives: `course_set`, `placement_facts`, `credit_facts`, `prerequisite_facts`, `description_evidence`
  - valid_empty remains distinct from insufficient_evidence; dependencies fail closed with deterministic topological execution
  - `topic_matches` remains intentionally pending; legacy global `retrieve()` is never a fallback
  - known unrelated stale tests: placement expectation 669 vs runtime 668; legacy `route="structured"` expectation
- [x] 4I.3c constrained `topic_matches` adapter — PASS/FROZEN
  - `topic_matches` consumes only its declared `course_set` dependency; it never re-queries a broader relation, reparses the question, uses route labels, or calls legacy global `retrieve()`/ANN
  - each materialized dependency partition produces a separate topic result with preserved `effective_scope`, provenance lineage, and deterministic ordering
  - frozen retrieval uses `retrieve_constrained_topic_evidence()` with conservative lexical rescue and inclusive cosine threshold `<= 0.4428954516935646`
  - no recalibration, top-k membership truncation, metadata fallback, or global fallback; lexical rescue with `distance=None` remains valid
  - status mapping: dependency insufficient -> `insufficient_evidence`; empty structural dependency -> `valid_empty` / `empty_structural_candidates`; accepted matches -> `complete`; no accepted matches -> `valid_empty` / `no_threshold_matches`; missing descriptions -> `insufficient_evidence` / `description_missing`; invalid/missing vectors without lexical rescue -> `insufficient_evidence` / `vector_missing_or_invalid`
  - final regression: dependency source PASS; partition preservation PASS; frozen retrieval behavior PASS; status mapping PASS; safety/immutability PASS; 152 focused tests passed
- [x] 4I.3 PASS/FROZEN: complete EvidencePlan graph execution
  - immutable `EvidenceExecutionResult` / `EvidenceBundle` preserve the original EvidencePlan and planned requests
  - typed payload, `primitive_state`, concrete `effective_scope`, and provenance are retained through deterministic topological execution
  - no question reparsing, route-driven retrieval, aggregation, judgement, answer generation, model calls, or global retrieval inside the executor
  - symbolic APPLICABLE / UNCONSTRAINED scopes materialize deterministically; plan is always a runtime partition axis; grouped year/semester remain separate; ungrouped unconstrained year/semester become concrete tuples; course normally remains a relation axis; category remains a filter
  - frozen primitives: `course_set`, `placement_facts`, `credit_facts`, `prerequisite_facts`, `description_evidence`, `topic_matches`
  - `topic_matches` consumes only its declared `course_set` dependency, preserves partition lineage, and retains lexical rescue with inclusive threshold `<= 0.4428954516935646` without global fallback
  - malformed graph/materialization failures may fail request-level; after concrete scopes exist, execution failures are isolated per partition as `insufficient_evidence` / `execution_failure`; successful siblings and independent requests survive
  - dependent topic results inherit only their matching partition state; no cross-partition candidate borrowing
  - final regression: whole-plan execution PASS; lineage/partition preservation PASS; dependency fan-out PASS; partition-isolated failure PASS; mixed primitive bundle PASS; downstream 4I.4 readiness PASS; regression 4I.3a–3c PASS; 155 focused tests passed
  - known unrelated stale tests remain: placement expectation 669 vs runtime 668; legacy `route="structured"` expectation
- [ ] 4I.4 typed grounded answer/provenance integration
  - [x] 4I.4a PASS/FROZEN: typed grounded answer/provenance design
    - runtime answer boundary: `ResolutionOutcome` + `EvidenceBundle` -> deterministic composition -> ordered `GroundedClaim` values -> deterministic rendering and/or bounded grounded synthesis -> `GroundedAnswerResult`
    - `GroundedClaim` is immutable and contains a deterministic stable `claim_id`, operation, `effective_scope`, status, kind (`deterministic_fact` | `grounded_summary`), typed value, typed evidence, and claim-level provenance; typed evidence is never flattened to text
    - `GroundedAnswerResult` contains status (`answer` | `no_data` | `valid_empty` | `insufficient_evidence` | `unsupported` | `clarify_program` | `context_conflict`), answer mode (`deterministic` | `grounded_synthesis` | `mixed`), `final_answer`, ordered claims, and a stable first-seen union provenance; claim provenance is authoritative and top-level provenance is UI/CLI convenience
    - deterministic facts are never recalculated or replaced by an LLM: identity, list/count/existence, credits, placement/earliest, prerequisites, comparisons, quantity facts, and workload proxy facts/relations
    - grounded synthesis is limited to description summaries, topic/preference wording without ranking, two-course similarity explanation without binary similarity judgement, and semantic portions of mixed answers
    - the typed path does not use legacy recovery model calls; the answer layer does not re-query DB/vector data; route labels do not determine correctness
    - partial answers: all complete -> `answer`; complete + `valid_empty` retains both and is `answer` when a complete claim remains; complete + `insufficient_evidence` is overall `insufficient_evidence` while verified partial claims remain; all insufficient -> `insufficient_evidence`
    - blocked resolution statuses remain distinct and bypass normal claim composition; exact no-data fallback remains `ไม่พบข้อมูลนี้ในเล่มหลักสูตร`; valid-empty zero/existence false never becomes `no_data`
  - [x] 4I.4b PASS/FROZEN: deterministic grounded-claim composition
    - immutable `GroundedClaim` / `GroundedAnswerResult`
    - typed values/evidence are preserved; no deterministic recomputation
    - stable claim IDs/order and partition isolation are preserved
    - claim-level provenance is authoritative; top-level provenance is a stable first-seen union
    - partial-result semantics are frozen; zero / `exists=False` remain valid facts
    - identity variants remain preserved
    - similarity preserves left/right provenance, partition, distance/similarity, and descriptive mean/min/max only
    - no binary similarity judgement or threshold; no DB/vector query, question reparsing, route dependency, model, or recovery call
    - final regression: 139 focused tests passed
  - [x] 4I.4c PASS/FROZEN: Grounded Rendering / Synthesis
    - deterministic claims render with Python only
    - `grounded_summary` uses `answer_model_callable` on that claim's grounded evidence only
    - mixed answers render/synthesize claim-by-claim in claim order
    - the model never recalculates or replaces locked deterministic facts
    - describe synthesis uses description evidence only
    - preference synthesis uses grounded options/descriptions only; no ranking/best claim
    - similarity numeric facts render deterministically; synthesis may use only the two descriptions
    - synthesis failure, empty output, non-string output, or exception falls back to deterministic grounded rendering
    - rendering never changes claims, status, or provenance
    - the typed path does not call legacy `answer_question()` recovery logic
    - legacy `answer_question()` remains temporarily for compatibility
    - final focused regression: 109 tests passed
  - [ ] 4I.4d QA + Hybrid Integration / Provenance Freeze
    - topic_matches correctly narrows course candidates
    - dependent credit_facts currently ignores that payload
    - topic-filtered credits therefore overcount structural scope
    - fix belongs in the evidence executor before QA integration
    - preserve non-topic structural credit behavior
    - preserve counted_credit_units, alternative groups, and provenance
    - [x] 4I.4d.0 PASS/FROZEN: Fix topic-filtered credit dependency propagation
      - topic-dependent credit_facts consumes only matched topic candidates
      - partition isolation preserved; no cross-partition borrowing
      - non-topic credits keep full structural-scope behavior
      - authoritative counted_credit_units, alternative groups, components, and provenance remain unchanged
      - regression: 34 passed
      - no blockers
    - [x] 4I.4d.1 PASS/FROZEN: Executor Payload Adapters + Operation Mapping
      - private adapters live in `rag/qa.py`
      - consume `EvidenceBundle` only; no DB/vector re-query
      - `QuerySpec.operations` defines claim order
      - executor order defines partition order
      - one relation may support list/count/existence without new retrieval
      - topic operations use `topic_matches`, never upstream `course_set`
      - preserve `effective_scope`, status, and provenance
      - compare may use only already-created aggregates
      - identity remains resolution-owned
      - similarity remains deferred because it is not in `EvidenceBundle`
      - generic multi-partition greatest/best comparison remains unsupported until a deterministic pairing policy is frozen
      - final cache audit: 61 passed; no blockers
    - [x] 4I.4d.2a PASS/FROZEN: `composed_claims` bridge
      - `compose_grounded_answer()` accepts keyword-only `composed_claims`
      - preserves claim objects/order exactly
      - reuses frozen status, `answer_mode`, validation, and provenance-union logic
      - malformed/mixed inputs fail closed
      - `EvidenceBundle` and identity paths remain unchanged
      - final audit: 36 passed; no blockers
    - [x] 4I.4d.2b PASS/FROZEN: Exact Similarity Execution Bridge
      - similarity uses already-executed `description_evidence`
      - no description refetch
      - stored vectors are compared through existing exact similarity logic
      - partitions and plans remain separate
      - distances, identities, descriptions, and provenance are preserved
      - malformed or missing evidence fails safely
      - no ANN, topic threshold, or new planner primitive
      - final audit: 54 passed; no blockers
    - 4I.4d.2c qa.ask() Typed Pipeline Wiring
- [ ] 4I.5 runtime smoke/regression
- [ ] wire QuerySpec -> resolution -> planner -> retrieval -> aggregation -> judgement/similarity -> grounded answer/provenance
- [ ] keep guided questions for prospective/high-school users in the UI, not inferred by RAG

### Phase 5 — Pending

- [ ] held-out Natural QA 40-question evaluation
- [ ] do not tune architecture/rules from held-out failures
- [ ] report correctness, coverage/abstain, provenance/groundedness, retrieval metrics where applicable, and failure-stage breakdown

Each task must have focused regression tests.

---

## Phase 5 — End-to-End Natural QA v1

Evaluate all 40 cases separately for:
- understanding
- retrieval/evidence
- answer
- provenance
- ambiguity
- no-data
- unsupported claims

Internal route match is not strict correctness.

- [ ] 40-case E2E evaluation
- [ ] failure report reviewed

---

## Current Task

Do ONLY:

**4I.4d.2c — qa.ask() Typed Pipeline Wiring**

Implementation scope is limited to Phase 4I integration design.
Keep Phase 4H frozen while integration is designed.
