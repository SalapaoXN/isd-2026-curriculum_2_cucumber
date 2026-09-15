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

### Phase 4I — PASS/FROZEN

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
  - [x] 4I.4d PASS/FROZEN: QA + Hybrid Integration / Provenance Freeze
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
    - [x] 4I.4d.2c PASS/FROZEN: qa.ask() Typed Pipeline Wiring
      - blocked outcomes keep the existing metadata dictionary for now
      - normal and identity answers use `GroundedAnswerResult`
      - similarity uses one `GroundedClaim` containing the original `SimilarityEvidence`
      - one partition -> claim `effective_scope` is that concrete scope
      - multiple partitions -> claim `effective_scope=None`; each `SimilarityPair.partition` is authoritative
      - preserve original pairs, descriptions, distances, summaries, and provenance
      - multi-partition similarity renders deterministically; no synthesis/model judgement
      - no similarity threshold or binary similar/not-similar result
      - `qa.ask()` uses the exact similarity bridge once after `EvidenceBundle` execution
      - `hybrid_demo.py` and `ask.py` remain deferred
      - `qa.ask()` uses the typed RAG pipeline; blocked outcomes stop before execution/model calls
      - identity bypasses `EvidencePlan`; normal flow is plan -> execute -> adapters -> compose -> render
      - legacy route/retrieve/answer paths are not used
      - status, evidence, and provenance survive end-to-end
      - final audit: 105 passed; no blockers
    - 4I.4d.3 Hybrid / CLI Compatibility Adapters — PASS/FROZEN
      - `hybrid_demo.py` calls `qa.ask()` exactly once
      - typed `GroundedAnswerResult` remains authoritative
      - hybrid compatibility output exposes `final_answer`, provenance, and status
      - blocked dictionary results remain unchanged
      - typed path never calls legacy `answer_question()` or a recovery model
      - `answer_model_callable` is forwarded into `qa.ask()`
      - `ask.py` remains display-only and reads provenance from `GroundedAnswerResult`
      - normal runtime never calls `ensure_index()`
      - explicit `source_json_path` compatibility mode may still call `ensure_index()`
      - route labels do not control correctness
      - final audit: 56 passed; no blockers
    - [x] 4I.4d.4 PASS/FROZEN: End-to-End Provenance / Status Regression
      - 10 real-runtime probes completed; 9/10 runtime-flow cases passed
      - exact similarity failed because executor scope materialization expands each singleton description request across both course targets
      - planner `request.course_targets` are already correct
      - root cause: `_materialize_scopes()` uses broad `request.scope.course_targets` instead of `request.course_targets`
      - fix belongs in `rag/evidence_executor.py`
      - no planner, `qa.py`, similarity bridge, or frozen contract change needed
      - normal runtime made 0 extra model calls and no index rebuild
      - request-level course target materialization is fixed
      - runtime similarity now reaches persisted vector access
      - the vector table is a `vec0` virtual table
      - `compare_stored_vectors()` opens SQLite without loading `sqlite-vec`
      - `score_candidate_vectors()` has the same hidden issue
      - normal ANN paths already load `sqlite-vec` correctly
      - fix belongs in `rag/retrieval/vector_store.py`
      - no architecture or frozen contract change required
      - probe 1 underlying typed result, status, and provenance are correct
      - CLI final answer becomes blank
      - `ask.py` display cleaner removes the whole answer line when it contains the word `provenance`
      - fix belongs to CLI display only; RAG core is unchanged
      - [ ] 4I.4d.4d Fix typed CLI answer cleaning
      - `qa.ask()` returns the correct blocked `no_data` dictionary
      - `hybrid_demo` preserves it unchanged
      - `ask.py` incorrectly expects `final_answer` and prints blank
      - fix belongs to CLI display only
      - required exact fallback: `ไม่พบข้อมูลนี้ในเล่มหลักสูตร`
      - [x] 4I.4d.4e PASS/FROZEN: Fix blocked `no_data` CLI display
        - blocked `no_data` displays the exact fallback `ไม่พบข้อมูลนี้ในเล่มหลักสูตร`
        - other blocked and typed display behavior remains unchanged
        - final audit: 22 passed; no blockers
      - final real-runtime probes: 10/10 passed
      - end-to-end status, provenance, and similarity: PASS
      - exact similarity returns numeric `coop` + `no_coop` partitions
      - `no_data` exact fallback works
      - blocked metadata preserved
      - model calls: 0
      - index rebuild calls: 0
      - legacy retrieval/recovery calls: 0
      - focused regression: 267 passed
      - no blockers
    - [x] 4I.4d.4b PASS/FROZEN: Load sqlite-vec on direct vector read paths
      - sqlite-vec direct-read issue is fixed
      - persisted coop/no_coop descriptions and vectors are correct
      - executor currently reuses the representative `course_id` across plans
      - no_coop description evidence therefore incorrectly uses the coop chunk
      - similarity validation correctly rejects the mismatched provenance
      - fix belongs in `rag/evidence_executor.py`
      - each concrete effective scope must remap logical `(program, course_code)` to its plan-specific `course_id` using the existing `scoped_course_set()` helper
      - no planner/index/vector/similarity contract change is needed
    - [x] 4I.4d.4c PASS/FROZEN: Fix plan-specific description course mapping
      - plan-specific course mapping and provenance are correct
      - no-plan similarity now has 2 aligned partitions
      - regression: 114 tests passed; no blockers
- [x] 4I.5 PASS/FROZEN: Runtime Smoke / Regression Freeze
  - 10/10 natural runtime probes passed
  - all QueryContext smoke cases passed
  - repeated-query stability passed
  - `valid_empty` / zero / `exists=False` passed
  - sqlite-vec and coop/no_coop isolation passed
  - unexpected model calls: 0
  - index rebuild calls: 0
  - legacy route/recovery calls: 0
  - runtime exceptions: 0
  - focused regression: 329 passed
  - latency baseline: p50 27.60 ms, p95 248.11 ms, max 338.06 ms
  - no blockers
  - [x] 4I.5a PASS/FROZEN: QueryContext propagation into parsing/planning
    - missing program can be supplied by `QueryContext`
    - plan remains optional
    - `plan=None` keeps applicable plans separate
    - `plan=coop`/`no_coop` restricts evidence to that plan only
    - no-context still returns `clarify_program`
    - explicit conflicts still return `context_conflict`
    - final audit: 72 passed; no blockers
  - resolved prerequisite blocker: QueryContext is resolved correctly but was not fully propagated into parsing/planning
  - resolved Case A: a year/semester-only question dropped its detected `list` operation when program was absent from the text; fixed in `rag/query_spec.py`
  - resolved Case B: the planner ignored `resolved_plans=("coop",)` when `QuerySpec.plans` was empty; fixed in `rag/evidence_planner.py`
  - frozen expectation: program context may supply a missing program; plan remains optional; no selected plan keeps applicable plans separate; `plan=coop`/`no_coop` uses only that plan; no-context ambiguity remains unchanged
  - use the existing 10 natural runtime probes
  - valid_empty remains covered by controlled regression, not a new natural or Gold question
  - verify the normal CLI path, QueryContext, provenance, and repeated-query stability
  - deterministic queries must produce zero unexpected model calls
  - normal runtime must produce zero `ensure_index()` or rebuild calls
  - legacy route and recovery paths must not run
  - sqlite-vec direct reads and coop/no_coop isolation must work
  - runtime exceptions must be zero
  - no Gold/final evaluation and no threshold tuning
  - freeze criteria:
    - all 10 runtime probes match expected statuses
    - required answers are non-empty
    - exact `no_data` fallback is preserved
    - similarity has numeric coop/no_coop evidence
    - repeated identical queries produce identical typed results
    - focused regression suites pass
    - `git diff --check` passes
- [x] wire QuerySpec -> resolution -> planner -> retrieval -> aggregation -> judgement/similarity -> grounded answer/provenance
- [x] keep guided questions for prospective/high-school users in the UI, not inferred by RAG

### Phase 5 — Pending

- [x] Phase 5A — Prerequisite Plan Isolation — PASS/FROZEN
  - proven failure: IT `06016420`; coop is course_id `635` -> prerequisite `624`, while no_coop incorrectly uses `635` -> `624` instead of course_id `739` -> prerequisite `728`
  - no_coop provenance must also be plan-specific
  - root cause: `rag/evidence_executor.py::_execute_prerequisites()` uses the representative `request.course_targets` course_id directly and does not remap logical `(program, course_code)` to the physical row for each concrete effective scope
  - `prerequisites_of_course()` is correct and requires no change
  - fix: resolve logical targets against the concrete effective scope, use the plan-specific physical course_id, then call `prerequisites_of_course()`; follow the existing description remapping principle
  - regression: one prerequisite request spanning coop + no_coop must assert separate scopes, plan-specific physical course/prerequisite IDs, plan-specific provenance, and no cross-plan leakage
  - GOLD blocker: yes
  - [x] Phase 5A.1 — Implement prerequisite per-plan physical-row remapping
    - prerequisite execution remaps logical `(program, course_code)` to plan-specific physical course rows
    - coop `06016420`: `635 -> 624`; no_coop `06016420`: `739 -> 728`
    - provenance remains plan-specific; coop/no_coop scopes remain isolated
    - shared scoped remapping does not regress description execution
    - focused audit: 4 passed; no blockers
- [x] Phase 5C.1 — Comparison Missing-Producer Fail-Closed Guard — PASS/FROZEN
  - requested `compare` with no `ComparisonAggregation` fails closed with status `insufficient_evidence`
  - no relation, value, evidence, or provenance is fabricated
  - existing comparison composition behavior is preserved
  - focused QA tests: 2 passed
  - `git diff --check`: passed
  - no commit performed
- [x] Phase 5C.2 — Comparison Operand / Scope Policy Audit — AUDIT COMPLETE
  - Plan comparison: POLICY REQUIRED; plan-scoped course/placement evidence exists, but generic “ต่างกันยังไง” has no frozen metric and remains fail-closed
  - Earliest placement comparison: supported by existing `EarliestAggregation` + `compare_aggregates()`; one `EarliestAggregation` is produced per concrete plan, and plan evidence must remain isolated
  - Ordered scalar / greatest comparison: POLICY REQUIRED; credit evidence is at concrete plan/year/semester grain, with no frozen multi-operand greatest/tie/reduction or semester roll-up policy, and remains fail-closed
  - Set difference and generic multi-partition comparison remain unsupported/insufficient and fail-closed
  - focused audit tests: 5 passed
  - no source files modified
  - `tests/test_rag_query_spec.py` does not exist; parser shape was verified by direct reproduction
- [x] Phase 5C.2a — Freeze Earliest Placement Comparison Policy — PASS/FROZEN
  - supported shape: one resolved program, one logical course code, `placement` + `earliest` + `compare`, grouped by `plan`
  - comparison group identity: `(program, course_code)`
  - operand identity: `(program, course_code, concrete plan)`
  - each concrete plan produces its own earliest operand; `coop` and `no_coop` remain separate and are never merged
  - unsupported generic plan difference, credit greatest/most, set difference, and generic multi-partition comparison remain fail-closed
- [x] Phase 5C.3 — Earliest Placement Comparison Implementation — PASS/FROZEN
  - per-plan placement evidence is validated separately
  - one `EarliestAggregation` is produced per concrete plan
  - exactly two valid operands are compared through existing `compare_aggregates()`
  - unsupported/invalid comparison requests fail closed
  - `EarliestAggregation` provenance survives `ComparisonAggregation` claim composition
  - focused implementation tests: 8 passed
  - focused QA module: 25 passed, with 1 unrelated environment error from unavailable `sqlite_vec`
  - `git diff --check` passed
  - no remaining blocker for the supported earliest-placement comparison family
- [x] Phase 5D — Coarse Credit Aggregation Audit — COMPLETE
  - `ปี 2 ของ IT รวมกี่หน่วยกิต` produces operations `("sum_credits",)`, program `IT`, years `(2,)`, no semester filter, and no `group_by`
  - `EvidencePlan` emits `course_set` plus dependent `credit_facts`
  - the executor materializes `credit_facts` at `(program, concrete plan, year, semester)` while QA aggregates each semester result independently
  - requested output grain is `(program, concrete plan, year)`
  - defect is executor fine-grain materialization plus missing QA regroup; it is not an aggregation-contract gap
  - `aggregate_sum_credits()` remains authoritative for credit arithmetic
  - `counted_credit_units`, alternative groups, and nested provenance must survive
  - a valid-empty semester contributes no components; any insufficient semester makes that plan-year roll-up insufficient
  - plans must never be merged, and no new cross-semester deduplication may be introduced
  - no planner, executor, or aggregation change is required
- [x] Phase 5D.1 — Coarse Credit Roll-up Implementation — PASS/FROZEN
  - QA regroups `credit_facts` by `(program, concrete plan, year)`
  - finer semester evidence is combined without introducing new cross-semester deduplication
  - arithmetic remains delegated to `aggregate_sum_credits()`
  - one year-level claim is produced per concrete plan
  - explicit semester queries retain existing semester-level behavior
  - focused QA credit tests: 6 passed
  - `git diff --check`: passed
  - counted-credit, alternative-group, and provenance semantics remain preserved
  - no remaining blocker for coarse year-level credit aggregation
- [x] Phase 5E — Deterministic Greatest Credit Comparison Audit — COMPLETE
  - frozen greatest-credit policy supports factual greatest-credit questions such as `IT เทอมไหนหน่วยกิตเยอะสุด`
  - this policy is specific to greatest-credit selection and does not redefine generic `group_by=("semester",)` behavior
  - candidate identity is `(program, concrete plan, year, semester)`; each concrete curriculum term is one candidate
  - plans remain isolated; `coop` and `no_coop` produce independent greatest-credit results
  - candidate credit values come from existing `aggregate_sum_credits()` / `ComponentAggregation`; QA must not calculate credit totals manually
  - `valid_empty` is a known zero-credit candidate
  - any expected `insufficient_evidence`, or missing expected evidence without explicit `valid_empty`, makes that plan's greatest result fail closed; partial winners are not selected
  - tie policy returns all candidate terms tied for the maximum, ordered deterministically by `(year, semester)`
  - `compare_aggregates()` is not used as an N-way reducer
  - no generic `best`/argmax abstraction is introduced; only narrow greatest-credit orchestration is supported
  - Phase 5D.1 year-level roll-up is not reused; each concrete year-semester candidate is independently aggregated using existing `aggregate_sum_credits()` semantics
  - no planner, executor, or aggregation contract change is required
- [x] Phase 5E.1 — Greatest Credit Selection Implementation — PASS/FROZEN
  - greatest-credit candidates are concrete `(program, concrete plan, year, semester)`
  - plans are reduced independently
  - candidate values come from `aggregate_sum_credits()`
  - ties return all maximum terms in deterministic term order
  - missing, insufficient, invalid, or incomplete candidates fail closed
  - no partial winner is emitted
  - focused tests: 15 passed
  - real target query was verified with separate `coop` and `no_coop` winners
  - `git diff --check`: passed
  - no remaining Phase 5E.1 blocker
- [x] Phase 5F — Alternative-course Group Correctness Audit — COMPLETE
  - executor preserves alternative-group parent rows with `course_code=None` and nested `alternative_courses`
  - credit aggregation already handles alternative groups correctly through `counted_credit_units` and `alternative_group_id`
  - `aggregate_course_set()` currently requires non-empty `course_code`, so an alternative parent raises `ValueError`
  - QA catches the error and fails closed
  - affected proven families: list / count / existence course-set queries
  - credit totals and raw placement/earliest paths are not affected by this defect
  - no executor, DB, schema, or credit-aggregation change is required
  - frozen implementation policy: concrete course identity remains unchanged; alternative parent identity is scoped `(program, alternative_group_id)`
  - preserve each alternative parent as one logical course-set item with nested `alternative_courses` and parent/member provenance
  - never flatten alternatives into independently required courses or silently drop the parent
  - do not introduce new credit or minimum-choice arithmetic; keep existing credit aggregation unchanged
- [x] Phase 5F.1 — Alternative Course-set Aggregation Implementation — PASS/FROZEN
  - `aggregate_course_set()` accepts valid alternative-group parents with `course_code=None`
  - alternative groups deduplicate by `(program, alternative_group_id)` within exact partitions
  - parent, nested members, and provenance are preserved
  - plans remain isolated
  - credit aggregation remains unchanged
  - aggregation tests: 36 passed
  - focused QA tests: 3 passed
  - `git diff --check`: passed
  - no remaining Phase 5F.1 blocker
- [x] Phase 5G — Thai Topic Alias Implementation — PASS/FROZEN
  - deterministic Thai alias "ฐานข้อมูล" maps to canonical topic "database"
  - generic "ข้อมูล" remains unmatched
  - existing topic behavior remains unchanged
  - focused parser tests: 20 passed
  - `git diff --check`: passed
  - no remaining Phase 5G blocker
- [x] Phase 5H — Multi-partition Scope Rendering Audit — COMPLETE
  - `GroundedClaim.effective_scope` already preserves program/plan/year/semester
  - QA claim construction preserves that scope
  - `_deterministic_claim_text()` currently ignores it
  - `render_grounded_answer()` only joins rendered claim segments
  - therefore multi-partition claims can render as ambiguous bare values
  - greatest-credit tied winners are also ambiguous without term labels
  - earliest comparisons retain operand scopes inside `EarliestPartition.partition`
  - defect classification: renderer-only
  - no QA, type, arithmetic, or re-query change is required
  - frozen rendering policy: retain single-partition output; for multiple sibling claims, prefix each affected claim with only differing plan/year/semester dimensions in that order
  - deterministic prefix format: `plan=<value>, year=<value>, semester=<value> | <existing claim text>`
  - greatest-credit ties remain separate and retain deterministic claim order
  - earliest comparisons expose existing left/right operand scope as `left[plan=<...>, year=<...>, semester=<...>]` and `right[plan=<...>, year=<...>, semester=<...>]` when no usable outer scope exists
  - rendering never infers scope, recomputes values, re-queries, changes provenance, reorders claims, or changes aggregation semantics
- [x] Phase 5H.1 — Multi-partition Scope Rendering Implementation — PASS/FROZEN
  - renderer adds contextual scope labels only for sibling dimensions that differ
  - supported dimensions are plan, year, semester in fixed order
  - single-claim output remains unchanged
  - greatest-credit tied winners remain separate and identifiable
  - earliest comparisons expose existing left/right operand scopes
  - no QA, aggregation, re-query, or provenance behavior changed
  - focused rendering tests: 12 passed
  - `git diff --check`: passed
  - no remaining Phase 5H.1 blocker
- [x] Phase 5I — Whole-system Factual QA Audit — COMPLETE / NOT READY
  - course list: PASS
  - course count: PASS
  - existence: PASS
  - prerequisite: PASS
  - earliest plan comparison: PASS
  - year-level credits: PASS
  - greatest-credit term: PASS
  - explicit-program topic query: PASS
  - alternative-course group: PASS
  - unsupported generic plan comparison: PASS, fail-closed
  - provenance, plan isolation, and rendering: PASS
  - no environment blocker encountered
  - single remaining factual blocker: `"มีวิชาเกี่ยวกับฐานข้อมูลอะไรบ้าง"`
  - parser recognizes canonical topic `"database"`; without program context, `clarify_program` is correct
  - with `QueryContext(program="IT")`, program resolution succeeds but `operations` remains empty, so the planner emits no `topic_matches` request
  - equivalent explicit query `"IT มีวิชาเกี่ยวกับฐานข้อมูลอะไรบ้าง"` works through the existing `topic_matches` path
  - readiness: C. NOT READY — FOCUSED BLOCKER
- [x] Phase 5I.1 — Context-resolved Topic Query Operation Fix — PASS/FROZEN
  - topic-only wording now enables the existing `list` operation independently of whether program text is explicit in the utterance
  - `QueryContext(program="IT")` allows `"มีวิชาเกี่ยวกับฐานข้อมูลอะไรบ้าง"` to reach the existing `topic_matches` evidence path
  - no-context form still clarifies program
  - explicit-program behavior remains unchanged
  - canonical `"ฐานข้อมูล"` -> `"database"` behavior remains unchanged
  - no new retrieval, fuzzy, or LLM inference behavior was introduced
  - focused tests: 4 passed
  - `git diff --check`: passed
  - no remaining Phase 5I.1 blocker
- [x] Phase 5I.2 — Factual Gold Readiness Re-audit — PASS/FROZEN
  - all 9 required factual re-audit cases passed
  - context-resolved and explicit topic queries both reach the existing `topic_matches` path
  - no-context topic query still clarifies program
  - prerequisite, earliest comparison, year credits, greatest-credit, and alternative-course-group paths all passed
  - generic plan comparison still fails closed correctly
  - provenance preserved
  - plan identities remained isolated
  - deterministic scope rendering remained correct
  - no environment-only blocker
  - `git diff --check`: passed
  - no files modified
  - readiness: A. READY FOR GOLD
- [x] Phase 5J — Initial Gold Evaluation — INVALID / NON-SCORABLE
  - all 30 cases reached evaluator integration errors
  - current `qa.ask()` returns `GroundedAnswerResult` for successful typed results
  - evaluator still expects legacy route/result mappings
  - 25 cases failed by subscripting `GroundedAnswerResult`
  - 5 blocked cases failed because legacy `structured`/`semantic` keys were absent
  - therefore 0/30 execution success is not a system-quality measurement
  - Gold dataset itself was not modified
  - production RAG was not modified
  - no environment blocker occurred
- [x] Phase 5J.1 — Gold Harness Typed-result Contract Audit — COMPLETE
  - `route_question()` already provides the compatibility diagnostic route; route must not be derived from typed claims
  - `GroundedAnswerResult.final_answer` is the authoritative rendered answer
  - typed claims, evidence, and provenance must feed existing scoring
  - blocked statuses must normalize without legacy result subscripting
  - expected-answer scoring rules can remain unchanged
  - production RAG does not require changes
- [x] Phase 5J.2 — Typed Gold Evaluator Compatibility Implementation — PASS/FROZEN
  - typed `GroundedAnswerResult` is normalized by the evaluator
  - `final_answer` is consumed directly
  - `route_question()` diagnostic is preserved
  - typed structured, semantic, and hybrid evidence is projected from existing claims
  - blocked `no_data`, `clarify_program`, and `insufficient_evidence` results normalize without fabricated evidence
  - legacy evaluator compatibility remains supported
  - production RAG and Gold dataset were not modified
  - focused evaluator tests: 31 passed
  - `git diff --check`: passed
  - no remaining evaluator compatibility blocker
  - the initial 30-case Gold run remains INVALID / NON-SCORABLE because every case failed at the old evaluator integration boundary
- [x] Phase 5J.3 — Canonical Gold Evaluation Rerun — COMPLETE / GOLD FAIL
  - execution success: 30/30
  - strict correct: 13/30
  - raw strict score: 43.3%
  - factual in-scope strict: 13/29 = 44.8%
  - out-of-scope: 1
  - correct out-of-scope abstention: 1/1
  - errors: 0
  - failure clusters:
    - semantic evidence unavailable from typed runtime: 5
    - requested multi-operation typed claims incomplete: 6
    - year-5 scope not captured by parser: 1
    - cross-program semantic comparison unavailable: 1
    - evaluator mismatches: 3
    - generic plan comparison: 1 intentional out-of-scope
  - final classification: GOLD FAIL — FOCUSED FACTUAL DEFECTS
  - the three evaluator mismatches are not production-RAG defects
- [x] Phase 5K — Typed Semantic Evidence Runtime Audit — COMPLETE
  - representative semantic course queries resolve course and program correctly
  - parsed operations are empty
  - the planner therefore emits no `description_evidence` request
  - executor and retrieval never run
  - no `GroundedClaim`, evidence, or provenance can be produced
  - first-loss boundary: `rag/query_spec.py::_extract_operations()`
  - classification: parser-to-planner `describe`-operation gap
  - `sqlite_vec` is not involved
  - planner, executor, retrieval, and QA do not require changes
  - `git diff --check`: passed
  - no files were modified
- [x] Phase 5K.1 — Semantic Describe Wording Policy Freeze — COMPLETE
  - five ordinary single-program semantic Gold failures share a parser operation gap
  - exact course and program resolution already works
  - operations remains empty because existing `describe` patterns are too narrow
  - semantic request and retrieval therefore never execute
  - supported new wording families are narrowly:
    - ลักษณะไหน
    - ด้านไหน / ด้านใด
    - พูดถึง
    - course-targeted ...อะไรบ้าง
    - อย่างไร
    - ...แบบไหน
  - new `describe` recognition must require an exact course target
  - existing list, count, placement, prerequisite, and topic behavior must remain unchanged
  - no planner, executor, retrieval, or QA change is required
  - `git diff --check`: passed
  - no files were modified
- [x] Phase 5K.2 — Course-targeted Describe Operation Implementation — PASS/FROZEN
  - deterministic `describe` cues added: ลักษณะไหน, ด้านไหน/ด้านใด, พูดถึง, อะไรบ้าง, อย่างไร, แบบไหน
  - the new rule applies only when an exact course code or name target exists
  - existing list, count, placement, prerequisite, and topic paths remain guarded
  - focused parser tests: 3 passed
  - focused typed QA semantic regression: 1 passed
  - `git diff --check`: passed
  - no remaining Phase 5K.2 blocker
- [x] Phase 5K.3 — Focused Semantic Runtime Re-evaluation — PASS/FROZEN
  - all 5 previously failing ordinary semantic cases now parse `describe`
  - all 5 emit `description_evidence`
  - all 5 execute with non-empty persisted description evidence
  - all 5 produce complete `describe` `GroundedClaim`s
  - provenance survives through `GroundedAnswerResult`
  - `sqlite_vec` and vector search are not involved in this path
  - the original 5-case semantic runtime cluster is CLOSED
  - `git diff --check`: passed
  - no files were modified
- [x] Phase 5L — Multi-operation Typed Claim Completeness Audit — COMPLETE
  - six audited cases split into multiple causes
  - three factual failures are caused by placement wording: `อยู่ปีไหน` / `อยู่เทอมไหน` not producing the existing placement operation
  - one equivalent `เรียนปีไหน` / `เทอมไหน` case already works completely
  - fixed/flexible placement plus description case already works completely
  - no executor, aggregation, or claim-composition defect was found
  - one generic plan-comparison case remains unsupported by policy and correctly emits `insufficient_evidence` for comparison
  - evidence reuse for supported multi-operation requests is already sufficient
  - generic comparison must not be implemented as part of this fix
  - `git diff --check`: passed
  - no files were modified
- [x] Phase 5L.1 — Placement Wording Parser Implementation — PASS/FROZEN
  - deterministic placement wording added: `อยู่ปีไหน`, `อยู่เทอมไหน`
  - combined wording emits placement once
  - three target multi-operation queries now include `placement`, `sum_credits`, `describe`
  - existing planner, executor, and QA contracts handle the new placement operation
  - focused tests: 5 passed
  - generic comparison remains intentionally unsupported and fail-closed
  - `git diff --check`: passed
  - no remaining blocker for the supported placement wording family
- [x] Phase 5M — Year-5 Scope Parsing Audit — COMPLETE
  - explicit `ปี 5` is currently dropped by query parsing
  - `_YEAR_PATTERN` accepts only years 1-4
  - `StructuralScope` and `EvidencePlan` can represent year 5
  - structured `scoped_course_set()` independently rejects years outside 1-4
  - current BIT/no-coop data has no year-5 placements
  - if explicit year 5 is preserved and no rows exist, the correct structured result is `no_data` -> executor `valid_empty`
  - silently dropping year 5 is unsafe because it broadens the query scope
  - classification: parser + structured-query validation gap
  - `git diff --check`: passed
  - no files were modified
  - frozen policy: explicitly requested study years remain in scope; years 1-5 are valid representable scopes, absence of year-5 rows is valid empty evidence, values above 5 remain unsupported, and no year-5 data may be fabricated
- [x] Phase 5M.1 — Year-5 Explicit Scope Preservation Implementation — PASS/FROZEN
  - deterministic year parsing now supports years 1-5
  - explicit year 5 is preserved in `QuerySpec`
  - year 6 remains unsupported
  - structured placement and credit validation accepts year 5
  - no matching year-5 rows produce `no_data` -> executor `valid_empty`
  - year 5 never broadens silently into years 1-4
  - semester validation remains unchanged
  - focused tests: 6 passed
  - `git diff --check`: passed
  - no remaining Phase 5M.1 blocker
- [x] Phase 5N — Cross-program Multi-target Query Audit — COMPLETE / OUT OF SCOPE
  - frozen product scope: one Natural QA request is scoped to one curriculum/program
  - multiple course targets are allowed only when they belong to the same program scope
  - cross-program questions combining IT/DSBA/BIT/AIT in one request are not supported
  - do not add multi-program `StructuralScope` or orchestration
  - do not split one user query into independent program pipelines
  - do not synthesize semantic comparisons across programs
  - such requests must fail closed rather than fabricate an answer
  - audited Gold case `semantic_cross_program_database_topics_06016402_06026207` is OUT OF SCOPE under this frozen requirement
  - same-program comparisons across curriculum years/versions may be considered in the future when such source data exists; no architecture is added for that now
  - canonical Gold dataset remains unchanged; retain the item for raw-score traceability and classify it as out-of-scope during evaluation reporting
- [x] Phase 5O.0 — Final Gold Scope Review — PASS/FROZEN
  - total canonical Gold: 30
  - IN-SCOPE factual: 28
  - OUT-OF-SCOPE: 2
  - out-of-scope:
    - `structured_bit_same_code_06036103`
      - generic/unmetricized comparison
    - `semantic_cross_program_database_topics_06016402_06026207`
      - cross-program multi-target request
  - frozen evaluation policy:
    - retain all 30 questions for raw benchmark traceability
    - final product-scope denominator is 28
    - do not modify runtime to force the two out-of-scope cases to pass
    - do not weaken fail-closed behavior
    - classification was based on question semantics before the next evaluation, not on pass/fail outcome
- [ ] Phase 5 review findings before Gold
  - [x] similarity partition alignment — Phase 5B root cause confirmed and partition projection fixed; year/semester placement no longer blocks same-plan pairing
  - [x] comparison execution gap — missing `ComparisonAggregation` producer confirmed; Phase 5C.1 now fails closed explicitly
  - [x] coarse credit aggregation — Phase 5D confirmed the fine-grain executor materialization and missing QA regroup; Phase 5D.1 completed the regroup and is no longer an undiagnosed blocker

- [x] Phase 5O — Post-fix Canonical Gold Evaluation — COMPLETE
  - execution: 30/30
  - raw strict: 18/30 = 60.0%
  - final in-scope strict: 18/28 = 64.3%
  - errors: 0
  - evidence correctness: 25/30
  - provenance correctness: 29/30
  - route match: 24/25
  - semantic coverage: 64/75
  - final scope: in-scope 28; out-of-scope 2
  - remaining in-scope findings: real factual defects 4; evaluator/dataset mismatches 5; review/unclear 1
  - no new regression was observed in previously frozen fixes

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

- [x] Phase 5P — Residual Hybrid Evidence Completeness Audit — COMPLETE
  - 4 residual cases were audited end-to-end
  - true runtime defects: 2
  - evaluator/projection mismatches: 2
  - true runtime defects:
    - `hybrid_it_coop_course_06016418`
      - explicit placement + description request
      - `QuerySpec.operations=()`
      - parser misses both `เรียนช่วงไหนของหลักสูตร` and the course-detail wording `เนื้อหาครอบคลุม...เรื่องใด...บ้าง`
    - `hybrid_bit_coop_course_06036115`
      - placement already parses correctly
      - parser misses course-detail wording `เนื้อหาช่วยจัดการ...เรื่องใดบ้าง`
  - reclassified evaluator mismatches:
    - `hybrid_it_no_coop_course_06016419`
    - `hybrid_it_no_coop_course_06016404`
      - typed runtime produces complete placement / credit / describe claims
      - one logical persisted description record survives executor and claim
      - provenance is preserved
      - do not change production RAG for evaluator semantic-row expectations
- [x] Phase 5P.1 — Residual Course-targeted Parser Wording Fix — PASS/FROZEN
  - `hybrid_it_coop_course_06016418` now parses `placement` + `describe`
  - `hybrid_bit_coop_course_06036115` now parses `placement` + `describe`
  - deterministic placement wording added: `เรียนช่วงไหนของหลักสูตร`
  - deterministic course-targeted describe wording added for `เนื้อหา...ครอบคลุม...เรื่องใด` and `เนื้อหา...ช่วยจัดการ...เรื่องใด`
  - both target queries produce complete typed placement/describe claims with provenance through existing planner/executor/QA contracts
  - focused tests: 10 passed
  - `git diff --check`: passed
  - no remaining known production-runtime factual defect from Phase 5P
  - evaluator-side cases `hybrid_it_no_coop_course_06016419` and `hybrid_it_no_coop_course_06016404` are not production RAG defects; their typed runtime produces complete placement, credit, and describe claims with provenance
- [x] Phase 5Q — Evaluator / Dataset Mismatch Audit — COMPLETE
  - remaining production factual defects: 0; do not modify `rag/*` for the audited cases
  - evaluator compatibility defects: 6
    - `hybrid_it_no_coop_course_06016419`: typed runtime complete; evaluator expects legacy semantic-row granularity
    - `hybrid_it_no_coop_course_06016404`: typed runtime complete; evaluator expects legacy semantic-row granularity
    - `hybrid_dsba_coop_course_06026259`: typed placement evidence/claims complete; evaluator text matching cannot deterministically verify placement fields
    - `hybrid_bit_no_coop_course_06036122`: same typed placement verification issue
    - `hybrid_bit_no_coop_course_06036114`: same typed placement verification issue
    - `hybrid_it_fixed_vs_flexible_06016481`: current typed `year_semester_choices` representation complete; evaluator expects legacy flexible-placement metadata
  - dataset expectation mismatch: 1
    - `unknown_bit_no_coop_year5_sem1`: frozen product contract remains explicit supported year 1–5 plus no rows = `valid_empty`; do not restore legacy unknown-course fallback semantics
  - legitimate review: 1
    - `structured_bit_coop_course_06036107`: current runtime is factually complete; keep as REVIEW unless a separate deterministic grading policy is proven; do not special-case it
  - frozen evaluator policy:
    - scorer may inspect typed claims/evidence directly and align with current typed representations
    - scorer must not invent evidence or legacy metadata
    - one persisted logical description record remains one logical semantic unit
    - do not weaken factual correctness requirements
- [x] Phase 5Q.1 — Typed Evaluator Compatibility Fix — PASS/FROZEN
  - typed semantic scoring treats one complete `describe` claim as one logical semantic evidence unit
  - typed placement scoring reads `year_number` and `semester_number` directly
  - typed flexible placement scoring accepts `year_semester_choices`
  - no legacy flexible metadata is fabricated
  - legacy evaluator compatibility remains passing
  - focused evaluator tests: 37 passed
  - `git diff --check`: passed
  - production RAG was not modified
  - frozen exclusions: `unknown_bit_no_coop_year5_sem1`, `structured_bit_coop_course_06036107`, and both product-scope OUT-OF-SCOPE cases
  - current production-runtime factual defect count: 0
- [x] Phase 5Q.2 — Year-5 `valid_empty` Evaluator Alignment — PASS/FROZEN
  - typed `valid_empty` is accepted only for a valid explicit structural scope
  - supported year range remains 1–5
  - concrete plan must be `coop`/`no_coop`
  - semester must be 1–2
  - all relevant claims must be `valid_empty`
  - unknown/not-found behavior remains distinct
  - malformed, unsupported, missing, and insufficient scopes do not pass
  - legacy evaluator compatibility remains intact
  - `structured_bit_coop_course_06036107` REVIEW behavior remains unchanged
  - focused evaluator tests: 40 passed
  - `git diff --check`: passed
  - production RAG was not modified
  - current known production factual defect count: 0
  - final frozen Gold scope: canonical total 30; in-scope factual 28; out-of-scope 2
  - frozen out-of-scope IDs: `structured_bit_same_code_06036103` and `semantic_cross_program_database_topics_06016402_06026207`
  - known intentional REVIEW: `structured_bit_coop_course_06036107`
- [x] Phase 5R — Final Canonical Gold Evaluation — PASS/FROZEN
  - execution: 30/30
  - errors: 0
  - raw PASS: 19; raw REVIEW: 10; raw FAIL: 1
  - raw strict: 19/30 = 63.3%
  - final in-scope strict: 19/28 = 67.9%
  - evidence correctness: 30/30
  - provenance correctness: 30/30
  - semantic coverage: 75/75
  - route match: 24/25
  - runtime factual defects: 0
  - new regressions: 0
  - final product scope: canonical total 30; in-scope factual 28; out-of-scope 2
  - frozen OUT-OF-SCOPE: `structured_bit_same_code_06036103` and `semantic_cross_program_database_topics_06016402_06026207`
  - remaining non-strict IN-SCOPE results: evaluator/final-answer verification mismatch 8; legitimate deterministic REVIEW 1
  - all canonical cases executed; all required evidence, provenance, and semantic evidence checks passed
  - no currently known production RAG factual defect remains
  - remaining non-strict outcomes must not justify further production-RAG changes
  - do not tune the evaluator further against this canonical Gold merely to increase strict score
- [x] Phase 5S — Regression Set Freeze — COMPLETE
  - freeze the current canonical Gold as a DEVELOPMENT / REGRESSION benchmark
  - no more code changes should be driven solely by failures/reviews in this set
  - future correctness claims must use an unseen factual evaluation set
  - canonical Gold remains useful for regression detection only
  - rerun canonical Gold only if future implementation regresses previously passing factual behavior
  - do not use canonical expected answers to tune future unseen behavior

## Current Task

### Phase 5T — Unseen Factual Evaluation Design

Status: AUDIT / DESIGN ONLY

Goal:
Design a new unseen factual evaluation set aligned with the frozen final
product scope and instructor requirement.

The unseen set must:

- contain only objectively answerable curriculum questions
- avoid AI-style subjective reasoning
- remain within one program per query
- include varied natural wording
- cover factual operations and safe multi-operation combinations
- contain no cross-program synthesis
- contain no subjective workload/preference/difficulty/career questions
- not be created by simply changing course codes in canonical Gold questions
- be evaluated without tuning runtime against expected answers

No production implementation.
No evaluator implementation.
Do not create the unseen dataset yet in this task.

Run:
git diff --check

Stop immediately.
