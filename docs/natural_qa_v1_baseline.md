# Natural QA v1 — Phase 2 Baseline Report

## Scope and methodology

This report records the baseline audit of the current `heart` implementation
against the 40 standalone Natural QA v1 cases. It uses the reviewed Phase 2A
and Phase 2B runtime evidence and the corrected fixture contract.

- No production code was changed.
- No Gemini or external LLM was called.
- Runtime checks used the existing local database/index only.
- Answer synthesis with a model was not invoked.
- `structured`, `semantic`, and `hybrid` are diagnostic route labels only;
  route choice is not itself a correctness criterion.

## Case baseline

| ID | Canonical question | Expected operation / action | Actual route | Earliest failure stage | Short reason | Confidence |
|---|---|---|---|---|---|---|
| nq_001 | IT ปี 2 เทอม 1 เรียนอะไรบ้าง | `list` / `answer` | structured | structured query/filter | Natural list wording has no deterministic matcher; model fallback is required. | high |
| nq_002 | ปีสองเทอมสองของ IT มีวิชาอะไร | `list` / `answer` | structured | normalization | `ปีสอง` is not normalized to year 2. | high |
| nq_003 | IT ปี 3 ต้องเรียนกี่วิชา | `count` / `answer` | structured | structured query/filter | Year-level count has no deterministic matcher. | high |
| nq_004 | ปี 2 ของ IT รวมกี่หน่วยกิต | `sum_credits` / `answer` | structured | structured query/filter | Year-only credit aggregation is not handled deterministically. | high |
| nq_005 | IT เทอมไหนหน่วยกิตเยอะสุด | `sum_credits`, `compare` / `answer` | structured | structured query/filter | Cross-semester credit comparison has no deterministic matcher. | high |
| nq_006 | 06016414 เรียนปีไหน | `placement` / `answer` | structured | structured query/filter | Code-only placement does not activate the IT placement matcher. | high |
| nq_007 | 06016414 เปิดให้ลงช่วงไหนได้บ้าง | `placement` / `answer` | semantic | supported | Exact-code filtering returned both IT placement rows with provenance. | medium |
| nq_008 | 06016414 ต้องเรียนวิชาอะไรมาก่อน | `prerequisite` / `answer` | semantic | semantic retrieval | Retrieval returned placement/description chunks, not prerequisite edges. | high |
| nq_009 | IT สหกิจกับไม่สหกิจต่างกันยังไง | `compare` / `answer` | semantic | semantic retrieval | Coop/no_coop constraints are not enforced before ranking. | high |
| nq_010 | 06016465 แผนไหนได้เรียนเร็วกว่า | `placement`, `earliest`, `compare` / `answer` | semantic | composition | Both plan placements were retrieved, but earliest-plan composition is absent. | high |
| nq_011 | 06016414 เรียนเกี่ยวกับอะไร | `describe` / `answer` | semantic | supported | Exact-code filtering returned the target description chunks with provenance. | medium |
| nq_012 | วิชา NOSQL เรียนเรื่องอะไรบ้าง | `describe` / `clarify_program` | semantic | ambiguity | Exact name resolves to IT `06016414` and DSBA `06026207`; program clarification is required. | high |
| nq_013 | IT มีวิชาเกี่ยวกับ AI อะไรบ้าง | `list` / `answer` | semantic | semantic retrieval | Program and plan constraints are not enforced before ranking. | high |
| nq_014 | มีวิชาเกี่ยวกับ database อะไรบ้างใน IT | `list` / `answer` | semantic | semantic retrieval | IT filtering is not enforced before ranking. | high |
| nq_015 | ถ้าชอบเขียนโปรแกรม IT มีวิชาอะไรน่าสนใจ | `list` / `answer` | semantic | semantic retrieval | IT and plan constraints are not enforced; preference composition is also unguarded. | medium |
| nq_016 | มีวิชาเกี่ยวกับเว็บไหม | none / `clarify_program` | semantic | ambiguity | Missing program does not trigger clarification. | high |
| nq_017 | IT ปี 2 มีวิชาเกี่ยวกับ database อะไรบ้าง | `list` / `answer` | semantic | semantic retrieval | IT, year, and plan constraints are not enforced. | high |
| nq_018 | ปี 3 ของ IT มีวิชาเกี่ยวกับ AI กี่วิชา | `count` / `answer` | hybrid | structured query/filter | No deterministic topic-count matcher; structured execution requires a model. | high |
| nq_019 | IT ปี 2 เทอม 1 มีวิชาเกี่ยวกับ programming ไหม | `existence` / `answer` | hybrid | structured query/filter | No deterministic topic-existence composition for the requested term. | high |
| nq_020 | ปี 2 มีวิชาเกี่ยวกับคอมพิวเตอร์เยอะมั้ย | none / `clarify_program` | semantic | ambiguity | Missing program does not trigger clarification. | high |
| nq_021 | IT ปี 2 มีวิชาเกี่ยวกับคอมพิวเตอร์เยอะมั้ย | `count` / `answer` | semantic | semantic retrieval | Topic count is not restricted to IT year 2 and applicable plans. | high |
| nq_022 | ปีสองของ IT เรียนคอมหนักมั้ย | `count`, `sum_credits` / `answer` | semantic | normalization | `ปีสอง` is not normalized to year 2. | high |
| nq_023 | IT ปี 2 กับปี 3 ปีไหนมีวิชา programming เยอะกว่า | `count`, `compare` / `answer` | semantic | semantic retrieval | Evidence is not partitioned by IT year 2 versus year 3. | high |
| nq_024 | เทอมไหนของ IT มีวิชาเกี่ยวกับ database เยอะสุด | `count`, `compare` / `answer` | hybrid | semantic retrieval | Semester groups are not constrained before ranking or aggregation. | high |
| nq_025 | แผนสหกิจมีวิชาเกี่ยวกับ data มากกว่าแผนปกติไหม | none / `clarify_program` | semantic | ambiguity | Program is missing and should be clarified before plan comparison. | high |
| nq_026 | วิชาเลือกของ IT ที่เกี่ยวกับ AI มีอะไรบ้าง | `list` / `answer` | semantic | semantic retrieval | IT and elective-category filters are not enforced before ranking. | high |
| nq_027 | ปี 3 มีวิชาเลือกเกี่ยวกับเว็บกี่ตัว | none / `clarify_program` | hybrid | ambiguity | Program is missing and should be clarified before category counting. | high |
| nq_028 | ถ้าอยากเรียน AI เริ่มมีวิชาแนวนี้ตั้งแต่ปีไหน | none / `clarify_program` | semantic | ambiguity | Program is missing before earliest-year reasoning. | high |
| nq_029 | ถ้าอยากเรียน database ต้องผ่านวิชาอะไรมาก่อนบ้าง | none / `clarify_program` | semantic | ambiguity | Program is missing before prerequisite lookup. | high |
| nq_030 | 06016414 เรียนเรื่องอะไร แล้วเรียนปีไหน | `describe`, `placement` / `answer` | structured | composition | Existing description and placement facts are not composed for this wording. | high |
| nq_031 | 06016414 กับ 06016419 เนื้อหาคล้ายกันไหม | `similarity` / `answer` | semantic | aggregation/comparison | Both exact-code description evidences are available; similarity comparison remains to be composed. | high |
| nq_032 | 06016414 กับ 06016419 ตัวไหนเรียนก่อน | `placement`, `compare` / `answer` | semantic | composition | Both placement sets were retrieved, but earlier-course composition is absent. | high |
| nq_033 | ปี 4 ต้องเรียนอะไรบ้าง | none / `clarify_program` | semantic | ambiguity | Missing program does not trigger clarification. | high |
| nq_034 | IT ปี 4 ต้องเรียนอะไรบ้าง | `list` / `answer` | semantic | semantic retrieval | IT, year 4, and plan constraints are not enforced before ranking. | high |
| nq_035 | เทอมสองปีสามมีตัว database เยอะปะ | none / `clarify_program` | structured | normalization | `เทอมสอง` and `ปีสาม` are not normalized before the program check. | high |
| nq_036 | IT ปีสามเทอมปลายเรียนหนักไหม | `count`, `sum_credits` / `answer` | structured | normalization | `ปีสาม` and `เทอมปลาย` are not normalized. | high |
| nq_037 | วิชาไหนยากที่สุดใน IT | none / `unsupported` | semantic | unsupported guard | Difficulty is not a curriculum-grounded claim and lacks a safe rejection guard. | high |
| nq_038 | เรียนวิชาไหนแล้วเงินเดือนสูงสุด | none / `unsupported` | semantic | unsupported guard | Salary outcomes are not curriculum-grounded and lack a safe rejection guard. | high |
| nq_039 | 06019999 เรียนอะไร | `describe` / `no_data` | semantic | supported | Exact-code filtering returned no chunks, supporting canonical no-data behavior. | high |
| nq_040 | ตอนปีสามของ IT มีวิชาเกี่ยวกับ network อะไรบ้าง | `list` / `answer` | semantic | normalization | `ตอนปีสาม` is not normalized before constrained semantic retrieval. | high |

## Aggregate baseline counts

| Earliest failure stage | Count |
|---|---:|
| semantic retrieval | 11 |
| ambiguity | 8 |
| structured query/filter | 7 |
| normalization | 5 |
| composition | 3 |
| unsupported guard | 2 |
| aggregation/comparison | 1 |
| supported | 3 |
| **Total** | **40** |

## Failure clusters

### Constrained semantic retrieval

Semantic retrieval scopes explicit course codes, but does not apply program,
plan, year, semester, or category constraints before vector ranking. This
affects topic lists, counts, comparisons, and workload questions.

### Missing-program and exact-name ambiguity

Questions without a program proceed to retrieval instead of clarifying. The
corrected `nq_012` contract additionally records that the exact name NOSQL is
ambiguous across IT and DSBA and therefore requires `clarify_program`.

### Deterministic structured query/filter gaps

Several natural list, count, credit, and existence questions route to a
structured diagnostic path but have no matching deterministic operation. They
therefore require NL-to-SQL/model fallback.

### Thai normalization

Thai forms including `ปีสอง`, `ปีสาม`, `เทอมสอง`, `เทอมปลาย`, and `ตอนปีสาม`
are not normalized before filtering or intent handling.

### Composition

The system can retrieve supporting placement or course facts for several
multi-fact questions, but does not consistently compose description,
placement, earliest, or cross-course results upstream.

### Semantic comparison

Exact-code evidence for both courses is available for similarity/comparison
questions, but comparison remains dependent on later synthesis rather than a
deterministic evidence composition step.

### Unsupported guard

Difficulty and salary questions are not grounded by the curriculum, but the
current semantic path does not deterministically reject or qualify them.

## Already-supported behavior

- Exact 8-digit course-code extraction.
- Exact-course semantic candidate filtering.
- Explicit-course description evidence retrieval.
- Some explicit-course placement evidence retrieval.
- Provenance attached to local semantic and structured evidence.
- Deterministic exact no-data behavior for an explicit nonexistent code.
- Existing deterministic structured primitives for previously supported wording.

## Architecture implications

The evidence supports the following implications for Phase 3 discussion only;
no implementation design is proposed here:

- A router rewrite is not proven necessary by this baseline.
- Central query understanding/representation is likely needed.
- Semantic retrieval needs structural constraints supplied before ranking.
- Ambiguity handling must occur before retrieval.
- Aggregation and composition should operate on grounded evidence.
- Unsupported claims need deterministic guards.

## Priority order for Phase 3 discussion

The dependency-based order is:

1. Normalization and query-understanding representation.
2. Missing-program and exact-name ambiguity guards.
3. Structural program/plan/year/semester/category constraints.
4. Constrained semantic retrieval.
5. Aggregation, comparison, and composition.
6. Judgement and unsupported-claim handling.

This order follows prerequisite dependencies: later operations cannot reliably
aggregate or judge evidence until entities, ambiguity, and retrieval scope are
correct.
