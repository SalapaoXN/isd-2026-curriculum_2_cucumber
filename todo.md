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
| nq_012 | describe | none | answer |
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

- [ ] baseline audit
- [ ] review priorities before coding

---

## Phase 3 — Architecture Design

Design the smallest implementation needed from baseline evidence.

Do not assume router rewrite is required.

Review architecture before coding.

- [ ] architecture plan approved

---

## Phase 4 — Implementation Micro-tasks

Do separately:

- [ ] 4A Thai normalization
- [ ] 4B entity/operation representation
- [ ] 4C missing-program ambiguity guard
- [ ] 4D plan-aware behavior
- [ ] 4E constrained semantic retrieval
- [ ] 4F semantic aggregation/comparison
- [ ] 4G exact-course composition
- [ ] 4H grounded judgement policy
- [ ] 4I no-data + unsupported guards

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

**Phase 2 — Baseline Audit**

Audit the current `heart` implementation against all 40 Natural QA v1 cases.

No production changes.