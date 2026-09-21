# CUCUMBER — Academic Rules Integration TODO

> Branch: `heart`  
> Scope: Integrate institution regulations + program-specific graduation requirements into the existing CUCUMBER pipeline and RAG without regressing the frozen curriculum QA path.

---

## 0. Current State

### Completed

- [x] Source acquisition complete
  - [x] `data/input/rule/rule_page_001.png` … `rule_page_013.png`
  - [x] `data/input/rule/ait_page_005.png`
  - [x] `data/input/rule/bit_page_006.png`
  - [x] `data/input/rule/dsba_page_006.png`
  - [x] `data/input/rule/it_page_006.png`
- [x] OCR of 13 institution regulation pages complete
- [x] Existing `RuleExtractor` run successfully
- [x] `rules_extracted.json` produced
- [x] Structural audit passed
  - [x] 155 structured rules
  - [x] top-level rules 1–53 present
  - [x] no duplicate rule IDs
  - [x] valid `parent_rule_id`
  - [x] valid `section_path`
  - [x] valid references
  - [x] source pages 1–13 covered
  - [x] original `rule_page_###.png` provenance preserved
  - [x] cross-page rules preserved
  - [x] no signature/footer contamination at final rule
- [x] Existing `RulesPolicyMapper` audited and extended
  - [x] 16 semantic categories implemented (R1 COMPLETE / FROZEN)
  - [x] fail-closed behavior exists
  - [x] provenance/supporting text exists
  - [x] focused mapper tests already exist

### Verified program-specific total credits

- [x] AIT = 120 credits
- [x] BIT = 126 credits
- [x] DSBA = 132 credits
- [x] IT = 129 credits

These values come from the individual curriculum documents, not the institution regulation.

---

# 1. Non-Negotiable Contracts

Keep these rules for every phase.

- [ ] `RULE` must **not** be added to `SUPPORTED_PROGRAMS`
- [ ] Institution regulations must remain a separate authority from curriculum/program facts
- [ ] `rules_ground_truth.json` must **not** become production factual authority
- [ ] Do not modify production behavior to match GT when GT conflicts with the official source
- [ ] Do not hardcode AIT/BIT/DSBA/IT total credits as production truth in code
- [ ] Do not globally repair OCR values such as:
  - [ ] `3.51 -> 3.50`
  - [ ] `8 -> B`
- [ ] Source-verified corrections must be scoped to the exact rule/source context
- [ ] Do not rewrite `rule:33.11` reference `21 -> 22`
  - preserve the document's original reference
  - flag it as a source/internal-reference anomaly if needed
- [ ] LLM must not invent legal/regulation thresholds
- [ ] Deterministic evidence remains factual authority
- [ ] Fail closed when evidence is incomplete or ambiguous
- [ ] Preserve provenance from source to answer
- [ ] Do not rewrite `RuleExtractor` unless a real structural defect is demonstrated
- [ ] Do not modify the frozen curriculum EvidencePlan path until standalone policy QA is proven
- [ ] Preserve unrelated dirty/untracked work
- [ ] Coding agents must not `git add`, `commit`, `push`, `reset`, `restore`, `checkout`, `switch`, or `revert` unless explicitly instructed

---

# 2. Known Source/OCR Issues

These are expected cases and must be handled deliberately.

## `rule:11` — Registration

- [x] OCR retains the important values
  - regular semester minimum = 9 credits
  - regular semester maximum = 22 credits
  - graduation exception maximum = 27 credits
  - special semester maximum = 9 credits

## `rule:22` — Probation

- [x] `2.OO` can be safely normalized to `2.00` using existing bounded numeric normalization
- [ ] Preserve noisy/incomplete raw text as evidence
- [ ] Do not silently rewrite surrounding text

Expected semantic facts:

```text
GPA < 2.00  -> probation
GPA >= 2.00 -> probation cleared
```

## `rule:25.1` — Graduation

OCR loses the numeric value after `ไม่ต่ำกว่า`.

- [x] Verify against official source image
- [x] Represent GPA threshold as source-verified fact
- [x] Do not infer from OCR alone

Expected:

```text
curriculum-structure GPA >= 2.00
cumulative GPA >= 2.00
```

## `rule:27.2.2` — Honors

OCR damage:

```text
3.51
8
```

Official source indicates:

```text
3.50
B
```

- [x] Implement source-scoped correction only for this rule
- [x] Mark correction as `source_verified`
- [x] Ensure `3.51` never becomes a production factual threshold

## `rule:33.8`

OCR drops the number of exam-cheating occurrences.

- [ ] Verify source
- [ ] Add source-verified semantic fact only if exact value is clear

## `rule:33.11`

Source itself references `ข้อ 21`.

- [ ] Preserve exactly
- [ ] Do not "fix" to 22
- [ ] Optionally mark `source_internal_reference_anomaly`

---

# 3. R1 — Complete Semantic Rule Mapping

**Smallest sufficient model/mode:** `Luna + Build`

## Goal

Extend the existing `RulesPolicyMapper` from 12 categories to the 16-category evaluation taxonomy without touching DB/RAG yet.

## Allowed files

Primary:

```text
src/pipeline/tools/merge/policy.py
tests/tools/test_rules_policy_mapper.py
```

Optional new semantic evaluation fixture:

```text
ground_truth/rules_semantic_eval.json
```

Do **not** modify:

```text
ground_truth/rules_ground_truth.json
```

## Add categories

- [x] `GRADUATION_CATEGORY = "เกณฑ์การสำเร็จการศึกษา"`
- [x] `REGISTRATION_CATEGORY = "เกณฑ์การลงทะเบียน"`
- [x] `TRANSFER_CATEGORY = "การเทียบโอนหน่วยกิต"`
- [x] `OTHER_CATEGORY = "ระเบียบอื่น ๆ"`

## Suggested rule groups

Verify against actual extracted/source data before finalizing exact lists.

```text
graduation
  -> 25, 25.x ...

registration
  -> 10 ... 16

transfer
  -> 28, 29

other
  -> explicit unknown/null unless a deterministic definition exists
```

## Required structured registration facts

- [x] `regular_semester_min_credits = 9`
- [x] `regular_semester_max_credits = 22`
- [x] `graduation_exception_max_credits = 27`
- [x] `special_semester_max_credits = 9`
- [x] preserve conditions/exceptions separately

Recommended representation:

```json
{
  "value": "22",
  "unit": "credits",
  "label": "หน่วยกิตสูงสุดภาคการศึกษาปกติ",
  "condition": "at_most",
  "context": "regular_semester",
  "source_rule_id": "rule:11"
}
```

## Required structured graduation facts

- [x] curriculum structure completed
- [x] all required courses passed
- [x] curriculum-structure GPA >= 2.00
- [x] cumulative GPA >= 2.00
- [x] English Exit Exam requirement where source supports it
- [x] no outstanding institutional obligation where source supports it

## Required honors facts

Ensure mapper produces safe values:

- [x] first-class honors gold >= 3.75
- [x] first-class honors >= 3.50
- [x] second-class honors >= 3.25
- [x] transferred-course grade condition >= B or S where applicable
- [x] institution study fraction >= 2/3 where applicable
- [x] unsafe OCR `3.51` rejected
- [x] unsafe OCR grade `8` rejected unless source-scoped verification resolves it

## Source verification metadata

For values repaired using visual verification, attach something equivalent to:

```json
{
  "verification_status": "source_verified",
  "source_rule_id": "rule:27.2.2"
}
```

Do not create a generic OCR repair rule.

## R1 tests

- [x] existing 12 categories still behave identically
- [x] mapper emits 16 categories
- [x] missing required evidence => `present = null`
- [x] registration produces 9 / 22 / 27 / 9 correctly
- [x] graduation GPA is `at_least 2.00`
- [x] honors gives 3.75 / 3.50 / 3.25
- [x] `3.51` is not emitted as a factual value
- [x] source provenance retained
- [x] supporting rule text retained
- [x] `33.11` source reference is not rewritten

## R1 gate

```text
focused mapper tests PASS
git diff --check PASS
no unrelated files changed
```

R1 COMPLETE / FROZEN — 28/28 focused tests PASS; final audit PASS.

STOP after R1.

---

# 4. R2 — Program-Specific Requirement Extraction

**Smallest sufficient model/mode:** `Luna + Build`

## Goal

Extract total program credits from the 4 curriculum-page images independently of the institution rule parser.

## New component

Suggested:

```text
src/pipeline/tools/extraction/program_requirements.py
tests/tools/test_program_requirements.py
```

Optional source config:

```text
config/rule_sources.yaml
```

Config may identify source files/pages, but must not contain production truth values.

Example:

```yaml
AIT:
  source_file: ait_page_005.png
  source_page: 5

BIT:
  source_file: bit_page_006.png
  source_page: 6

DSBA:
  source_file: dsba_page_006.png
  source_page: 6

IT:
  source_file: it_page_006.png
  source_page: 6
```

## Extraction contract

Anchor on the heading:

```text
จำนวนหน่วยกิตที่เรียนตลอดหลักสูตร
```

Then extract the bounded nearby credit value.

Output shape:

```json
{
  "program": "IT",
  "requirement_type": "total_program_credits",
  "operator": "=",
  "value": 129,
  "unit": "credits",
  "source_provenance": [
    {
      "source_filename": "it_page_006.png",
      "source_page": 6,
      "document_page": 1,
      "document_category": "program_requirement"
    }
  ]
}
```

## Safety

- [x] zero candidates => fail closed
- [x] multiple plausible candidates => fail closed
- [x] no prefix-based program inference
- [x] do not use instructor GT as fallback truth

## R2 expected values

```text
AIT   120
BIT   126
DSBA  132
IT    129
```

## R2 gate

```text
4/4 program requirements extracted correctly
provenance correct
focused tests PASS
git diff --check PASS
```

R2 COMPLETE / FROZEN — 5/5 focused tests PASS; 4/4 real-image smoke PASS; audit PASS.

STOP after R2.

---

# 5. R3 — Proper Rules Pipeline Entry Point

**Smallest sufficient model/mode:** `Luna + Build`

## Goal

Create a supported production pipeline for rules without pretending `RULE` is a curriculum program.

## Do not do

```text
python -m src.pipeline.run --program rule
```

Do not add `RULE` to `SUPPORTED_PROGRAMS`.

## Preferred entry point

Suggested:

```text
src/pipeline/run_rules.py
```

Run:

```powershell
python -m src.pipeline.run_rules
```

## Flow

```text
13 institution regulation PNG
  -> OCR
  -> RuleExtractor
  -> RulesPolicyMapper
  -> institution_policy.json

4 program curriculum PNG
  -> OCR
  -> ProgramRequirementExtractor
  -> program_requirements.json
```

## Output layout

```text
data/output/
├── rules_extracted.json
└── final/
    ├── institution_policy.json
    └── program_requirements.json
```

Do not create a new nested hierarchy unless a real requirement appears.

## Minimal useful flags

- [x] `--no-gpu`
- [x] `--skip-ocr`
- [x] `--dry-run`
- [x] `--output-dir`

## R3 tests

- [x] dry-run shows correct source set
- [x] only 13 `rule_page_*` files go through `RuleExtractor`
- [x] 4 program pages go through `ProgramRequirementExtractor`
- [x] output paths are deterministic
- [x] rerun is deterministic
- [x] unrelated curriculum pipeline behavior unchanged

## R3 gate

```text
run_rules dry-run PASS
real run PASS
canonical outputs generated
focused tests PASS
```

Status: [DONE] — real run PASS, focused tests 37/37 PASS, final audit PASS.

R3 COMPLETE / FROZEN — 4/4 R3 orchestration tests PASS; 37/37 combined R1-R3 focused tests PASS; final audit PASS.

STOP after R3.

---

# 6. R4 — Runtime SQLite Integration

**Smallest sufficient model/mode:** `Luna + Build`

## Goal

Load institution policy + program requirements into `cucumber_outputs/runtime/curriculum.db`.

## Existing limitation

Current loader accepts provenance categories only:

```text
plan
description
unknown
```

Add support for:

```text
rule
program_requirement
```

## Suggested minimal schema

```text
regulation_rules
policy_facts
policy_fact_provenance

program_requirements
program_requirement_provenance
```

Avoid building an oversized legal ontology.

### `regulation_rules`

Preserve extracted legal structure/text:

```text
rule_id
section_number
parent_rule_id
category
rule_text
references_json
```

### `policy_facts`

Normalized deterministic facts:

```text
fact_id
category
fact_key
operator
value
unit
condition
source_rule_id
verification_status
```

Examples:

```text
registration.regular_max
<=
22
credits
rule:11
extracted_verified
```

```text
honors.first_class_min_gpa
>=
3.50
GPA
rule:27.2.2
source_verified
```

### `program_requirements`

```text
requirement_id
program_code
requirement_type
operator
value
unit
```

## Loader contract

- [x] curriculum corrected JSON remains existing source family
- [x] `institution_policy.json` loaded explicitly as supplemental canonical data
- [x] `program_requirements.json` loaded explicitly
- [x] never feed rule JSON through the curriculum-course loader accidentally
- [x] GT never enters runtime loader

## Build-index consideration

Current default source discovery focuses on:

```text
data/output/final/*_corrected.json
```

Update build orchestration explicitly; do not broaden a glob in a way that treats rule JSON as a curriculum plan document.

## R4 tests

- [x] existing curriculum DB counts remain valid
- [x] new rule tables populated
- [x] 4 program requirement rows available
- [x] provenance references source pages correctly
- [x] unsupported provenance still fails closed
- [x] rebuild deterministic
- [x] old curriculum QA-focused DB tests still pass

## R4 gate

```text
DB rebuild PASS
rule rows present
program requirement rows present
curriculum regression focused tests PASS
```

R4 COMPLETE / FROZEN — 47 focused curriculum/R4 tests PASS; DB rebuild PASS; final audit PASS.

STOP after R4.

---

# 7. R5 — Standalone Policy QA

**Smallest sufficient model/mode:** `Luna + Build`

## Goal

Answer deterministic policy questions without modifying the frozen curriculum EvidencePlan.

## Preferred separation

Suggested:

```text
rag/policy/
├── __init__.py
├── query.py
├── repository.py
└── answer.py
```

Do not add policy primitives to the existing six curriculum evidence primitives yet.

## First supported question set

### Registration

- [x] `ปกติลงทะเบียนได้สูงสุดกี่หน่วยกิต`
- [x] `ขั้นต่ำกี่หน่วยกิต`
- [x] `ลง 24 หน่วยกิตได้ไหม`
- [x] `กรณีพิเศษลงได้สูงสุดเท่าไร`
- [x] `ซัมเมอร์ลงได้กี่หน่วยกิต`

### Probation

- [x] `GPA เท่าไรถึงติดโปร`
- [x] `GPA เท่าไรถึงพ้นโปร`

### Program total credits

- [x] `IT ต้องเรียนกี่หน่วยกิต`
- [x] `AIT ต้องเรียนกี่หน่วยกิต`
- [x] `BIT ต้องเรียนกี่หน่วยกิต`
- [x] `DSBA ต้องเรียนกี่หน่วยกิต`

### Honors

- [x] `เกียรตินิยมอันดับหนึ่ง GPA เท่าไร`
- [x] `เกียรตินิยมอันดับสอง GPA เท่าไร`

### Other

- [x] `กลับเข้าศึกษาได้ภายในกี่ปี`

## LLM budget

Easy policy factual query:

```text
0 LLM calls
```

Use deterministic answer rendering when possible.

## Answer contract

Every answer should carry:

- [x] factual value
- [x] condition/operator
- [x] applicable scope
- [x] source rule / source document
- [x] provenance
- [x] no unsupported extrapolation

## R5 gate

```text
deterministic policy QA PASS
no Gemini/API required for easy policy facts
provenance visible
fail-closed cases tested
```

R5 COMPLETE / FROZEN — 9 focused tests PASS; real-data smoke PASS; 0 LLM/API calls; final audit PASS.

STOP after R5.

---

# 8. R6 — Combined Curriculum + Regulation QA

**Smallest sufficient model/mode for design:** `Luna + Plan`

Use `Terra` only if a genuine cross-system architecture tradeoff appears.

## Goal

Answer questions requiring both course/curriculum evidence and institution policy evidence.

Example:

```text
IT ปี 3 เทอม 1 มี 21 หน่วยกิต
ถ้าลงเพิ่มอีก 3 หน่วยกิตได้ไหม
```

Required reasoning:

```text
curriculum evidence
  -> current load = 21

policy evidence
  -> normal maximum = 22
  -> exception maximum = 27
  -> exception conditions

deterministic calculation
  -> 21 + 3 = 24
  -> 24 > 22
  -> normal rule not satisfied
  -> exception may apply only under its documented conditions
```

## Important

- [x] LLM must not compute the authoritative numeric result
- [x] LLM may synthesize wording only after grounded facts exist
- [x] no hidden assumption that exception conditions are satisfied
- [x] ambiguous student-specific conditions => explain conditional result
- [x] preserve authority boundaries

## Possible integration only after design review

At this point evaluate whether to:

- extend `EvidencePlan` with policy primitives, or
- create a policy sub-plan/composed evidence result

Do not choose before reviewing the actual seam.

## R6 gate

- [x] cross-source query works
- [x] deterministic arithmetic
- [x] policy conditions preserved
- [x] no curriculum regression
- [x] provenance from both authority families retained

R6 COMPLETE / FROZEN — 19/19 focused/regression tests PASS; real cross-source smoke PASS; deterministic arithmetic PASS; dual provenance PASS; final gate audit PASS. The illustrative IT Y3S1=21 example conflicts with current canonical curriculum evidence (36), so canonical evidence correctly wins and the query fails closed (INFO, not a gate failure).

STOP after R6.

---

# 9. R7 — Evaluation and Documentation

**Smallest sufficient model/mode:** `Muse Spark 1.3 Free`

## Focused test set

```text
tests/tools/test_rule_extractor.py
tests/tools/test_rules_policy_mapper.py
tests/tools/test_program_requirements.py
tests/rag/policy/...
```

Plus focused curriculum regression tests around any touched seam.

Do not run the entire historical suite after every micro-task.

## Final validation

- [x] RuleExtractor structural tests pass
- [x] 16 semantic categories represented
- [x] AIT/BIT/DSBA/IT program totals correct
- [x] canonical final Rule files generated
- [x] DB loads regulation data
- [x] standalone policy QA works
- [x] combined QA works
- [x] provenance end-to-end
- [x] fail-closed cases verified
- [x] no LLM factual authority
- [x] curriculum QA regression gate passes
- [x] `git diff --check` passes
- [x] README/docs updated only after implementation is stable (`docs/academic_rules.md`)

R7 COMPLETE / FROZEN — focused validation PASS (R0–R2 55/55, R3 4/4,
R4 5/5, R5/R6 19/19, curriculum regression 12/12); real end-to-end
smoke PASS (10/10); provenance/authority audit PASS; final Rules
audit PASS with no BLOCKER and no SHOULD_FIX.

---

# 10. Ground Truth Policy

## `ground_truth/rules_ground_truth.json`

Treat as:

```text
instructor taxonomy / evaluation seed
```

Not:

```text
production source of truth
```

Known problems include incorrectly directed comparison operators.

Do not silently modify this file during implementation.

If a corrected evaluation GT is needed, create a separately audited artifact after official-source verification.

## `ground_truth/rules_extraction_eval.json`

Keep as:

```text
RuleExtractor regression subset
```

It is not full semantic truth.

---

# 11. Canonical Final Authority

After R3/R4, expected hierarchy:

```text
Official Institution Regulation
    -> data/output/final/institution_policy.json
    -> runtime regulation/policy tables

Official Program Curriculum Pages
    -> data/output/final/program_requirements.json
    -> runtime program requirement tables
```

Curriculum course facts continue to use:

```text
data/output/final/*_corrected.json
```

The two authority families must remain distinguishable.

---

# 12. Recommended Working Method Per Phase

For every phase:

1. Inspect only the smallest required files.
2. Confirm current behavior with focused tests.
3. Implement one bounded seam.
4. Add focused regression tests.
5. Run `git diff --check`.
6. Report:
   - files changed
   - behavior/contracts added
   - focused test count/result
   - blockers
7. STOP.
8. User reviews and commits manually.

Do not bundle R1–R7 into one coding-agent task.

---

# 13. Progress Tracker

```text
R0  Source acquisition / OCR / structural extraction   [DONE]
R1  Complete semantic policy mapper                    [DONE]
R2  Program total-credit extractor                     [DONE]
R3  Rules pipeline entry                               [DONE]
R4  SQLite/runtime integration                         [DONE]
R5  Standalone Policy QA                               [DONE]
R6  Combined Curriculum + Regulation QA                [DONE]
R7  Evaluation + documentation                         [DONE]
```

---

## Immediate Next Task

### Rules system COMPLETE — R0–R7 DONE / FROZEN

No NEXT Rules milestone. Per-milestone details in §§0–9 above;
final behavior and evidence boundaries in `docs/academic_rules.md`.

Any future work (Query Planner, judgement framework, UI) is a new
project scope — do not reopen frozen R0–R7 behavior to chase scores.
