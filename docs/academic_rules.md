# Academic Rules System (R0–R7, frozen)

Deterministic, source-backed handling of institution regulations alongside
curriculum data. Factual authority is always canonical runtime evidence —
never the LLM, never evaluation ground truth, never user-stated values.

## 1. Architecture

```text
13 rule_page_*.png + 4 program pages (*_page_00[56].png)
  -> OCR (cached intermediates)
  -> RuleExtractor (155 structured rules, rules_extracted.json)
  -> RulesPolicyMapper (16 categories -> institution_policy.json)
  -> ProgramRequirementExtractor (4 totals -> program_requirements.json)
  -> supplemental loader (explicit, validated) -> runtime SQLite
  -> rag/policy standalone QA + combined curriculum+policy QA
```

Entry point: `python -m src.pipeline.run_rules`
(`--dry-run`, `--skip-ocr`, `--no-gpu`, `--output-dir`).
`RULE` is intentionally **not** in `SUPPORTED_PROGRAMS`.

## 2. Source inputs

- `data/input/rule/rule_page_001.png` … `rule_page_013.png` (institution regulation)
- `data/input/rule/{ait,bit,dsba,it}_page_00[56].png` (program curriculum pages)
- OCR cached intermediates reused by `--skip-ocr` reruns (deterministic SHA-256 outputs)

## 3. Canonical generated outputs

- `data/output/rules_extracted.json` (extraction intermediate, 155 rules)
- `data/output/final/institution_policy.json` (16 policy categories + provenance)
- `data/output/final/program_requirements.json` (AIT 120 / BIT 126 / DSBA 132 / IT 129 credits)

## 4. Runtime SQLite tables (`cucumber_outputs/runtime/curriculum.db`)

Curriculum family (unchanged behavior): `programs`, `courses`,
`curriculum_plans`, `plan_placements`, plus provenance/chunk tables.
Rules family (supplemental, loaded explicitly by path, never globbed):

- `regulation_rules` — extracted legal structure/text per rule
- `policy_facts` — normalized deterministic facts (`fact_key`, operator,
  value, unit, `source_rule_id`, `verification_status`)
- `policy_fact_provenance` — rule-page provenance per fact
- `program_requirements` + `program_requirement_provenance`

Reloads are deterministic; unsupported provenance fails the load closed.

## 5. Authority families (must stay separate)

| Family | Source of truth | Provenance category |
|---|---|---|
| Curriculum / plan | corrected plan JSON -> `plan_placements` etc. | `plan`, `description` |
| Institution regulation | `institution_policy.json` -> `policy_facts` | `rule` |
| Program totals | `program_requirements.json` -> `program_requirements` | `program_requirement` |

`ground_truth/rules_ground_truth.json` is an instructor evaluation seed
only. Nothing in `rag/` or the pipeline loads it; production never falls
back to it, and GT conflicts resolve in favor of canonical evidence.
Rule/program-requirement JSON can never load through the curriculum-course
loader (enforced + tested).

## 6. Standalone Policy QA (`rag/policy`, `answer_policy_question`)

Bounded question parser (`query.py`) -> canonical-DB fact fetch
(`repository.py`) -> deterministic render (`answer.py`).
Supported: registration min / regular max / exception max (27) /
special max (9), 24-credit-style comparisons, probation entry/cleared,
honors first/second, re-entry limit, per-program total credits.
Statuses: `complete` | `unsupported` | `insufficient_evidence` |
`invalid_query`. Factual answers require **0 LLM/API calls**
(`rag/policy` imports no providers).

## 7. Combined Curriculum + Regulation QA (`answer_combined_question`)

`semester_load_plus_credits` family only: canonical semester load (from
`plan_placements`) + added credits -> deterministic decision:

- `within_normal_limit` (total <= 22)
- `conditional_exception` (22 < total <= 27, exception eligibility
  **never assumed** — answer states the condition)
- `exceeds_exception_maximum` (total > 27)
- `insufficient_evidence` for conflicting user-stated loads, missing
  scope, student-specific eligibility claims, or missing policy facts

Arithmetic is plain deterministic Python; the LLM (if any) only
re-words already-grounded facts.

## 8. Supported question boundaries

Standalone: the §6 list. Combined: semester-load-plus-credits with one
resolvable program + plan + year + semester. Anything else
(dormitories, fees, unknown programs, multi-program text, free-form
legal advice) returns `unsupported` / `insufficient_evidence` by design.

## 9. Fail-closed behavior

Missing/ambiguous evidence, unknown programs, conflicting user premises,
student-specific exception claims, and absent policy facts all return
`unsupported` or `insufficient_evidence` — never a guessed value.
Canonical evidence wins over conflicting user-stated values.

## 10. Provenance model

Every fact carries `source_rule_id` + page-level provenance
(`source_filename`, `source_page`, `document_category`) from extraction
through mapping, loading, and answer rendering. Combined answers expose
both families separately (`curriculum_provenance` vs
`policy_provenance`). OCR repairs are narrow and rule-scoped with
`verification_status: source_verified` (rule 25.1 GPA 2.00; rule 27.2.2
3.50 and B-or-S; rule 43 / 51.2 appeal 30-day deadlines). Unsafe tokens
(`3.51`, `8`, `OO`-only numerics) never become production values.

## 11. Deterministic arithmetic / zero-LLM factual path

Policy comparisons and combined totals use exact numeric comparison in
code. No embedding, retrieval-score, or LLM output participates in the
factual result. Tests assert exact decisions and totals.

## 12. Known limitations

- Grade table rows destroyed by OCR (B/C/D/F in rule 19.3) stay unmapped;
  only verifiable A / C+ rows are emitted.
- Appeal coverage is the four source-backed deadlines (43: 30d, 48: 15
  working days, 49: 7 working days, 51.2: 30d); other appeal prose has
  no extractable numeric fact.
- `honors_first` returns both gold (3.75) and first-class (3.50) facts.
- The R6 doc example "IT Y3S1 = 21 credits" is illustrative, not
  canonical truth (canonical load differs); conflicting statements fail
  closed — this is correct behavior, not a bug.
- Thai rendering in some consoles may show Mojibake; stored UTF-8 data
  is unaffected.

## 13. How to rebuild rules data

```powershell
python -m src.pipeline.run_rules --dry-run   # verify 13 + 4 source set
python -m src.pipeline.run_rules              # full run (GPU optional)
python -m src.pipeline.run_rules --skip-ocr   # rerun from cached OCR
```

## 14. How to rebuild the runtime DB

Default build loads curriculum corrected JSONs plus both supplemental
files explicitly (see `rag/build_index.py::_default_supplemental_sources`):

```powershell
python -m rag.build_index
```

## 15. How to run focused Rules tests

```powershell
python -m unittest tests.tools.test_rule_extractor tests.tools.test_rules_policy_mapper tests.tools.test_program_requirements
python -m unittest tests.pipeline.test_run_rules
python -m unittest tests.rag.test_rag_supplemental_loader
python -m unittest tests.rag.test_rag_policy tests.rag.test_rag_policy_combined
python -m unittest tests.rag.test_rag_index   # curriculum regression gate
```

## 16. How to run representative QA smoke

```powershell
$env:PYTHONPATH = '<repo-root>'
python -c "from pathlib import Path; from rag.policy import answer_policy_question as q; db = Path('cucumber_outputs/runtime/curriculum.db'); print(q(db, 'ภาคปกติลงทะเบียนได้สูงสุดกี่หน่วยกิต').rendered_answer); print(q(db, 'ลงทะเบียน 24 หน่วยกิตได้ไหม').rendered_answer); print(q(db, 'IT ต้องเรียนกี่หน่วยกิต').rendered_answer)"
python -c "from pathlib import Path; from rag.policy import answer_combined_question as c; db = Path('cucumber_outputs/runtime/curriculum.db'); print(c(db, 'IT แผนไม่สหกิจ ปี 1 เทอม 1 มี 18 หน่วยกิต ถ้าลงเพิ่มอีก 3 หน่วยกิตได้ไหม').decision)"
```
