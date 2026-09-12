# AGENTS.md

## Purpose

This repository is the **CUCUMBER curriculum QA / RAG** project for 06026240 Intelligent System Development.

The system turns curriculum documents into structured curriculum data and uses that data to answer **student-facing questions about studying in the curriculum**.

Primary principle:

> Prefer the smallest change that satisfies the current task.  
> Do not redesign working architecture unless the task proves a real blocker.

---

## Repository

Expected repository root:

`D:\Year3\1\ISD\isd-2026-curriculum_2_cucumber`

Before doing any task:

1. Confirm the working directory is the repository root above.
2. Run `git status --short`.
3. Read only the files required for the current task.
4. State the smallest intended change before editing.
5. Stop if required files are missing instead of guessing.

Important submission files should exist under:

`submission/`

---

## Working Style

- Work in **small, scoped tasks**.
- Prefer targeted `grep`, `find`, file reads, and focused tests.
- Do not broadly explore the repository unless necessary.
- Do not spawn exploratory subagents unless explicitly requested.
- Do not refactor unrelated code.
- Do not change architecture to satisfy an artificial metric or internal taxonomy.
- If a task is review/audit only, do not edit files.
- If evidence is insufficient, report the blocker instead of inferring facts.

For shell commands:

- Do not chain logically separate commands with `;`.
- Put separate steps in separate commands.
- Keep one long command on one physical line.

---

## Git Policy

Unless explicitly requested:

- Do **not** `git add`.
- Do **not** `git commit`.
- Do **not** `git push`.
- Do **not** create, delete, or switch branches.
- Do **not** modify branch refs.
- Do **not** revert unrelated working-tree changes.

Allowed:

- `git status --short`
- `git diff`
- `git diff --check`

The user reviews and performs commits/pushes manually.

---

## Project Scope

The end user is a university student asking questions that should be answerable from the curriculum source.

Questions should sound like natural Thai student questions, for example:

- `วิชา 06016420 ชื่ออะไรและมีกี่หน่วยกิต?`
- `วิชา 06016414 เรียนเกี่ยวกับอะไรบ้าง?`
- `ปี 2 เทอม 2 แบบสหกิจต้องเรียนอะไรบ้าง?`
- `ก่อนลงวิชา 06016420 ต้องผ่านวิชาอะไร?`
- `ถ้าติดวิชาบังคับก่อน จะกระทบแผนเรียนต่อยังไง?`

Avoid database/developer wording such as:

- schema field names
- internal IDs
- route names
- implementation-specific terms

Do not create Gold questions about instructor, classroom, timetable, tuition, or similar data merely because the current schema lacks those fields.

**Absence from extracted schema is not proof that the curriculum source does not contain a fact.**

---

## Current Submission Scope

Submission scope is:

- Program: **IT**
- Plans:
  - `coop`
  - `no_coop`

Do not expand the submission to other programs unless explicitly requested.

The broader repository may still contain data for other programs.

---

## Canonical Data and Runtime

Canonical curriculum source:

`outputs/consolidated/`

Unified runtime DB:

`cucumber_outputs/runtime/curriculum.db`

Submission DB:

`submission/curriculum.db`

Core architecture:

```text
curriculum source
    ↓
canonical structured JSON
    ↓
SQLite relational data + semantic chunks/vectors
    ↓
question
    ↓
structured SQL and/or semantic retrieval
    ↓
retrieved evidence
    ↓
grounded answer
```

`structured`, `semantic`, `hybrid`, and `unknown` are **internal diagnostic/routing labels**.

They are not the primary basis for designing student Gold questions.

---

## Stable Data Rules

Treat these as established behavior unless the current task proves otherwise:

- Course codes are valid only when exactly 8 ASCII digits.
- Do not "correct" valid `96...` prefixes to `06...`.
- Course code alone is not globally unique across catalogs.
- One catalog course can have multiple plan placements.
- Preserve repeated placements when they are meaningful.
- Missing/null/non-string codes must not crash processing.
- Flexible timing must remain explicit.
- Numeric credit value and raw credit string are separate concepts.
- Alternative course groups must not be double-counted.
- Preserve provenance:
  - source page
  - document page when truly available
  - source filename/document key
- Do not fabricate document-page values.
- Do not leak raw absolute paths into generated artifacts.
- Do not patch the SQLite DB merely to hide an upstream data problem.
- If source extraction is wrong, fix the source/canonical data and rebuild.

---

## Lab 8B Principles

Lab 8B flow:

```text
Markdown / extracted text
    ↓
JSON
    ↓
validation + repair
    ↓
SQLite
    ↓
Thai question
    ↓
query/retrieval
    ↓
grounded answer
```

Important principles:

- Schema first, extraction second.
- Validation passing does **not** prove the data is true.
- Prompt is a request; code is enforcement.
- Deterministic checks should enforce facts that must always be correct.
- When validation warns, distinguish:
  1. extraction/data is wrong
  2. the source genuinely contains an exception
- Repair loops must be bounded.
- After repeated repair failure, escalate to human review.

---

## Lab 8B Verification Rules

The current verifier uses these seven checks:

- `CHK1` plan total credits = declared credits
- `CHK2` every course code in the plan has a course description
- `CHK3` course code is exactly 8 numeric digits
- `CHK4` plan credits = description credits
- `CHK5` prerequisite appears in an earlier valid study position
- `CHK6` no duplicate course in the same semester
- `CHK7` semester credits are normally between 9–22

Known exception classes:

- alternative course group → count once where appropriate
- cooperative education / internship semester → may contain one 6-credit course
- summer semester → low credits can be valid
- co-requisite → may be in the same semester

Do not weaken verification rules merely to suppress warnings.

---

## Gold Question Specification

Final Gold set:

- **30 questions total**
- **Easy: 15**
- **Medium: 10**
- **Hard: 5**
- At least **2 safe in-domain not-found cases**

This 15/10/5 distribution is the project decision used for the 30-question Lab 8B Gold set.

### Easy

Direct fact or direct topic lookup from the curriculum.

Examples:

- course name
- course credits
- course placement
- `วิชา X เรียนเกี่ยวกับอะไร?`

### Medium

Requires connecting, filtering, aggregating, or checking at least two curriculum facts.

Examples:

- semester course list
- semester credit total
- prerequisite question
- alternative-course rule
- flexible placement
- plan-specific filtering

### Hard

Requires genuine multi-step reasoning, study planning, comparison, or consequence analysis across several curriculum facts.

A question is **not Hard** merely because it combines:

- placement + description
- two direct lookup facts

Hard questions should resemble realistic student planning scenarios.

---

## Not-Found Rules

At least two Gold questions must correctly produce the exact fallback:

`ไม่พบข้อมูลนี้ในเล่มหลักสูตร`

Use safe in-domain negative cases, such as demonstrably missing IT course codes.

Do not use out-of-domain questions merely to force the fallback.

Do not infer global absence from a missing schema field.

---

## Evidence and Grounding Rules

Every answer must be grounded in curriculum evidence.

Never invent unsupported curriculum facts.

When verifying a proposed Gold question:

1. Verify every expected fact against current IT data.
2. Verify provenance/source page.
3. Flag any fact that is:
   - inferred
   - broadened
   - normalized
   - silently corrected
   - unsupported
4. Preserve source/OCR wording when uncertain.
5. Do not silently repair source wording in Gold expectations.

For semantic/topic questions, verify concepts against the actual description text.

---

## Evaluation Contract

Primary evaluation concepts:

- `execution_success`
- `answer_correct`
- `provenance_correct`
- `evidence_correct`
- `not_found_correct`
- `latency_sec`

Diagnostic fields may include:

- `actual_route`
- `route_match`
- `model_retry_count`
- `earliest_failure_stage`
- raw SQL/query details
- error message

Internal route mismatch alone must not make an otherwise correct grounded answer fail.

### Answer correctness

Structured facts should be checked deterministically where possible:

- course code
- course name
- credits
- year
- semester
- prerequisite
- counts/totals

Semantic answers should be checked by concept/fact coverage rather than exact English phrase matching.

Do not fail a correct Thai paraphrase merely because wording differs.

Use `REVIEW` only when semantic equivalence genuinely cannot be determined deterministically.

### Provenance

Evaluate provenance separately from answer wording.

A final answer does not need to reproduce the provenance in an exact text format if the system already returns correct structured provenance.

### Not-found

For Gold not-found cases, exact fallback matching is allowed and expected.

### Latency

Record latency for diagnostics/bonus reporting.

Do not make latency a correctness failure unless explicitly required.

Useful aggregate reporting:

- average latency
- p95 latency
- execution success rate
- answer correctness breakdown
- strict correctness
- breakdown by Easy / Medium / Hard
- not-found exact success
- provenance coverage
- retry/error counts

---

## Project Evaluation Philosophy

The system should answer what students actually ask.

The evaluation should make it possible to distinguish:

- routing/query failure
- retrieval failure
- missing evidence
- wrong factual answer
- synthesis failure
- grading-only failure

Do not collapse all failures into one score when diagnosing the system.

For project-level accuracy, a strict correct answer should be:

- factually correct
- relevant to the question
- grounded in evidence
- supported by correct provenance

---

## Architecture Freeze Before Submission

Before the current submission deadline, do **not** change these unless a verified blocker requires it:

- DB schema
- curriculum DB design
- canonical curriculum data
- embeddings/vector model
- router architecture
- provenance architecture
- verifier architecture
- conversion/repair architecture

Do not:

- add L4/multi-version work
- add an LLM-as-a-judge
- redesign NL-to-SQL
- add new deterministic operations just to improve an internal metric
- rerun full evaluation repeatedly to select the best stochastic result

Prefer:

1. verify Gold facts
2. finalize Gold
3. minimally align evaluator
4. run focused local tests
5. run one final full evaluation
6. audit artifacts
7. freeze

---

## Required Lab 8B Submission Artifacts

The required submission artifacts are:

1. `schema/curriculum.schema.json`
2. `schema/schema.sql`
3. `curriculum.json`
4. `curriculum.conversion.json`
5. `curriculum.db`
6. `verify.json`
7. `gold_questions.json`
8. `eval_result.json`

Current submission location:

`submission/`

Do not remove or rename required artifacts without explicit instruction.

---

## Current Known Working-Tree Rule

The repository may already contain modified and untracked files from previous work.

Never assume an unrelated change is safe to revert.

Before editing:

- inspect `git status --short`
- touch only files in the current task scope
- preserve unrelated local work

---

## Model / Task Guidance

For routine review, inspection, planning, and deterministic verification:

- prefer **Luna + Plan**

For small scoped implementation/testing:

- prefer **Luna + Build**

For architecture, evaluation design, or semantic tradeoffs:

- use **Terra** when deeper reasoning is genuinely needed

For major near-final independent review:

- use **Sol High**

Do not use a stronger model merely because it is available.

---

## Stop Conditions

Stop and report instead of guessing when:

- repository root is wrong
- required files are missing
- a fact cannot be proven from current data/source
- a not-found claim depends only on a missing schema field
- a proposed Hard question is not genuinely multi-step
- a requested change would require architecture redesign outside the current task
- a task would overwrite unrelated local work

The preferred outcome is a precise blocker report, not speculative implementation.
