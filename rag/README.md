# rag/ — QA architecture (current)

`rag/` มีทั้ง legacy QA machinery และ **Semantic QA production path** ปัจจุบัน

## Current production path

เมื่อ `CUCUMBER_QA_MODE=semantic`:

```text
question
→ rag.semantic.interpreter
→ SemanticIntent
→ rag.semantic.validation
→ rag.semantic.context
→ rag.semantic.resolver
→ ResolvedIntent
→ rag.semantic.planner
→ rag.semantic.executor
→ VerifiedResult + provenance
→ rag.semantic.answerer
→ public API response
```

`rag/semantic/modes.py` เป็น adapter ระหว่าง semantic pipeline กับ API contract

Legacy modules เช่น `query_spec.py`, `resolution.py`, `evidence_planner.py`, `evidence_executor.py`, `grounded_answer.py` ยังมีบทบาทเป็น proven deterministic primitives และ legacy runtime support แต่ raw student wording ใน semantic mode จะไม่ถูกส่งกลับไปให้ legacy parser ทำ language understanding

## Authority boundary

```text
LLM                    = language proposal only
canonical resolver     = identity/scope authority
SQLite/evidence        = factual authority
VerifiedResult         = answerable fact boundary
provenance             = trust/audit boundary
answerer/renderer      = presentation only
```

ห้ามใช้ `ground_truth/`, evaluation fixture หรือ LLM output เป็น production fact

## Semantic modules

- `semantic/schema.py` — closed intent/resolved/verified contracts
- `semantic/prompts.py` — interpreter v15 / answerer v1 prompts
- `semantic/interpreter.py` — strict structured interpretation
- `semantic/validation.py` — grounding + supported-shape contracts
- `semantic/context.py` — bounded client-held context merge
- `semantic/resolver.py` — canonical identity/scope resolution
- `semantic/compiler.py` — resolved structure → deterministic execution specs
- `semantic/planner.py` — deterministic / policy / guarded SQL / unsupported routing
- `semantic/executor.py` — evidence execution and typed higher-order composition
- `semantic/answerer.py` — deterministic complex rendering + bounded presentation
- `semantic/pipeline.py` — end-to-end pipeline and trace
- `semantic/modes.py` — legacy/shadow/semantic API adaptation
- `semantic/trace.py` — observability

## Supported semantic contracts

### Exact course facts

รองรับ code/name/description/credits/placement/direct prerequisites และ multi-field lookup โดย accepted requested fields ต้องถูก consume downstream หรือ fail closed

### Collections and aggregates

รองรับ scoped lists/counts/credit totals, semantic topic discovery และ whole-program totals จาก `program_requirements`

### Explicit course sets

`target.kind="literal_set"` ใช้สำหรับหลายวิชาที่ผู้ใช้ระบุชัดเจน สมาชิกทุกตัวต้อง resolve แยกกันและครบทั้งหมด

### Alternative groups

ตรวจ canonical group membership และ min/max choice bounds จากฐานข้อมูล ไม่สร้าง choice count จาก LLM

### Mixed-scope composition

คำถามหนึ่งข้อสามารถรวม course-local facts กับ enclosing-term total โดยสอง scope ถูก execute แยกกันก่อนประกอบเป็นผลลัพธ์เดียว

### Placement comparison

รองรับ comparison ของ complete placement sets ข้ามแผน ทั้ง explicit plan operands และ `available_plans` selector

`earliest_placement` derivation มาจาก canonical terms ไม่ใช่ LLM judgement

### Placement sequence

สำหรับ explicit course set ที่ถามลำดับปี/เทอม:

- placement ของแต่ละวิชาต้อง verified ครบ
- direct prerequisite ownership เป็นของ target course นั้น ๆ
- sort จาก canonical `(year, semester)`
- ไม่อนุมาน prerequisite จาก chronological order
- same-term tie / overlapping ranges ที่พิสูจน์ลำดับเดียวไม่ได้จะ fail closed
- `result_courses` ถูกเรียงให้ตรงกับ sequence ที่แสดง เพื่อให้ ordinal follow-up ชี้ตัวเดียวกัน

## Conversation context

Web/API ส่ง `next_context` กลับมาในเทิร์นถัดไป Context เป็น bounded structural state เช่น:

- program / catalog_key / plan / years / semesters
- focus course
- result course identities
- limited previous-operation references

Context ไม่ใช่ transcript memory และไม่เก็บ cached factual prose เป็น authority

Explicit current-turn scope ต้องชนะ inherited context

## Policy path

Institution rules / program requirements ใช้ canonical supplemental authority และ deterministic/policy execution แยกจาก curriculum facts

ดู `docs/academic_rules.md`

## Fail-closed behavior

ระบบควร non-answer เมื่อ:

- identity/scope ambiguous
- edition/plan ไม่ชัดเมื่อจำเป็น
- evidence/provenance ไม่ครบ
- explicit course-set member หายหรือ resolve ไม่ได้
- accepted request มี field/clause ที่ downstream consume ไม่ได้
- sequence ไม่มี unique canonical order
- policy threshold ไม่มีใน authority

wrong factual answer มี priority สูงกว่า false failure; เมื่อไม่แน่ใจให้หยุดแทนการเดา

## Running

Semantic backend:

```powershell
$env:CUCUMBER_QA_MODE="semantic"
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Full tests:

```powershell
python -m unittest discover -s tests -t .
```

Closeout snapshot ก่อน refresh docs:

- full provider-isolated discovery: 2,804 tests / 0 failures / 3 skipped
- G5-C sequence: 35/35
- focused semantic regression after retained-order fix: 140/140

## Legacy and historical tests

`tests/rag/test_final_core_eval.py` และ legacy robustness fixtures ยังเป็น regression coverage ที่มีประโยชน์ แต่ไม่ใช่ตัวแทนความสามารถ semantic ทั้งหมดในปัจจุบัน

Dated reports ใต้ `eval/results/` เป็น historical snapshots ของแต่ละ checkpoint; อย่าใช้ failure list เก่าเป็น current TODO โดยไม่ reproduce บน current tree

## Non-goals / deferred

- arbitrary multi-hop prerequisite graph traversal
- arbitrary SQL as factual authority
- unrestricted cross-session memory
- future offering prediction
- personal eligibility without canonical criteria
- speculative architecture refactor during closeout
