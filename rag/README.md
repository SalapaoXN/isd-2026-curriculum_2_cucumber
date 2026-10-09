# rag/ — โครงสร้างระบบถาม-ตอบหลักสูตรปัจจุบัน

โฟลเดอร์ `rag/` มีทั้งโค้ดของระบบ QA รุ่นเดิม และ **Semantic QA ที่ใช้เป็นเส้นทางหลักในปัจจุบัน**

## เส้นทางการทำงานปัจจุบัน

เมื่อกำหนด `CUCUMBER_QA_MODE=semantic` ระบบจะทำงานตามลำดับนี้:

```text
คำถามของผู้ใช้
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
→ response ของ API
```

`rag/semantic/modes.py` ทำหน้าที่เชื่อม Semantic QA เข้ากับรูปแบบ API เดิม

โมดูลรุ่นเดิม เช่น `query_spec.py`, `resolution.py`, `evidence_planner.py`, `evidence_executor.py`, `grounded_answer.py` ยังถูกใช้เป็นเครื่องมือ deterministic ที่ผ่านการทดสอบแล้ว และยังรองรับ legacy mode อยู่ แต่ใน semantic mode **ประโยคดิบของนักศึกษาจะไม่ถูกส่งกลับไปให้ legacy parser ตีความภาษาอีกครั้ง**

## แบ่งหน้าที่ว่าอะไรเชื่อถือได้แค่ไหน

```text
LLM                = เสนอความหมายของภาษาเท่านั้น
resolver            = ระบุตัวตนและขอบเขตจากข้อมูลจริง
SQLite / evidence   = แหล่งข้อเท็จจริง
VerifiedResult      = ขอบเขตของข้อมูลที่อนุญาตให้นำไปตอบ
provenance          = หลักฐานสำหรับตรวจสอบย้อนกลับ
answerer / renderer = เรียบเรียงคำตอบเท่านั้น
```

SQLite และ evidence ที่ตรวจสอบ provenance ได้ยังเป็นข้อเท็จจริงที่ QA อ่านตอนตอบคำถาม. Accepted teacher course-plan Ground Truth ใช้แบบ offline สอง stage: เป็น reference สำหรับประเมิน raw reviewed finals ก่อน แล้วจึงใช้ canonicalize เฉพาะ structured fields ที่ยอมรับแล้ว. QA และ rag.build_index ไม่อ่าน GT โดยตรง.

## Curriculum runtime source

- data/output/final/*_final.json คือ reviewed prediction ก่อน GT override และเป็น input ของ evaluation ไม่ใช่ default curriculum source ของ RAG.
- Current GT-backed curricula ใช้ accepted teacher GT หลัง evaluation เฉพาะ fields ที่มีใน GT: code, name_th, name_en, credits, year, semester, category, type, prerequisite, flexible_year_semester และ note; descriptions และ provenance ยังคงมาจาก reviewed finals.
- Runtime curriculum source คือ data/output/canonical/*_final.json; build ต้องผ่าน preflight ก่อน.
- Legacy 2560 ไม่มี accepted GT edition เดียวกัน จึงไม่ evaluate เทียบ current GT และใช้เฉพาะ source-verified corrections จาก data/corrections/legacy_2560_source_verified_corrections.json เมื่อจำเป็น.
- institution_policy.json และ program_requirements.json ยังคงโหลดจาก data/output/final/ เป็น supplemental sources.
- Runtime rules ใช้ source-verified rules pipeline/correction artifacts. ground_truth/rules_ground_truth.json และ ground_truth/rules_extraction_eval.json ใช้สำหรับ evaluation เท่านั้น.

การสร้างซ้ำจาก reviewed finals ทำได้โดยไม่รัน OCR ใหม่ ดูคำสั่งและขอบเขต evaluation ที่ ../src/pipeline/README.md และ ../reports/README.md.

## โมดูลสำคัญของ Semantic QA

- `semantic/schema.py` — กำหนดรูปแบบของ intent, resolved intent และ verified result
- `semantic/prompts.py` — prompt ของ interpreter v15 และ answerer v1
- `semantic/interpreter.py` — แปลงภาษาผู้ใช้เป็นโครงสร้างตาม schema
- `semantic/validation.py` — ตรวจว่าคำถามและโครงสร้างที่ตีความมารองรับจริงหรือไม่
- `semantic/context.py` — รวมบริบทการสนทนาแบบจำกัดขอบเขต
- `semantic/resolver.py` — ระบุ program, `catalog_key`, plan และ course จากข้อมูล canonical
- `semantic/compiler.py` — แปลง resolved structure เป็นคำสั่งที่ระบบใช้ทำงานต่อ
- `semantic/planner.py` — เลือกว่าจะใช้ deterministic path, policy, guarded SQL หรือหยุดเป็น `unsupported`
- `semantic/executor.py` — ดึงหลักฐานจริงและประกอบผลลัพธ์ที่ซับซ้อน
- `semantic/answerer.py` — สร้างคำตอบจากข้อมูลที่ตรวจสอบแล้ว
- `semantic/pipeline.py` — จุดรวมของ Semantic QA ตั้งแต่ต้นจนจบ พร้อม trace
- `semantic/modes.py` — จัดการโหมด legacy / shadow / semantic
- `semantic/trace.py` — เก็บข้อมูลสำหรับตรวจสอบว่าแต่ละขั้นทำอะไร

## ความสามารถที่รองรับ

### ข้อมูลรายวิชาโดยตรง

รองรับรหัสวิชา ชื่อ คำอธิบาย หน่วยกิต ช่วงปี/เทอม และวิชาบังคับก่อน รวมถึงการถามหลาย field ในคำถามเดียว

ถ้าระบบยอมรับ field ใดจากคำถามแล้ว field นั้นต้องถูกใช้จริงในขั้นตอนถัดไป หรือระบบต้องหยุดแบบ fail closed ห้ามรับมาแล้วละทิ้งเงียบ ๆ

### รายการวิชาและผลรวม

รองรับ:

- รายชื่อวิชาตามขอบเขต
- จำนวนวิชา
- ผลรวมหน่วยกิตตามปี/เทอม/แผน
- semantic topic discovery จากคำอธิบายรายวิชา
- หน่วยกิตรวมทั้งหลักสูตรจาก `program_requirements`

### หลายวิชาที่ผู้ใช้ระบุชัดเจน

ใช้ `target.kind="literal_set"` เมื่อผู้ใช้ระบุหลายวิชา เช่น วิชา A, B และ C ระบบต้อง resolve ทุกวิชาแยกกันและครบทั้งหมด ถ้าหาไม่เจอแม้แต่หนึ่งวิชาจะไม่ตอบเฉพาะส่วนที่เหลือ

### กลุ่มวิชาเลือก

ระบบตรวจสมาชิกของกลุ่มและจำนวนวิชาที่ต้องเลือกจากข้อมูล canonical จริง ไม่ให้ LLM สร้างจำนวนหรือ group ID ขึ้นเอง

### คำถามที่มีหลายขอบเขต (Mixed scope)

คำถามหนึ่งข้อสามารถถามข้อมูลเฉพาะรายวิชา พร้อมถามผลรวมของเทอมที่รายวิชานั้นอยู่ได้ ระบบจะคำนวณแต่ละส่วนแยกกันก่อน แล้วจึงรวมคำตอบเมื่อหลักฐานครบ

### เปรียบเทียบช่วงเรียนระหว่างแผน

รองรับการเปรียบเทียบ **ชุดปี/เทอมที่เรียนได้ทั้งหมด** ของแต่ละวิชาในแต่ละแผน ทั้งกรณีผู้ใช้ระบุชื่อแผนเอง และกรณีใช้ `available_plans` เพื่อดึงรายการแผนที่มีอยู่จริงมาจากฐานข้อมูล

`earliest_placement` คำนวณจากข้อมูลปี/เทอมจริง ไม่ให้ LLM ตัดสินว่าแผนใดเร็วกว่าเอง

### เรียงลำดับรายวิชา (`placement_sequence`)

สำหรับคำถามที่ต้องการเรียงหลายวิชาตามปี/เทอม:

- placement ของทุกวิชาต้องตรวจสอบครบก่อน
- วิชาบังคับก่อนต้องผูกกับวิชาเป้าหมายที่ถูกต้อง
- เรียงจากค่า `(year, semester)` ที่ยืนยันแล้ว
- ไม่สรุปว่า “เรียนก่อน” เท่ากับ “เป็น prerequisite”
- ถ้ามีวิชาอยู่เทอมเดียวกันหรือช่วงเวลาซ้อนกันจนพิสูจน์ลำดับเดียวไม่ได้ ระบบจะ fail closed
- `result_courses` จะเรียงตามลำดับเดียวกับคำตอบ เพื่อให้คำถามต่อ เช่น “วิชาที่ 2” อ้างถึงวิชาถูกตัว

## บริบทการสนทนา

Web/API จะส่ง `next_context` กลับมาเพื่อใช้ในเทิร์นถัดไป โดยเก็บเฉพาะข้อมูลโครงสร้างที่จำเป็น เช่น:

- program / `catalog_key` / plan / ปี / เทอม
- วิชาที่กำลังพูดถึง
- รายการวิชาจากผลลัพธ์ก่อนหน้า
- reference ของ operation ก่อนหน้าบางส่วน

Context นี้ **ไม่ใช่ chat transcript memory** และไม่เก็บข้อความคำตอบเก่าไว้เป็นข้อเท็จจริง

ถ้าผู้ใช้ระบุ scope ใหม่ในเทิร์นปัจจุบัน ข้อมูลใหม่นั้นต้องมีสิทธิ์เหนือ context เก่า

## เส้นทางกฎและข้อกำหนดของสถาบัน

คำถามเกี่ยวกับกฎของสถาบันและ `program_requirements` ใช้แหล่งข้อมูล canonical เพิ่มเติมที่เตรียมไว้โดยเฉพาะ และแยกการทำงานออกจากข้อเท็จจริงของรายวิชา

ดูรายละเอียดที่ `docs/academic_rules.md`

## กรณีที่ต้องหยุดแทนการเดา (Fail closed)

ระบบควรไม่ตอบหรือขอข้อมูลเพิ่มเมื่อ:

- ระบุ identity หรือ scope ไม่ได้ชัดเจน
- ฉบับหลักสูตรหรือแผนยังไม่ชัดเจนในคำถามที่จำเป็นต้องใช้
- evidence หรือ provenance ไม่ครบ
- มีวิชาใน `literal_set` ที่หาไม่เจอ
- คำถามมี field หรือเงื่อนไขที่ระบบรับแล้วแต่ยังไม่มีทาง execute ครบ
- placement sequence ไม่มีลำดับเดียวที่พิสูจน์ได้
- policy ไม่มี threshold ที่จำเป็นต่อคำตอบ

หลักการคือ **ตอบไม่ครบยังดีกว่าตอบข้อเท็จจริงผิด** ดังนั้นเมื่อหลักฐานไม่พอ ระบบจะหยุดแทนการคาดเดา

## วิธีรัน Semantic backend

```powershell
$env:CUCUMBER_QA_MODE="semantic"
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

รันชุดทดสอบทั้งหมด:

```powershell
python -m unittest discover -s tests -t .
```

snapshot ล่าสุดก่อนปรับเอกสารรอบนี้:

- ชุดทดสอบแบบไม่พึ่ง provider ภายนอก: **2,804 tests / 0 failures / 3 skipped**
- G5-C placement sequence: **35/35 ผ่าน**
- focused semantic regression หลังแก้ retained-order: **140/140 ผ่าน**

## ชุดทดสอบและรายงานเก่า

`tests/rag/test_final_core_eval.py` และ legacy robustness fixtures ยังมีประโยชน์สำหรับตรวจ regression ของระบบเดิม แต่ไม่ได้ครอบคลุมความสามารถ Semantic QA ทั้งหมดที่มีในปัจจุบัน

รายงานที่มีวันที่ใน `eval/results/` เป็น snapshot ของแต่ละ checkpoint หากพบ failure ในรายงานเก่า ต้อง reproduce บน code ปัจจุบันก่อนจึงจะถือว่าเป็น bug ปัจจุบัน

## สิ่งที่ตั้งใจยังไม่ทำ

- การไล่ prerequisite แบบหลายทอดโดยอัตโนมัติ
- การใช้ SQL แบบอิสระเป็นแหล่งข้อเท็จจริง
- memory ข้าม session แบบไม่จำกัด
- การทำนายว่ารายวิชาจะเปิดสอนจริงในอนาคตหรือไม่
- การตัดสินสิทธิ์ส่วนบุคคลโดยไม่มีเกณฑ์ canonical ครบ
- การรื้อ architecture ครั้งใหญ่ในช่วงปิดงาน
