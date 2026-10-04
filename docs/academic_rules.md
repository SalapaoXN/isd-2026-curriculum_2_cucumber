# Academic Rules / Policy Data

เอกสารนี้อธิบายสายข้อมูลข้อกำหนดสถาบันที่ใช้ร่วมกับ curriculum QA ปัจจุบัน

## Flow

```text
rule source pages
→ OCR
→ RuleExtractor
→ RulesPolicyMapper
→ institution_policy.json
→ supplemental loader
→ runtime SQLite
→ policy QA
```

ข้อกำหนดหน่วยกิตรวมของแต่ละฉบับหลักสูตรถูกเก็บแยกใน:

```text
data/output/final/program_requirements.json
```

และโหลดเข้าฐานข้อมูลด้วย `catalog_key` เพื่อไม่ให้ค่าจากคนละ edition ปนกัน

## Canonical supplemental files

- `data/output/rules_extracted.json` — extraction intermediate
- `data/output/final/institution_policy.json` — normalized policy facts + provenance
- `data/output/final/program_requirements.json` — edition-scoped total-credit requirements

ค่า total program credits ที่มีอยู่ในไฟล์ปัจจุบัน:

| catalog_key | program | total credits |
| --- | --- | ---: |
| `ait-2566` | AIT | 120 |
| `bit-2560` | BIT | 126 |
| `bit-2565` | BIT | 126 |
| `dsba-2560` | DSBA | 126 |
| `dsba-2565` | DSBA | 132 |
| `it-2560` | IT | 130 |
| `it-2565` | IT | 129 |

GENED ไม่มี total-program-credit record ในชุดนี้

## Authority boundaries

ข้อมูลสามครอบครัวต้องแยกกัน:

| Family | Authority |
| --- | --- |
| Curriculum / placement | canonical corrected curriculum JSON → SQLite |
| Institution policy | `institution_policy.json` → policy tables |
| Program total credits | `program_requirements.json` → program requirement tables |

`ground_truth/` ใช้ประเมินผลเท่านั้น ไม่ใช่ fallback ของ production

## Runtime tables

supplemental loader เติมข้อมูล เช่น:

- `regulation_rules`
- `policy_facts`
- `policy_fact_provenance`
- `program_requirements`
- `program_requirement_provenance`

ทุก factual record ต้องมี provenance ที่ตรวจสอบได้

## QA behavior

policy QA ใช้ deterministic parsing / lookup / arithmetic สำหรับคำถามที่รองรับ เช่น:

- หน่วยกิตขั้นต่ำ/สูงสุดในการลงทะเบียน
- เงื่อนไขบางประเภทตามข้อบังคับ
- เกียรตินิยม / probation / re-entry ที่มี fact รองรับ
- ลาพักการศึกษา — ดึงข้อ 31.1–31.4 จาก `regulation_rules`
- การลาออก — ดึงข้อ 32 จาก `regulation_rules`
- การเทียบโอนหน่วยกิต — ดึงข้อ 28–29 จาก `regulation_rules`
- การทุจริตในการสอบ — ดึงข้อ 20 แบบ bounded text-backed answer
- บทลงโทษทางวินัย — ดึงข้อ 38–39 และข้อย่อยที่ระบุประเภทโทษ
- การอุทธรณ์คำสั่งลงโทษ — รองรับขั้นตอนจากข้อ 43 และ deadline 30 วันจาก structured fact ที่ผูกกับข้อ 43
- total program credits ตาม edition
- semester-load comparison ที่มี curriculum + policy evidence ครบ

หากข้อมูลไม่พอ, scope ไม่ชัด, หรือเป็นข้อมูลเฉพาะบุคคลที่ canonical evidence ไม่ยืนยัน ระบบต้อง fail closed

## Rebuild

rules data:

```powershell
python -m src.pipeline.run_rules --dry-run
python -m src.pipeline.run_rules
python -m src.pipeline.run_rules --skip-ocr
```

runtime DB:

```powershell
python -m rag.build_index
```

## Focused tests

```powershell
python -m unittest tests.tools.test_rule_extractor tests.tools.test_rules_policy_mapper tests.tools.test_program_requirements
python -m unittest tests.pipeline.test_run_rules
python -m unittest tests.rag.test_rag_policy tests.rag.test_rag_policy_combined
```

คำถามแบบ text-backed ด้านลาพัก/ลาออก/เทียบโอน ใช้ bounded deterministic routing และส่งข้อความกฎจาก canonical `regulation_rules` พร้อม provenance โดยตรง ไม่เรียก LLM เพื่อสร้างข้อเท็จจริง

ข้อจำกัดหลัก: policy QA ตอบได้เฉพาะ fact/rule family ที่มี route รองรับและอยู่ใน authority files ปัจจุบัน; ระบบไม่อนุมานกฎที่ไม่มีหลักฐาน
