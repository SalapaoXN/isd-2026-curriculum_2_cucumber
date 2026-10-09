# Reports — รายงานผลการประเมิน

โฟลเดอร์นี้เก็บ **ผลการประเมินและหลักฐานการทดสอบย้อนหลัง** ของ CUCUMBER ไม่ใช่เอกสารที่ใช้อธิบายความสามารถล่าสุดของ Semantic QA โดยตรง

ควรอ่านไฟล์ในโฟลเดอร์นี้ดังนี้:

- `README.md` — สรุปผลประเมินคุณภาพข้อมูลชุดปัจจุบัน
- `runtime_benchmark.md` — ผลวัดเวลาในการทำงาน ณ วันที่รัน benchmark
- `final_automated_hardening.md` — snapshot การตรวจระบบวันที่ 2026-10-08 เก็บไว้เป็นหลักฐานย้อนหลัง รายการ failure/manual gate ในนั้นไม่ถือเป็น TODO ปัจจุบันโดยอัตโนมัติ
- ถ้าต้องการดูสถาปัตยกรรมและความสามารถล่าสุดของ Semantic QA ให้อ่าน `../README.md`, `../docs/semantic-qa-vnext.md` และ `../rag/README.md`
- รายงานย้อนหลังไม่ควรถูกแก้ตัวเลขย้อนหลังเพียงเพื่อให้ตรงกับ code รุ่นใหม่

## 1. ผลประเมินข้อมูลหลักสูตรปัจจุบัน

รายงานชุดนี้สร้างใหม่เมื่อ **2026-10-04** จากไฟล์หลักสูตรที่ผ่านการตรวจแก้แล้ว และมี Ground Truth ที่ตรงกับฉบับหลักสูตรนั้นโดยตรง

ประเมินทั้งหมด 8 ขอบเขต:

- AIT
- BIT / coop
- BIT / no_coop
- DSBA / coop
- DSBA / no_coop
- GENED / gened
- IT / coop
- IT / no_coop

### ความครบถ้วนของจำนวน record

| ตัวชี้วัด | ผลลัพธ์ |
| --- | ---: |
| จำนวน record ใน Ground Truth | 839 |
| จำนวน record ที่ระบบสร้าง | 839 |
| จำนวน record ที่จับคู่ได้ | 839 |
| ขาด (Missing) | 0 |
| เกิน (Extra) | 0 |
| Precision | 100% |
| Recall | 100% |
| F1 | 100% |

**หมายเหตุ:** ค่า 100% ด้านบนหมายถึง **จำนวน record ครบและจับคู่ได้ครบ** เท่านั้น ไม่ได้หมายความว่าข้อความทุก field ถูกต้อง 100%

### คุณภาพของแต่ละ field ใน 839 record ที่จับคู่ได้

| Field | ความถูกต้องระดับตัวอักษร | ความถูกต้องระดับคำ |
| --- | ---: | ---: |
| code | 100.00% | N/A |
| name_th | 99.29% | 98.35% |
| name_en | 99.36% | 95.58% |
| credits | 99.59% | N/A |
| prerequisite | 99.98% | 99.90% |

รายละเอียดรายการที่ยังไม่ตรงอยู่ใน `evaluation_errors.csv`

### การแก้ให้การ rebuild ฐานข้อมูลมีความสอดคล้อง

การสร้าง runtime DB ใหม่ถูกทดสอบกับ corrected files ปัจจุบันทั้งชุดแล้ว โดยมีการแก้สำคัญดังนี้:

- เติม `catalog_key = bit-2565` ให้ข้อมูล BIT 2565 ทั้งแผน `coop` และ `no_coop`
- ทำให้ข้อเท็จจริงของวิชาที่ใช้ร่วมกันระหว่างสองแผน BIT 2565 ตรงกันก่อน merge
- ถ้า credit เป็นค่าว่าง จะถือว่าเป็น “ไม่มีข้อมูล” แทน empty string เพื่อลด false conflict; ถ้าอีกแผนของวิชาเดียวกันมีค่าที่ตรวจสอบได้ ระบบจึงสามารถใช้ค่าร่วมกันได้
- เพิ่ม regression test ที่โหลด corrected files จริงทั้งหมดร่วมกับ `program_requirements.json`

เส้นทาง rebuild แบบปกติผ่าน end-to-end test ด้วย deterministic test embeddings และ `program_requirements` ของ `bit-2565` สามารถระบุ `catalog_key` ได้เพียงหนึ่งค่าตาม contract

### รายการข้อมูลที่แก้หลังตรวจหลักฐาน

แก้เฉพาะรายการที่มีหลักฐานรองรับชัดเจน:

- IT `06066302`: ชื่อไทย → `การเขียนโปรแกรมเว็บพื้นฐาน`
- IT `06016465`: ชื่อไทย → `การออกแบบศูนย์ข้อมูล`
- GENED `90642045`: `BE MV BEV.` → `BE MY BEV.`

ชุดทดสอบ correction, fail-closed credit, canonical assertion และ evaluation regression ผ่านทั้งหมด

ส่วน BIT `06036135` ยังเว้นค่า credits ไว้โดยตั้งใจ เพราะยังไม่มีหลักฐานจาก source ฝั่ง production ที่เพียงพอให้เติมค่าจาก Ground Truth เข้าไป

## 2. ขอบเขตของการประเมิน

`data/output/final/` มีไฟล์หลายฉบับหลักสูตร รวมถึงฉบับเก่า เช่น 2560 และ 2557

แต่ Ground Truth ปัจจุบันยังไม่ได้แยกตามทุกฉบับหลักสูตร โดยส่วนใหญ่มีเพียง Ground Truth ที่ยอมรับแล้วหนึ่งชุดต่อ program/plan

ดังนั้นรายงานนี้ใช้เฉพาะ **8 คู่ prediction / Ground Truth ที่เป็นฉบับเดียวกันจริง** และจะไม่เอาข้อมูลของหลักสูตรฉบับเก่ามาเทียบกับ Ground Truth ของอีกฉบับ

ตัวเลขในรายงานนี้จึงหมายถึง:

> คุณภาพของชุดข้อมูลที่มี Ground Truth ตรงฉบับและถูกนำมาประเมิน

ไม่ใช่:

> ความถูกต้องของทุก curriculum edition ที่มีอยู่ใน runtime DB

ถ้าต้องการประเมินฉบับเก่าเพิ่มเติม ต้องสร้างและตรวจ Ground Truth ของแต่ละ edition ก่อน

## 3. วิธีสร้างรายงานชุดนี้ใหม่

ใช้คู่ไฟล์ที่กำหนดชัดเจนเพื่อป้องกันการนำข้อมูลคนละฉบับมาเทียบกัน:

```powershell
python -m src.pipeline.tools.evaluation.evaluate `
  --pair data/output/final/merged_ait_no_plan_full_corrected.json ground_truth/AIT/AIT_academic_plan.json `
  --pair data/output/final/merged_bit_coop_full_corrected.json ground_truth/BIT/BIT_academic_plan_coop.json `
  --pair data/output/final/merged_bit_no_coop_full_corrected.json ground_truth/BIT/BIT_academic_plan_no_coop.json `
  --pair data/output/final/merged_dsba_coop_full_corrected.json ground_truth/DSBA/DSBA_academic_plan_coop.json `
  --pair data/output/final/merged_dsba_no_coop_full_corrected.json ground_truth/DSBA/DSBA_academic_plan_no_coop.json `
  --pair data/output/final/merged_gened_gened_edition-gened-2564_full_corrected.json ground_truth/general_education_ground_truth.json `
  --pair data/output/final/merged_it_coop_full_corrected.json ground_truth/IT/IT_academic_plan_coop.json `
  --pair data/output/final/merged_it_no_coop_full_corrected.json ground_truth/IT/IT_academic_plan_no_coop.json
```

ผลลัพธ์จะถูกเขียนไว้ใน:

```text
reports/evaluation/
├── evaluation_summary.csv
├── field_metrics.csv
├── evaluation_errors.csv
└── evaluation.json
```

- `evaluation_summary.csv` — สรุปความครบถ้วนของ record
- `field_metrics.csv` — ค่า CER/WER และความถูกต้องของแต่ละ field
- `evaluation_errors.csv` — รายการค่าที่ไม่ตรงกับ Ground Truth
- `evaluation.json` — รายละเอียดผลประเมินแบบเต็ม

## 4. ผลประเมินเก่าสำหรับใช้อ้างอิง

```text
reports/evaluation_reference/
```

โฟลเดอร์นี้เป็น snapshot เก่าสำหรับเปรียบเทียบเท่านั้น ไม่ใช่ผลปัจจุบัน และ evaluator ปกติไม่ควรเขียนทับ

## 5. การวัดความเร็วของระบบ (Runtime benchmark)

รายงานอยู่ที่:

```text
reports/runtime_benchmark.md
```

benchmark นี้แยกเวลาในการทำงานของส่วน local deterministic ออกจากเวลาที่เสียไปกับ external model

รันด้วย:

```powershell
.\.venv\Scripts\python.exe scripts/benchmark_runtime.py --runs 5
```

snapshot ล่าสุดแสดงว่างาน parsing และ SQLite ใช้เวลาน้อยมากเมื่อเทียบกับเส้นทาง API ที่ต้องเรียกโมเดลภายนอก

## 6. ข้อเตือนเรื่องแหล่งข้อเท็จจริง

รายงานและ Ground Truth เป็น **ข้อมูลสำหรับประเมินผล** ไม่ใช่แหล่งข้อเท็จจริงที่ production QA ใช้ตอบคำถาม

เส้นทาง production คือ:

```text
ข้อมูลหลักสูตร / policy ที่ผ่านการตรวจแล้ว
→ runtime SQLite
→ QA ที่ตอบจากหลักฐาน
```

ประวัติการแก้ Ground Truth ดูที่ `ground_truth/GT_FIXED.md`
