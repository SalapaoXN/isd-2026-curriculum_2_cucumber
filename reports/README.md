# Reports

โฟลเดอร์ `reports/` ใช้เก็บผลการประเมินคุณภาพข้อมูลหลักสูตรหลังผ่าน pipeline แล้ว

โครงสร้างปัจจุบัน:

```text
reports/
├── README.md
├── evaluation/
│   ├── evaluation.json
│   ├── evaluation_summary.csv
│   ├── field_metrics.csv
│   └── evaluation_errors.csv
└── evaluation_reference/
    ├── evaluation.json
    ├── evaluation_summary.csv
    ├── field_metrics.csv
    └── evaluation_errors.csv
```

---

## 1. Current Evaluation

ผลประเมินปัจจุบันอยู่ใน:

```text
reports/evaluation/
```

ข้อมูลที่ใช้ประเมินมาจาก canonical corrected data:

```text
data/output/final/*_corrected.json
```

และเปรียบเทียบกับ Ground Truth ใน:

```text
ground_truth/
```

Evaluator ปัจจุบันรองรับทั้งหมด 8 program/plan scopes:

```text
AIT

BIT
├── coop
└── no_coop

DSBA
├── coop
└── no_coop

GENED
└── gened

IT
├── coop
└── no_coop
```

สร้าง report ใหม่ด้วยคำสั่ง:

```powershell
python -m src.pipeline.tools.evaluation.evaluate
```

ถ้าไม่ระบุ argument ระบบจะค้นหาไฟล์:

```text
data/output/final/*_corrected.json
```

จากนั้นจับคู่กับ Ground Truth ของแต่ละ program/plan และเขียนผลใหม่ลง:

```text
reports/evaluation/
```

---

## 2. evaluation_summary.csv

ไฟล์:

```text
reports/evaluation/evaluation_summary.csv
```

ใช้ดูว่า record รายวิชาของแต่ละ program/plan ถูกสร้างมาครบหรือไม่

column หลัก:

```text
program
plan
gt_total
pred_total
tp
fn
fp
precision
recall
f1
precision_percent
recall_percent
f1_percent
```

ความหมาย:

- `gt_total` = จำนวน record ใน Ground Truth
- `pred_total` = จำนวน record ที่ pipeline สร้าง
- `tp` = record ที่จับคู่กับ Ground Truth ได้
- `fn` = record ที่ Ground Truth มี แต่ prediction ไม่มี
- `fp` = record ที่ prediction มีเกินจาก Ground Truth
- `precision` = สัดส่วน prediction ที่ถูกต้อง
- `recall` = สัดส่วน Ground Truth ที่ระบบเก็บได้ครบ
- `f1` = harmonic mean ของ precision และ recall

สำหรับ current corpus ไฟล์นี้ควรมีครบ 8 scopes:

```text
AIT
BIT,coop
BIT,no_coop
DSBA,coop
DSBA,no_coop
GENED,gened
IT,coop
IT,no_coop
```

---

## 3. field_metrics.csv

ไฟล์:

```text
reports/evaluation/field_metrics.csv
```

ใช้วัดคุณภาพของข้อมูลราย field หลังจากจับคู่ record ได้แล้ว

field หลักที่ประเมิน เช่น:

```text
code
name_th
name_en
credits
prerequisite
```

column หลัก:

```text
program
plan
field
sample_count
cer
character_accuracy_percent
wer
word_accuracy_percent
```

### CER

`CER` หรือ Character Error Rate ใช้วัดความผิดพลาดระดับตัวอักษร

ค่าต่ำกว่า = ดีกว่า

ตัวอย่าง:

```text
CER = 0.00
```

หมายถึงข้อความตรงกับ Ground Truth ระดับตัวอักษร

### Character Accuracy

คำนวณจาก:

```text
1 - CER
```

แล้วแสดงเป็นเปอร์เซ็นต์

ตัวอย่าง:

```text
character_accuracy_percent = 100.0
```

### WER

`WER` หรือ Word Error Rate ใช้วัดความผิดพลาดระดับคำ

ระบบใช้ tokenization ตามชนิดของ field

เช่น field ภาษาไทย:

```text
name_th
desc_th
```

จะใช้ Thai tokenizer

ส่วนภาษาอังกฤษและ prerequisite ใช้ whitespace tokenization

field ที่ไม่เหมาะกับ WER เช่น:

```text
code
credits
year
semester
```

จะไม่คำนวณ WER

---

## 4. evaluation_errors.csv

ไฟล์:

```text
reports/evaluation/evaluation_errors.csv
```

ใช้ดูรายละเอียดของ record หรือ field ที่ไม่ตรงกับ Ground Truth

column หลัก:

```text
program
plan
alignment_status
code
field
gt_value
pred_value
cer
wer
```

`alignment_status` มีไว้บอกลักษณะของความผิดพลาด เช่น:

```text
matched
missing
extra
```

ความหมาย:

- `matched` = พบ record ทั้งสองฝั่ง แต่ค่าบาง field ไม่ตรง
- `missing` = Ground Truth มี แต่ prediction ไม่มี
- `extra` = prediction มี record ที่ Ground Truth ไม่มี

ไฟล์นี้เหมาะกับการ debug ว่าความผิดพลาดเกิดกับรายวิชาใดและ field ใด

---

## 5. evaluation.json

ไฟล์:

```text
reports/evaluation/evaluation.json
```

เป็นผล evaluation แบบเต็มในรูป JSON

เก็บรายละเอียดมากกว่า CSV เช่น:

```text
coverage
field_level
category_level
rubric
prediction_view
page_level
```

ตัวอย่างข้อมูลที่อยู่ในแต่ละ result:

```text
program
plan
file_name
total_gt_courses
total_pred_courses
matched_courses
coverage
field_level
category_level
rubric
```

ไฟล์นี้เหมาะกับ:

- ตรวจผลแบบละเอียด
- ใช้ใน script
- วิเคราะห์ผลเพิ่มเติม
- ตรวจ regression ระหว่างรอบ

---

## 6. Coverage

Coverage ใช้วัดว่าระบบดึง record รายวิชามาครบหรือไม่

ตัวอย่าง:

```text
gt_record_count = 63
prediction_record_count = 63
matched_count = 63
missing_gt_count = 0
extra_prediction_count = 0

precision = 1.0
recall = 1.0
f1 = 1.0
```

หมายถึง record ใน scope นั้นครบตรงกับ Ground Truth ทั้งหมด

Coverage ไม่ได้ใช้วัดว่าชื่อวิชาหรือข้อความสะกดถูกหรือไม่

คุณภาพข้อความให้ดู CER/WER ใน:

```text
field_metrics.csv
```

---

## 7. Category-level Evaluation

ใน `evaluation.json` ยังมีการแบ่งผลตาม category ของหลักสูตร เช่น:

```text
หมวดวิชาเฉพาะ
หมวดวิชาศึกษาทั่วไป
หมวดวิชาเสรี
```

แต่ละ category จะมี:

```text
text_quality
coverage
```

ทำให้ดูได้ว่า error กระจุกตัวอยู่ในหมวดใดหรือไม่

---

## 8. Page-level Evaluation

ระบบรองรับการประเมินระดับ source page เมื่อ Ground Truth มี provenance ที่เพียงพอ

ถ้า Ground Truth ไม่มี authoritative source/page provenance จะได้สถานะ:

```text
unavailable
```

พร้อมเหตุผล เช่น:

```text
Ground truth records do not contain authoritative source/page provenance.
```

สถานะนี้ไม่ได้แปลว่า pipeline ผิด แต่หมายถึง Ground Truth ชุดนั้นไม่มีข้อมูลพอสำหรับวัดระดับหน้า

---

## 9. evaluation_reference

โฟลเดอร์:

```text
reports/evaluation_reference/
```

เป็น historical/reference snapshot ที่เก็บไว้สำหรับเปรียบเทียบกับผลในอดีต

ประกอบด้วย:

```text
evaluation.json
evaluation_summary.csv
field_metrics.csv
evaluation_errors.csv
```

ไฟล์ในโฟลเดอร์นี้ไม่ใช่ current evaluation

ผลปัจจุบันให้ดูที่:

```text
reports/evaluation/
```

ไม่ควรเขียนทับ `evaluation_reference/` เมื่อรัน evaluator ปกติ

---

## 10. Source of Truth

ลำดับข้อมูลของ evaluation คือ:

```text
data/output/final/*_corrected.json
        +
ground_truth/
        ↓
src.pipeline.tools.evaluation.evaluate
        ↓
reports/evaluation/
```

ดังนั้น `reports/` เป็นเพียงผลการประเมิน

ไม่ใช่ factual source ของระบบถาม-ตอบ

RAG ใช้ canonical corrected data โดยตรง:

```text
data/output/final/*_corrected.json
        ↓
python -m rag.build_index
        ↓
cucumber_outputs/runtime/curriculum.db
```

---

## 11. วิธี Regenerate Report

รันจาก root ของ repository:

```powershell
python -m src.pipeline.tools.evaluation.evaluate
```

หลังรันเสร็จควรได้:

```text
reports/evaluation/
├── evaluation.json
├── evaluation_summary.csv
├── field_metrics.csv
└── evaluation_errors.csv
```

ตรวจว่า evaluation ครบทุก scope:

```powershell
Get-Content reports/evaluation/evaluation_summary.csv
```

ควรมี:

```text
AIT
BIT,coop
BIT,no_coop
DSBA,coop
DSBA,no_coop
GENED,gened
IT,coop
IT,no_coop
```

---

## 12. ไฟล์ไหนควรอ่านก่อน

ถ้าต้องการดูภาพรวมเร็วที่สุด:

```text
1. evaluation/evaluation_summary.csv
2. evaluation/field_metrics.csv
3. evaluation/evaluation_errors.csv
4. evaluation/evaluation.json
```

ลำดับการอ่าน:

```text
Coverage
→ Field Accuracy
→ Error Rows
→ Full JSON Detail
```

---

## 13. Current vs Reference

สรุป:

```text
reports/evaluation/
= ผลปัจจุบันจาก canonical data

reports/evaluation_reference/
= snapshot เก่าที่เก็บไว้สำหรับเปรียบเทียบ
```

เวลาแก้ข้อมูลใน:

```text
data/output/final/
```

แล้วต้องการอัปเดตผล ให้รัน evaluator ใหม่เพื่อ regenerate:

```text
reports/evaluation/
```

โดยไม่ต้องแก้ `evaluation_reference/`
