# scripts/ — ทางเข้าของผู้ใช้และ developer

## ask.py — CLI สำหรับผู้ใช้
ถามหลักสูตรผ่าน RAG pipeline ตัวจริง แล้วพิมพ์คำตอบพร้อม
แหล่งอ้างอิงที่จัดกลุ่มแล้ว

```powershell
python scripts/ask.py "IT ปี 1 เทอม 1 มีวิชาอะไรบ้าง"
python scripts/ask.py
```

โหมดถามต่อเนื่องไม่เก็บ state ฝั่ง server แต่ละครั้งเป็นอิสระ
ออกจากกัน (เลิกด้วย `exit` / `quit`) CLI สร้าง Gemini
provider ตั้งแต่เริ่มโปรแกรม จึงต้องมี `GEMINI_API_KEY`
แม้คำถามที่ไม่ต้องใช้โมเดลก็ตาม

## evaluate_gold_questions.py — ชุดประเมิน Gold
รันคำถาม Gold ผ่าน RAG pipeline แล้วเก็บผลไว้ให้คะแนน
ใช้เพื่อประเมินผลเท่านั้น โดยอาศัย helper สาย legacy
(route/answer) ประกอบการวินิจฉัย

## verify_curriculum.py — ตัวตรวจ submission
ตรวจ `submission/curriculum.json` เทียบกับ
`submission/schema/curriculum.schema.json` แบบ deterministic
(รูปเลขรหัสวิชา การอ่านหน่วยกิต placeholder
คีย์ provenance) แล้วเขียน `submission/verify.json`

```powershell
python scripts/verify_curriculum.py
```

## nl_robust_audit.py — ตัวช่วยตรวจ parser/seam ภายใน
เครื่องมือ audit ชั่วคราวเฉพาะเครื่องนี้ ตรวจ deterministic
parser กับ SQL fallback seam แบบมี guard ใช้ประเมินผลเท่านั้น
อ่าน production code อย่างเดียว ไม่แก้ไขอะไร รันจาก `scripts/`

```powershell
python nl_robust_audit.py <fixture.json> <curriculum.db>
```

ผลลัพธ์เป็นบรรทัด ASCII (หนึ่ง JSON object ต่อคำถามบวกสรุป)
ไฟล์นี้ผูก path ของเครื่องไว้ เก็บไว้ใช้ภายใน ไม่ต้อง commit

## หมายเหตุ
- `build_conversion_report.py` กับ
  `generate_semantic_threshold_dev.py` เป็นตัวสร้างรายงาน/
  dev-set แบบใช้ครั้งคราว ไม่ใช่ flow ปกติ
- ห้าม commit API key `.env` เก็บไว้เฉพาะเครื่อง
