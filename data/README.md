# data/

โครงข้อมูลของ pipeline (clean rebuild)

- `input/` — วางภาพต้นฉบับตรงนี้ โดยจัดตามโครงเดิมของ `inputs/`:
  `input/<program>/<program>_page_NNN.png`
  (ait, bit, dsba, gened, it, rule) ไฟล์ภาพต้นฉบับชุดเดิมไม่ได้
  ผูกมากับ repo (ไฟล์ใหญ่) ถ้าจะรัน OCR ให้คัดลอกเข้ามาก่อน
- `output/` — พื้นที่ทำงานของ pipeline โดยปกติไฟล์ระหว่างทาง
  จะอยู่ใน memory/temp แล้วลบทิ้ง (`--keep-intermediates` จะเก็บ
  `extracted/` + `consolidated/` ไว้เพื่อ debug)
- `output/final/` — **ผลลัพธ์หลักที่ RAG ใช้ได้จริง**
  (`*_corrected.json` + `*_corrections.json`) ซึ่งมาแทนที่
  `outputs/llm/` แบบเดิม `rag.build_index` อ่านเฉพาะ directory นี้

สำเนาอ้างอิงของผลลัพธ์เดิมเก็บไว้ใต้ `submission/` กับ
`reports/evaluation_reference/` เพื่อใช้เทียบพฤติกรรม

## ลำดับชั้นข้อมูลและสิทธิ์ความเป็นจริง (อ่านจากบนลงล่าง)
1. `output/final/*_corrected.json` — ชุดข้อมูลหลัก (corpus)
   ที่เป็นมาตรฐาน เป็น input เพียงอย่างเดียวของ
   `rag.build_index` และเป็นข้อเท็จจริงตั้งต้นของระบบ QA
2. `cucumber_outputs/runtime/curriculum.db` — ฐานข้อมูล SQLite
   ตอนรัน (runtime) ที่สร้างจากข้อ 1 พร้อม semantic index
   ฝั่ง QA อ่านอย่างเดียว ห้ามแก้ด้วยมือ ถ้าแก้ข้อ 1 ให้สร้าง
   ใหม่ด้วยคำสั่ง `python -m rag.build_index`
3. `output/ocr/`, `extracted/`, `consolidated/` — ไฟล์ระหว่าง
   ทางสำหรับ debug ไม่ใช่ input ของ QA
4. `input/` และ `inputs/` เดิม — ภาพต้นฉบับสำหรับ (รัน) OCR
   ใหม่ ไฟล์ภาพขนาดใหญ่ไม่ได้ผูกมาทั้งหมด ต้องคัดลอกเข้ามา
   ก่อนถ้าจะสร้างข้อมูลใหม่
