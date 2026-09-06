# 🚀 HOW IT WORKS: ระบบค้นหารายวิชาด้วย AI (Vector Search Pipeline)

โครงสร้างและลำดับการทำงานของระบบนี้ ถูกออกแบบมาเพื่อเปลี่ยนข้อมูลโครงสร้างหลักสูตรให้เป็นเวกเตอร์ และใช้ LLM ค้นหาความเชื่อมโยงของเนื้อหา โดยมีขั้นตอนการทำงานดังนี้:

## 1. แปลงข้อมูลดิบให้เป็นภาษาพูด (JSON to Text)

เป้าหมายคือการนำไฟล์ JSON ที่เก็บข้อมูลแบบ Key-Value มาเรียงร้อยให้เป็นประโยคภาษาคน เพื่อให้โมเดล AI เข้าใจบริบทได้ดีที่สุด (หาก Value ไหนเป็น `Null` ระบบจะข้ามข้อมูลนั้นไปโดยอัตโนมัติ)

**ตัวอย่างข้อมูลนำเข้า (JSON):**

```json
{
  "code": "06016401",
  "name_th": "คณิตศาสตร์สำหรับเทคโนโลยีสารสนเทศ",
  "name_en": "MATHEMATICS FOR INFORMATION TECHNOLOGY",
  "credits": "3(3-0-6)",
  "year": "1",
  "semester": "1",
  "category": "หมวดวิชาเฉพาะ",
  "type": "บังคับ",
  "prerequisite": "ไม่มี",
  "flexible_year_semester": null,
  "note": null
}

```

**ผลลัพธ์ที่ได้ (Text):**

> *"วิชา คณิตศาสตร์สำหรับเทคโนโลยีสารสนเทศ รหัส 060161401 เครดิต 3(3-0-6) เรียนปี 1 เทอม 1 วิชาเฉพาะ ประเภทบังคับ วิชาบังคับก่อนไม่มี"*

**ไฟล์ที่รับผิดชอบ:**

* `json_to_text.py`: ใช้แปลงข้อมูลวิชาพื้นฐาน
* `json_to_desc.py`: ใช้สำหรับวิชาที่มี Description ยาวๆ

---

## 2. แปลงข้อความเป็นตัวเลข (Text to Vector)

นำประโยคที่แปลงเสร็จแล้วมาผ่านกระบวนการ Embedding เพื่อสร้างเป็น Vector

**ไฟล์ที่รับผิดชอบ:** `prepare_data.py` (เรียกใช้ฟังก์ชัน `vector()` เพื่อรับค่า Vector และ Text กลับมา)

> **🤓 ☝️ ข้อควรรู้สำหรับนักพัฒนา:**
> * **Model:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
> * **Input Limit:** แนะนำให้ความยาวประโยคไม่เกิน 128 Tokens
> * **Output Size:** เวกเตอร์ขนาด 384 มิติ (Dimensions)
> * *Tip:* สามารถนับจำนวน Token ได้ด้วยคำสั่ง `from model_to_vector import vector_model; vector_model.count_token(text)`
> 
> 

---

## 3. เตรียมและจัดเก็บลงฐานข้อมูล (Database Storage)

นำเวกเตอร์และข้อความที่เตรียมไว้ ยัดลงฐานข้อมูล SQLite เพื่อรอการค้นหา

**ไฟล์ที่รับผิดชอบ:**

1. รัน `create_db.py`: เพื่อสร้างโครงสร้างตารางและ Schema ฐานข้อมูล (อ้างอิงโครงสร้างจากไฟล์ `schema.xlsx`)
2. รัน `write_db.py`: เพื่อเขียนข้อมูลทั้งหมดลง Database

---

## 4. ทดสอบใช้งาน (The Magic 🪄)

เมื่อข้อมูลพร้อม ก็ถึงเวลาดูเวทมนตร์การค้นหาด้วย Vector Search

**ไฟล์ที่รับผิดชอบ:** `useit.py` (หรือ `useit.db`)

* **วิธีใช้:** ป้อนคำถามที่ต้องการ
* **ผลลัพธ์ (Output):** ระบบจะพ่นรายวิชาที่เนื้อหาตรงกับคำถามมากที่สุด 5 อันดับแรกออกมา

---

## 5. ให้ LLM ตอบคำถาม (LLM Integration)

🚧 **Status:** *ยังบ่ได้ทำ LOL* (รอการเชื่อมต่อกับ LLM ตัวใหญ่เพื่อสรุปคำตอบให้ผู้ใช้)

---

## ⚠️ Known Issues & Brainstorming

### 1. Uncommon Use Case: การขอดูหน้าอ้างอิง (Metadata Reference)

* **ปัญหา:** หาก User ถามว่าข้อมูลนี้อ้างอิงมาจากหน้าไหน ระบบจะระเบิด เนื่องจากเราไม่ได้ฝัง Metadata (เช่น เลขหน้า, ชื่อไฟล์) ลงไปใน Vector ด้วย เราแค่เก็บแยกไว้ใน Database
* **วิธีแก้ (ฝั่ง Frontend):** เพิ่มปุ่ม **"ดูแหล่งอ้างอิง (Reference)"** ไว้ใต้คำถาม เมื่อผู้ใช้คลิก ระบบจะนำ ID ของวิชานั้นไป Query ดึง Metadata จาก Database มาแสดงผลให้แทน โดยไม่ต้องให้ AI เป็นคนตอบ

### 2. I have no idea: ค้นหาข้าม 3 หลักสูตรแล้วระบบช้า แก้ไงดี?

ปัญหาความช้าเกิดจากการกวาดเวกเตอร์ทุกตัวในระบบพร้อมกัน แนวทางแก้ปัญหาที่แนะนำมีดังนี้:

* **แนวทางที่ 1: ทำ Metadata Filtering (Pre-filter) ใน SQL:**
ก่อนที่จะสั่ง `MATCH` ให้ใช้ `WHERE fk_Course IN (...)` เพื่อกรองเฉพาะวิชาที่อยู่ในหลักสูตรที่ผู้ใช้สนใจก่อน วิธีนี้จะตัดเวกเตอร์ที่ไม่เกี่ยวข้องกันทิ้งไปเกินครึ่ง ทำให้ระบบค้นหาเร็วขึ้นมาก
* **แนวทางที่ 2: ให้ User ระบุเป้าหมาย:**
หน้า UI ควรมี Dropdown ให้ผู้ใช้เลือกก่อนว่าต้องการค้นหาใน **"หลักสูตรไหน"** หากผู้ใช้เลือกแค่ 1 หลักสูตร ระบบก็แค่ดึงข้อมูลเฉพาะหลักสูตรนั้นมาคำนวณ
* **แนวทางที่ 3: แยก Virtual Table:**
หากข้อมูลใหญ่มากจริงๆ การสร้าง Virtual Table  (เช่น `vector_curriculum_1`, `vector_curriculum_2`) แล้วแยก Query ตามหลักสูตร จะทำงานได้เร็วกว่าการจับรวมกันในตารางเดียว


# Work_together
1. Must install
    `pip install sqlite-vec transformers python-dotenv numpy pandas`
    ⚠️ ข้อควรระวัง library pytorch หรือ torch ที่เลือกไม่เข้ากับเครื่องของตัวเองทำให้มีปัญหาตามมา
    วิธีแก้ใช้คำสั่ง nvidia-smi ที่ command promp ดูตัวเลขตรงมุมขวา (ประมาณนี้ CUDA UMD Version: 13.4)
    แล้วไปที่ https://pytorch.org/get-started/locally/ ตรงหัวข้อ Start Locally
    ให้สนใจที่ Compute Platform ให้เลือกเลข CUDA ที่มากที่สุดแต่น้อยกว่า เลข CUDA UMD Version
    แล้ว copy ช่อง Run this Command แล้วนำไป run ตามปกติ
    ทดสอบความสำเร็จ ใช้คำสั่ง
    ```
    import torch
    print(torch.__version__) #ควรมีเลข version ออกมา เช่น 2.14.0
    print(torch.cuda.is_available()) # ควรออก TRUE ถ้าออก False แปลว่า จะได้ใช้แค่ CPU 
    print(torch.cuda.get_device_name(0)) # ชื่อ GPU Device
    ```

2. Follow this
    1. แก้ไฟล์ prepare_data.py ตรงตัวแปร your_path ให้ใส path ของ folder json
    รันไฟล์เหล่านี้ตามลำดับ
    2. create_db.py
    3. model_to_vector.py
    4. write_db.py
