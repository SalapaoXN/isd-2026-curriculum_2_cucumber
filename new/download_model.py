import os
from dotenv import load_dotenv

# 1. โหลดค่า HF_HOME จากไฟล์ .env เพื่อชี้ไปที่ไดรฟ์ D:
load_dotenv(r'C:\Users\TUF\OneDrive\Desktop\kmitl\ISD\new\.gitignore\.env')
print("ตำแหน่งที่บันทึกไฟล์:", os.environ.get("HF_HOME"))
'''
your file .env should be like this

HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxx
HF_HOME=D:\program
'''


# 2. Import ไลบรารีหลังจากโหลด .env แล้ว
from transformers import AutoModelForImageTextToText, AutoProcessor

model_id = "typhoon-ai/typhoon-ocr1.5-2b"

print(f"start download {model_id}")

# 3. สั่งดาวน์โหลดเฉพาะไฟล์ที่จำเป็นมาเก็บไว้ในเครื่อง (ยังไม่ได้ใช้งาน)
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(model_id)

print("done")